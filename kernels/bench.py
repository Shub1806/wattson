"""Benchmark fused residual-add + RMSNorm: PyTorch vs hand-written CUDA vs Triton.

For each implementation and tensor shape it checks correctness against a
float32 reference, measures time, and reports achieved memory bandwidth as a
percentage of the GPU's peak. Because this kernel is memory-bound, % of peak
bandwidth is the right score. Then it measures energy per call with NVML,
the same counter Wattson uses.

    python kernels/bench.py            # on a machine with an NVIDIA GPU
"""

import csv
import math
import os
import sys
import time

# Number GPUs the same way for PyTorch and NVML (matters on 2-GPU machines).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import torch.nn.functional as F
from triton.testing import do_bench

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rmsnorm_triton  # noqa: E402

EPS = 1e-6
# Peak memory bandwidth (GB/s) from NVIDIA datasheets.
PEAK_GBS = {"T4": 320, "P100": 732, "V100": 900, "A10": 600, "L4": 300, "A100": 1555}
# Rows = tokens processed together; hidden = model width (896 = Qwen2.5-0.5B, 4096 = 7B-class).
SHAPES = [(1, 896), (16, 896), (256, 896), (4096, 896), (1, 4096), (16, 4096), (256, 4096), (4096, 4096)]
ENERGY_SHAPE = (4096, 4096)


def load_cuda_extension():
    """Compiles fused_add_rmsnorm.cu with nvcc (about a minute the first time)."""
    from torch.utils.cpp_extension import load_inline
    decl = ("void fused_add_rmsnorm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, "
            "torch::Tensor out, torch::Tensor new_residual, double eps, bool vectorized);")
    return load_inline(name="wattson_fused_add_rmsnorm", cpp_sources=decl,
                       cuda_sources=open(os.path.join(HERE, "fused_add_rmsnorm.cu")).read(),
                       functions=["fused_add_rmsnorm"], extra_cuda_cflags=["-O3"])


def implementations(ext):
    def pytorch_eager(x, r, w, out, new_r):
        # What you'd write by hand: ~6 separate kernels, each a full pass over memory.
        h = x + r
        hf = h.float()
        y = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
        return y.half(), h

    def pytorch_rms_norm(x, r, w, out, new_r):
        # PyTorch's built-in RMSNorm (still a separate add kernel first).
        h = x + r
        return F.rms_norm(h, (h.shape[-1],), w, EPS), h

    def triton_fused(x, r, w, out, new_r):
        return rmsnorm_triton.fused_add_rmsnorm(x, r, w, out, new_r, EPS)

    impls = {"pytorch-eager": pytorch_eager, "pytorch-rms_norm": pytorch_rms_norm}
    if ext is not None:
        def cuda(vectorized):
            def run(x, r, w, out, new_r):
                ext.fused_add_rmsnorm(x, r, w, out, new_r, EPS, vectorized)
                return out, new_r
            return run
        impls["cuda-scalar"] = cuda(False)
        impls["cuda-half2"] = cuda(True)
    impls["triton"] = triton_fused
    return impls


def reference(x, r, w):
    h = (x.float() + r.float()).half().float()
    return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * w.float()), h


def min_bytes(rows, hidden):
    # Read x and residual, write out and new residual (2 bytes each), plus the weight once.
    return 4 * rows * hidden * 2 + hidden * 2


def main():
    assert torch.cuda.is_available(), "needs an NVIDIA GPU"
    gpu = torch.cuda.get_device_name(0)
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), None)
    print(f"GPU: {gpu}   peak memory bandwidth: {peak} GB/s\n")

    try:
        ext = load_cuda_extension()
        print("CUDA extension compiled\n")
    except Exception as err:  # keep going with the other implementations
        print(f"CUDA extension failed to build, skipping it:\n{err}\n")
        ext = None
    impls = implementations(ext)

    rows_out = []
    print(f"{'implementation':<17} {'shape':>11} {'time':>9} {'GB/s':>7} {'% peak':>7}  correct")
    for rows, hidden in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.half)
        r = torch.randn(rows, hidden, device="cuda", dtype=torch.half)
        w = torch.rand(hidden, device="cuda", dtype=torch.half) + 0.5
        out, new_r = torch.empty_like(x), torch.empty_like(x)
        ref_out, ref_h = reference(x, r, w)
        for name, fn in impls.items():
            got_out, got_h = fn(x, r, w, out, new_r)
            torch.cuda.synchronize()
            ok = (torch.allclose(got_out.float(), ref_out, atol=2e-2, rtol=2e-2)
                  and torch.allclose(got_h.float(), ref_h, atol=1e-3, rtol=1e-3))
            ms = do_bench(lambda: fn(x, r, w, out, new_r), warmup=25, rep=200)
            gbs = min_bytes(rows, hidden) / (ms * 1e-3) / 1e9
            pct = 100 * gbs / peak if peak else float("nan")
            print(f"{name:<17} {f'{rows}x{hidden}':>11} {ms * 1000:>7.1f}us {gbs:>7.1f} {pct:>6.1f}%  {'yes' if ok else 'NO'}")
            rows_out.append({"implementation": name, "rows": rows, "hidden": hidden,
                             "time_us": round(ms * 1000, 2), "gb_per_s": round(gbs, 1),
                             "pct_peak_bw": round(pct, 1), "correct": ok})
        print()

    os.makedirs("results", exist_ok=True)
    tag = "t4" if "T4" in gpu else gpu.split()[-1].lower()
    path = f"results/kernels-{tag}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"saved {path}")
    measure_energy(impls, tag)


def measure_energy(impls, tag):
    """Energy per call from NVML's energy counter, ~3 s of back-to-back calls each."""
    try:
        import pynvml
    except ImportError:
        print("\npynvml not installed (pip install nvidia-ml-py); skipping energy")
        return
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    energy_mj = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    time.sleep(2)
    e0, t0 = energy_mj(), time.time()
    time.sleep(3)
    idle_w = (energy_mj() - e0) / 1000 / (time.time() - t0)

    rows, hidden = ENERGY_SHAPE
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.half)
    r = torch.randn(rows, hidden, device="cuda", dtype=torch.half)
    w = torch.rand(hidden, device="cuda", dtype=torch.half) + 0.5
    out, new_r = torch.empty_like(x), torch.empty_like(x)
    gb = min_bytes(rows, hidden) / 1e9

    print(f"\nEnergy per call, shape {rows}x{hidden} ({gb * 1000:.0f} MB minimum traffic), idle {idle_w:.1f} W")
    print(f"{'implementation':<17} {'avg W':>6} {'mJ/call':>8} {'above idle':>11} {'J per GB':>9}")
    energy_rows = []
    for name, fn in impls.items():
        ms = do_bench(lambda: fn(x, r, w, out, new_r), warmup=25, rep=100)
        calls = max(10, math.ceil(3.0 / (ms * 1e-3)))
        torch.cuda.synchronize()
        e0, t0 = energy_mj(), time.time()
        for _ in range(calls):
            fn(x, r, w, out, new_r)
        torch.cuda.synchronize()
        seconds = time.time() - t0
        joules = (energy_mj() - e0) / 1000
        per_call = joules / calls
        dynamic = (joules - idle_w * seconds) / calls
        print(f"{name:<17} {joules / seconds:>6.1f} {per_call * 1000:>8.2f} {dynamic * 1000:>9.2f}mJ {per_call / gb:>9.2f}")
        energy_rows.append({"implementation": name, "rows": rows, "hidden": hidden,
                            "avg_power_w": round(joules / seconds, 1),
                            "mj_per_call": round(per_call * 1000, 3),
                            "mj_per_call_above_idle": round(dynamic * 1000, 3),
                            "idle_w": round(idle_w, 1)})
    path = f"results/kernels-energy-{tag}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(energy_rows[0]))
        writer.writeheader()
        writer.writerows(energy_rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
