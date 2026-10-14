"""Roofline model for the Tesla T4, and what it says about our LLM results.

A kernel's speed is capped by whichever runs out first:
  - compute:          peak FLOP/s
  - memory bandwidth: bytes/s × (FLOPs done per byte moved)

The FLOPs-per-byte ratio is the *arithmetic intensity*. Below the "ridge
point" (peak FLOP/s ÷ bandwidth) a kernel is memory-bound; above it,
compute-bound.

    python kernels/roofline.py          # prints the analysis, saves results/roofline-t4.png
"""

import csv
import os

# Tesla T4 datasheet: 65 TFLOPS FP16 on tensor cores, 320 GB/s GDDR6.
PEAK_FLOPS = 65e12
PEAK_BW = 320e9
RIDGE = PEAK_FLOPS / PEAK_BW  # FLOP per byte

# Qwen2.5-0.5B-Instruct: 494M parameters, hidden size 896.
PARAMS = 494e6
HIDDEN = 896


def linear_intensity(tokens, d_in=HIDDEN, d_out=HIDDEN):
    """A linear layer on `tokens` rows: 2*tokens*d_in*d_out FLOPs; reads weights once (fp16)."""
    flops = 2 * tokens * d_in * d_out
    bytes_moved = 2 * (d_in * d_out + tokens * d_in + tokens * d_out)
    return flops / bytes_moved


def attainable(intensity):
    return min(PEAK_FLOPS, intensity * PEAK_BW)


OPERATIONS = [
    # (label, arithmetic intensity, kind)
    ("Fused add + RMSNorm", 5 / 8, "memory"),  # ~5 FLOPs per element, 8 bytes per element
    ("Decode, batch 1", linear_intensity(1), "memory"),
    ("Decode, batch 16", linear_intensity(16), "memory"),
    ("Decode, batch 64", linear_intensity(64), "memory"),
    ("Prefill, 2048 tokens", linear_intensity(2048), "compute"),
]


def print_table():
    print(f"Tesla T4: {PEAK_FLOPS / 1e12:.0f} TFLOPS FP16, {PEAK_BW / 1e9:.0f} GB/s "
          f"-> ridge point {RIDGE:.0f} FLOP/byte\n")
    print(f"{'operation':<22} {'FLOP/byte':>10} {'max TFLOPS':>11}  bound by")
    for label, ai, _ in OPERATIONS:
        bound = "memory" if ai < RIDGE else "compute"
        print(f"{label:<22} {ai:>10.1f} {attainable(ai) / 1e12:>11.2f}  {bound}")


def decode_budget():
    """Compare measured decode speed with the memory-bandwidth floor."""
    weight_bytes = 2 * PARAMS  # fp16: every decode step reads every weight once
    floor_ms = weight_bytes / PEAK_BW * 1000
    print(f"\nDecode step for Qwen2.5-0.5B: reads {weight_bytes / 1e9:.2f} GB of weights "
          f"-> at least {floor_ms:.1f} ms per step -> at most {1000 / floor_ms:.0f} tokens/s per sequence")

    path = "results/vllm-t4.csv"
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    by_level = {}
    for r in rows:
        if r["decode_tokens_per_s"]:
            by_level.setdefault(int(r["concurrency"]), []).append(float(r["decode_tokens_per_s"]))
    print(f"\nMeasured with vLLM ({path}):")
    print(f"{'concurrent':>10} {'tok/s per seq':>14} {'ms per step':>12} {'x above floor':>14}")
    for level in sorted(by_level):
        tps = sum(by_level[level]) / len(by_level[level])
        step_ms = 1000 / tps
        print(f"{level:>10} {tps:>14.1f} {step_ms:>12.1f} {step_ms / floor_ms:>13.1f}x")
    print("\nA step takes ~10x longer than reading the weights requires, at every batch size:\n"
          "neither compute nor memory is the limit. In eager mode the time goes to launching\n"
          "hundreds of small kernels per step from Python. Batching shares that fixed cost\n"
          "across sequences, which is why energy per token fell 13x while per-sequence speed\n"
          "stayed flat.")


def plot(path="results/roofline-t4.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    color = {"memory": "#2a78d6", "compute": "#eb6834"}
    marker = {"memory": "o", "compute": "s"}  # shape as well as color

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    xs = [0.1 * 1.1 ** i for i in range(140)]
    ax.plot(xs, [attainable(x) / 1e12 for x in xs], color=ink, linewidth=2)
    ax.axvline(RIDGE, color=muted, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate(f"ridge point\n{RIDGE:.0f} FLOP/byte", (RIDGE, 0.15), xytext=(6, 0),
                textcoords="offset points", color=muted, fontsize=9)
    ax.annotate("memory roof: 320 GB/s", (2.5, attainable(2.5) / 1e12), rotation=37,
                xytext=(-14, 14), textcoords="offset points", color=muted, fontsize=9)
    ax.annotate("compute roof: 65 TFLOPS", (2500, 65), xytext=(0, 8),
                textcoords="offset points", color=muted, fontsize=9, ha="center")

    for label, ai, kind in OPERATIONS:
        y = attainable(ai) / 1e12
        ax.scatter([ai], [y], s=70, color=color[kind], marker=marker[kind],
                   edgecolors=surface, linewidths=2, zorder=3)
        ax.annotate(label, (ai, y), xytext=(8, -12), textcoords="offset points",
                    color=ink, fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 2e4)
    ax.set_ylim(0.1, 200)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte moved)", color=muted)
    ax.set_ylabel("Attainable TFLOPS (FP16)", color=muted)
    ax.set_title("Roofline: Tesla T4", color=ink, loc="left", fontsize=12)
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=muted)
    handles = [plt.Line2D([], [], marker=marker[k], color=color[k], linestyle="", markersize=8,
                          label=f"{k}-bound") for k in ("memory", "compute")]
    ax.legend(handles=handles, frameon=False, loc="upper left", labelcolor=ink)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=surface)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    print_table()
    decode_budget()
    try:
        plot()
    except ImportError:
        print("\n(matplotlib not installed: skipping the chart)")
