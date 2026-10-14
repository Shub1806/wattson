"""Fused residual-add + RMSNorm in Triton (same math as fused_add_rmsnorm.cu).

Triton works on whole blocks of values at once: one program handles one row,
loading all `hidden` values as a single vector. The compiler picks the
vectorized loads, the warp-level reduction and the thread layout that the
CUDA version spells out by hand.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm(X, R, W, OUT, NEW_R, hidden, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden
    offs = row * hidden + cols

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + offs, mask=mask, other=0.0).to(tl.float32)
    h = (x + r).to(tl.float16)  # the new residual is stored in fp16
    tl.store(NEW_R + offs, h, mask=mask)

    hf = h.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(hf * hf, axis=0) / hidden + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + offs, (hf * inv_rms * w).to(tl.float16), mask=mask)


def fused_add_rmsnorm(x, residual, weight, out, new_residual, eps=1e-6):
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    block = triton.next_power_of_2(hidden)
    num_warps = 4 if block <= 1024 else 8 if block <= 4096 else 16
    _fused_add_rmsnorm[(rows,)](x, residual, weight, out, new_residual, hidden, eps,
                                BLOCK=block, num_warps=num_warps)
    return out, new_residual
