from __future__ import annotations

from math import ceil

import torch
import triton
import triton.language as tl

@triton.jit
def _fused_norm_sq_kernel(
    H_ptr, P_ptr, OUT_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid       = tl.program_id(0)
    n_progs   = tl.num_programs(0)
    num_tiles = tl.cdiv(N, BLOCK_SIZE)

    h_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    d_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(pid, num_tiles, n_progs):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        h = tl.load(H_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(P_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        d = h - p
        h_acc += h * h
        d_acc += d * d

    tl.atomic_add(OUT_ptr + 0, tl.sum(h_acc, axis=0))
    tl.atomic_add(OUT_ptr + 1, tl.sum(d_acc, axis=0))


_BLOCK_SIZE = 8192
_NUM_WARPS = 8
_GRID_OCC = 8  # blocks per SM in the persistent grid

def _launch(h_flat: torch.Tensor, p_flat: torch.Tensor) -> torch.Tensor:
    if h_flat.numel() != p_flat.numel():
        raise ValueError("fused_norm: h and p must have equal numel")
    if h_flat.device != p_flat.device:
        raise ValueError("fused_norm: h and p must share a device")

    N = h_flat.numel()
    device = h_flat.device
    out = torch.zeros(2, dtype=torch.float32, device=device)
    if N == 0:
        return out

    nsm = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (min(nsm * _GRID_OCC, ceil(N / _BLOCK_SIZE)),)
    _fused_norm_sq_kernel[grid](
        h_flat, p_flat, out, N,
        BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS,
    )
    return out

def fused_norm_sq_triton(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return a CUDA fp32 tensor ``[sum(h^2), sum((h-p)^2)]``. Any float dtype."""
    h_flat = h.contiguous().view(-1)
    p_flat = p.contiguous().view(-1)
    return _launch(h_flat, p_flat)
