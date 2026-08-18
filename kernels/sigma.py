from __future__ import annotations

import torch

from . import sigma_cuda
from .sigma_triton import fused_norm_sq_triton

def _squared_norms(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    if h.numel() != p.numel():
        raise ValueError("fused_norm: h and p must have equal numel")
    if h.device != p.device:
        raise ValueError("fused_norm: h and p must share a device")
    if h.dtype != p.dtype:
        raise ValueError("fused_norm: h and p must share a dtype")

    if h.dtype == torch.bfloat16 and h.is_cuda and sigma_cuda.available():
        return sigma_cuda.fused_norm_sq_cuda(h, p)
    return fused_norm_sq_triton(h, p)

def fused_norm(
    h: torch.Tensor, p: torch.Tensor, eps: float = 1e-6,
) -> tuple[float, float]:
    out = _squared_norms(h, p)
    h_sq, d_sq = out.tolist()
    h_nrm = h_sq ** 0.5
    d_nrm = d_sq ** 0.5 + eps
    return h_nrm, d_nrm

def fused_norm_device(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return _squared_norms(h, p)

def backend_for(h: torch.Tensor) -> str:
    if h.dtype == torch.bfloat16 and h.is_cuda and sigma_cuda.available():
        return "cuda"
    return "triton"
