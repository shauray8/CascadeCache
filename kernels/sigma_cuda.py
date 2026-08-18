from __future__ import annotations

import threading

import torch
from torch.utils.cpp_extension import load_inline

_CPP_DECL = r"""
torch::Tensor fused_norm_sq_cuda(torch::Tensor h, torch::Tensor p);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define FN_THREADS 256
#define FN_WARPS   (FN_THREADS / 32)

// One thread per uint4 (= 8 packed bf16). Persistent grid-stride loop.
__global__ void fused_norm_sq_vec_kernel(
    const uint4* __restrict__ H,
    const uint4* __restrict__ P,
    float* __restrict__ out,     // out[0] = sum h^2, out[1] = sum (h-p)^2
    long n_vec)
{
    long idx    = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;

    float h_acc = 0.f, d_acc = 0.f;

    for (long i = idx; i < n_vec; i += stride) {
        uint4 hv = H[i];
        uint4 pv = P[i];
        const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&hv);
        const __nv_bfloat162* pp = reinterpret_cast<const __nv_bfloat162*>(&pv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 hh = hp[j];
            __nv_bfloat162 dd = __hsub2(hh, pp[j]);   // packed 2-wide subtract
            float2 hf = __bfloat1622float2(hh);
            float2 df = __bfloat1622float2(dd);
            h_acc += hf.x * hf.x + hf.y * hf.y;
            d_acc += df.x * df.x + df.y * df.y;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        h_acc += __shfl_down_sync(0xffffffffu, h_acc, off);
        d_acc += __shfl_down_sync(0xffffffffu, d_acc, off);
    }

    __shared__ float sh_h[FN_WARPS];
    __shared__ float sh_d[FN_WARPS];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    if (lane == 0) { sh_h[wid] = h_acc; sh_d[wid] = d_acc; }
    __syncthreads();

    if (wid == 0) {
        h_acc = (lane < FN_WARPS) ? sh_h[lane] : 0.f;
        d_acc = (lane < FN_WARPS) ? sh_d[lane] : 0.f;
        #pragma unroll
        for (int off = FN_WARPS / 2; off > 0; off >>= 1) {
            h_acc += __shfl_down_sync(0xffffffffu, h_acc, off);
            d_acc += __shfl_down_sync(0xffffffffu, d_acc, off);
        }
        if (lane == 0) {
            atomicAdd(&out[0], h_acc);
            atomicAdd(&out[1], d_acc);
        }
    }
}

__global__ void fused_norm_sq_tail_kernel(
    const __nv_bfloat16* __restrict__ H,
    const __nv_bfloat16* __restrict__ P,
    float* __restrict__ out,
    long start, long n)
{
    long i = start + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= start + n) return;
    float h = __bfloat162float(H[i]);
    float d = h - __bfloat162float(P[i]);
    atomicAdd(&out[0], h * h);
    atomicAdd(&out[1], d * d);
}

torch::Tensor fused_norm_sq_cuda(torch::Tensor h, torch::Tensor p) {
    TORCH_CHECK(h.is_cuda() && p.is_cuda(), "fused_norm_sq_cuda: tensors must be CUDA");
    TORCH_CHECK(h.scalar_type() == torch::kBFloat16 && p.scalar_type() == torch::kBFloat16,
                "fused_norm_sq_cuda: bf16 only");
    TORCH_CHECK(h.numel() == p.numel(), "fused_norm_sq_cuda: size mismatch");

    auto hc = h.contiguous();
    auto pc = p.contiguous();
    long N = hc.numel();
    auto out = torch::zeros({2}, hc.options().dtype(torch::kFloat32));
    if (N == 0) return out;

    auto stream = at::cuda::getCurrentCUDAStream();
    const at::BFloat16* hptr = hc.data_ptr<at::BFloat16>();
    const at::BFloat16* pptr = pc.data_ptr<at::BFloat16>();

    bool aligned = (reinterpret_cast<uintptr_t>(hptr) % 16 == 0) &&
                   (reinterpret_cast<uintptr_t>(pptr) % 16 == 0);

    long n_vec = aligned ? (N / 8) : 0;
    long tail_start = n_vec * 8;
    long tail = N - tail_start;

    if (n_vec > 0) {
        int device;
        cudaGetDevice(&device);
        int nsm = 0;
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device);
        long blocks_needed = (n_vec + FN_THREADS - 1) / FN_THREADS;
        long blocks_cap = (long)nsm * 8;
        int blocks = (int)(blocks_needed < blocks_cap ? blocks_needed : blocks_cap);
        if (blocks < 1) blocks = 1;

        fused_norm_sq_vec_kernel<<<blocks, FN_THREADS, 0, stream>>>(
            reinterpret_cast<const uint4*>(hptr),
            reinterpret_cast<const uint4*>(pptr),
            out.data_ptr<float>(), n_vec);
    }
    if (tail > 0) {
        int threads = 128;
        int blocks = (int)((tail + threads - 1) / threads);
        fused_norm_sq_tail_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hptr),
            reinterpret_cast<const __nv_bfloat16*>(pptr),
            out.data_ptr<float>(), tail_start, tail);
    }
    return out;
}
"""

_ext = None
_load_error: Exception | None = None
_lock = threading.Lock()


def _get_ext():
    global _ext, _load_error
    if _ext is not None or _load_error is not None:
        return _ext
    with _lock:
        if _ext is not None or _load_error is not None:
            return _ext
        try:
            _ext = load_inline(
                name="cascade_cache_fused_norm_cuda",
                cpp_sources=_CPP_DECL,
                cuda_sources=_CUDA_SRC,
                functions=["fused_norm_sq_cuda"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        except Exception as e:
            _load_error = e
            _ext = None
    return _ext


def available() -> bool:
    """True if CUDA is up and the inline extension compiled successfully."""
    return torch.cuda.is_available() and _get_ext() is not None


def load_error() -> Exception | None:
    return _load_error


def fused_norm_sq_cuda(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Return a CUDA fp32 tensor ``[sum(h^2), sum((h-p)^2)]``. bf16 inputs only."""
    ext = _get_ext()
    if ext is None:
        raise RuntimeError(f"inline CUDA sigma kernel unavailable: {_load_error!r}")
    return ext.fused_norm_sq_cuda(h, p)
