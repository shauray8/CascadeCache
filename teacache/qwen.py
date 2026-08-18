from __future__ import annotations

import contextlib
from typing import Callable

import numpy as np
import torch

QWEN_COEFFS = [-4.50000000e02, 2.80000000e02, -4.50000000e01, 3.20000000e00, -2.00000000e-02]

class _BlockState:
    __slots__ = ("threshold", "poly", "prev_input", "prev_residual", "cumdist",
                 "skip_count", "compute_count")

    def __init__(self, threshold: float, coeffs):
        self.threshold = float(threshold)
        self.poly = np.poly1d(coeffs)
        self.prev_input = None
        self.prev_residual = None
        self.cumdist = 0.0
        self.skip_count = 0
        self.compute_count = 0

    def decide(self, x: torch.Tensor) -> bool:
        x = x.detach()
        if self.prev_input is None:
            self.cumdist = 0.0
            self.prev_input = x.clone()
            return False
        num = (x - self.prev_input).abs().mean()
        den = self.prev_input.abs().mean().clamp_min(1e-8)
        rel = float((num / den).item())
        self.cumdist += float(self.poly(rel))
        if self.cumdist < self.threshold and self.prev_residual is not None:
            self.prev_input = x.clone()
            return True
        self.cumdist = 0.0
        self.prev_input = x.clone()
        return False


@contextlib.contextmanager
def teacache_qwen(
    transformer: torch.nn.Module,
    blocks: torch.nn.ModuleList,
    threshold: float,
    stream_key_fn: Callable | None = None,
    coeffs=QWEN_COEFFS,
):
    ctx = {"stream": 0}
    states: list[dict[object, _BlockState]] = [dict() for _ in blocks]
    fmts: list[dict] = [{"img_idx": None} for _ in blocks]

    orig_tf_forward = transformer.forward
    tf_had_own = "forward" in transformer.__dict__
    orig_block_forwards = [b.forward for b in blocks]
    blocks_had_own = ["forward" in b.__dict__ for b in blocks]

    def tf_forward(*args, **kwargs):
        ctx["stream"] = stream_key_fn(args, kwargs) if stream_key_fn else 0
        return orig_tf_forward(*args, **kwargs)

    def make_patched(i: int, orig: Callable):
        def patched(hidden_states, *args, **kwargs):
            stream = ctx["stream"]
            st = states[i].setdefault(stream, _BlockState(threshold, coeffs))
            fmt = fmts[i]
            ori_input = hidden_states.detach()
            skip = st.decide(hidden_states)

            if skip:
                st.skip_count += 1
                out_hs = ori_input + st.prev_residual
                if fmt["img_idx"] is None:
                    return out_hs
                enc = kwargs.get("encoder_hidden_states")
                if enc is None and args:
                    enc = args[0]
                return (out_hs, enc) if fmt["img_idx"] == 0 else (enc, out_hs)

            out = orig(hidden_states, *args, **kwargs)
            if isinstance(out, tuple):
                if out[0].shape == ori_input.shape:
                    fmt["img_idx"], out_img = 0, out[0]
                elif len(out) > 1 and out[1].shape == ori_input.shape:
                    fmt["img_idx"], out_img = 1, out[1]
                else:
                    fmt["img_idx"], out_img = len(out) - 1, out[-1]
            else:
                fmt["img_idx"], out_img = None, out
            st.compute_count += 1
            st.prev_residual = (out_img - ori_input).detach()
            return out
        return patched

    transformer.forward = tf_forward
    for i, b in enumerate(blocks):
        b.forward = make_patched(i, orig_block_forwards[i])

    try:
        yield {"states": states}
    finally:
        if tf_had_own:
            transformer.forward = orig_tf_forward
        else:
            del transformer.forward
        for b, f, had_own in zip(blocks, orig_block_forwards, blocks_had_own):
            if had_own:
                b.forward = f
            else:
                del b.forward
