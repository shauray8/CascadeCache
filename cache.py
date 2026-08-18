from __future__ import annotations

import contextlib
import inspect
import warnings
from typing import Any, Callable

import numpy as np
import torch

from .kernels.sigma import fused_norm

def as_stream_schedule(fn: Callable) -> Callable[[int, int], set]:
    """Accept both ``fn(step)`` and ``fn(step, stream)`` schedules."""
    try:
        n = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n = 1
    return fn if n >= 2 else (lambda step, stream: fn(step))

def _detach(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _detach(v) for k, v in obj.items()}
    return obj

def _primary_tensor(obj: Any) -> torch.Tensor | None:
    best, best_n = None, -1
    stack = [obj]
    while stack:
        o = stack.pop()
        if torch.is_tensor(o):
            if o.numel() > best_n:
                best, best_n = o, o.numel()
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
    return best

def _signature(args, kwargs) -> tuple:
    sig = []
    for v in list(args) + list(kwargs.values()):
        if torch.is_tensor(v):
            sig.append(tuple(v.shape))
        elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
            sig.append((len(v),) + tuple(v[0].shape))
    return tuple(sig)

@contextlib.contextmanager
def cascade_cache(
    transformer: torch.nn.Module,
    blocks: torch.nn.ModuleList,
    n_steps: int,
    schedule_fn: Callable = lambda step: set(),
    stream_key_fn: Callable | None = None,
    capture_sigma: bool = False,
    late_start: int | None = None,
    cache_mode: str = "last",
):
    n_blocks = len(blocks)
    sched = as_stream_schedule(schedule_fn)
    ls = late_start if late_start is not None else 0
    sigma0 = np.full((n_steps, n_blocks), np.nan, dtype=np.float64)

    st: dict[str, Any] = {
        "step": {},        # stream -> next step index
        "cache": {},       # stream -> {block_idx: output}
        "prev": {},        # stream -> [per-block previous primary tensor]
        "streams": {},     # stream key -> stream index
        "sig": {},         # stream -> last input signature
        "sigma": {0: sigma0},
        "sigma_mat": sigma0,
        "n_blocks": n_blocks,
        "late_start": ls,
        "n_streams": 0,
        "forwards": 0,
        "skipped": 0,
        "invalidations": 0,
    }
    # Mutable per-forward context the block wrappers read.
    ctx = {"stream": 0, "skip_from": None, "capture": False}

    orig_tf_forward = transformer.forward
    tf_had_own = "forward" in transformer.__dict__
    orig_block_forwards = [b.forward for b in blocks]
    blocks_had_own = ["forward" in b.__dict__ for b in blocks]

    last = n_blocks - 1

    def make_block_forward(i: int, orig: Callable):
        def wrapped(*args, **kwargs):
            stream = ctx["stream"]
            cache = st["cache"].setdefault(stream, {})
            key = last if cache_mode == "last" else i
            if ctx["skip_from"] is not None and i >= ctx["skip_from"] and key in cache:
                st["skipped"] += 1
                return cache[key]
            out = orig(*args, **kwargs)
            if cache_mode != "last" or i == last:
                cache[key] = _detach(out)
            if ctx["capture"]:
                cur = _primary_tensor(out)
                prev = st["prev"].setdefault(stream, [None] * n_blocks)
                step = st["step"].get(stream, 0)
                if prev[i] is not None and cur is not None and step < n_steps:
                    if prev[i].shape == cur.shape:
                        h, d = fused_norm(cur, prev[i])
                        smat = st["sigma"].setdefault(
                            stream, np.full((n_steps, n_blocks), np.nan))
                        smat[step, i] = h / d
                if cur is not None:
                    prev[i] = cur.detach().clone()
            return out
        return wrapped

    def tf_forward(*args, **kwargs):
        key = stream_key_fn(args, kwargs) if stream_key_fn else 0
        stream = st["streams"].setdefault(key, len(st["streams"]))
        st["n_streams"] = len(st["streams"])
        st["forwards"] += 1

        sig = _signature(args, kwargs)
        if st["sig"].get(stream) not in (None, sig):
            st["cache"].pop(stream, None)
            st["prev"].pop(stream, None)
            st["invalidations"] += 1
        st["sig"][stream] = sig

        step = st["step"].get(stream, 0)
        skip = sched(step, stream) if step < n_steps else set()
        ctx["stream"] = stream
        ctx["skip_from"] = min(skip) if skip else None
        ctx["capture"] = bool(capture_sigma) and step < n_steps and stream == 0

        out = orig_tf_forward(*args, **kwargs)
        st["step"][stream] = step + 1
        return out

    transformer.forward = tf_forward
    for i, b in enumerate(blocks):
        b.forward = make_block_forward(i, orig_block_forwards[i])
    try:
        yield st
    finally:
        # Drop the cached activations before anything downstream allocates.
        st["cache"].clear()
        st["prev"].clear()
        if tf_had_own:
            transformer.forward = orig_tf_forward
        else:
            del transformer.forward
        for b, f, had_own in zip(blocks, orig_block_forwards, blocks_had_own):
            if had_own:
                b.forward = f
            else:
                del b.forward

def step_sigma(
    st: dict, stream: int = 0, late_start: int | None = None, normalize: bool = False,
) -> np.ndarray:
    """Per-step σ averaged over the skip-eligible blocks of one CFG stream."""
    from .schedules import normalize_sigma

    ls = st["late_start"] if late_start is None else late_start
    mat = st["sigma"].get(stream)
    if mat is None:
        raise KeyError(f"no σ captured for stream {stream}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        out = np.nanmean(mat[:, ls:], axis=1)
    return normalize_sigma(out) if normalize else out

def boundaries(n_steps: int) -> tuple[int, int]:
    return max(2, n_steps // 6), max(1, n_steps // 25)
