from __future__ import annotations

import contextlib

import numpy as np
import torch

Z_COEFFS = [-4.50000000e02, 2.80000000e02, -4.50000000e01, 3.20000000e00, -2.00000000e-02]

@contextlib.contextmanager
def teacache_zimage(transformer, rel_l1_thresh: float,
                    force_skip_steps: set[int] | None = None):
    layers = transformer.layers
    n = len(layers)
    rescale = np.poly1d(Z_COEFFS)

    st = {
        "cnt": 0, "acc": 0.0, "prev_mod": None,
        "residual": None, "sig": None,
        "forwards": 0, "computed": 0, "skipped": 0, "skipped_steps": [],
        "invalidations": 0,
    }
    ctx = {"skip": False, "in0": None}

    orig_forwards = [l.forward for l in layers]
    had_own = ["forward" in l.__dict__ for l in layers]

    def make_first(orig):
        def wrapped(x, *args, **kwargs):
            sig = tuple(x.shape)
            if st["sig"] is not None and st["sig"] != sig:
                st["residual"] = None
                st["prev_mod"] = None
                st["invalidations"] += 1
            st["sig"] = sig
            st["forwards"] += 1
            step = st["cnt"]
            mod = x
            if force_skip_steps is not None:
                skip = (step in force_skip_steps and step != 0
                        and st["residual"] is not None)
            elif step == 0 or st["prev_mod"] is None or st["residual"] is None:
                skip = False
                st["acc"] = 0.0
            else:
                rel = ((mod - st["prev_mod"]).abs().mean()
                       / (st["prev_mod"].abs().mean() + 1e-8)).cpu().item()
                st["acc"] += abs(float(rescale(rel)))
                if st["acc"] < rel_l1_thresh:
                    skip = True
                else:
                    skip = False
                    st["acc"] = 0.0
            st["prev_mod"] = mod.detach()
            ctx["skip"] = skip
            ctx["in0"] = x.detach()
            if skip:
                st["skipped"] += 1
                st["skipped_steps"].append(step)
                return x + st["residual"]
            st["computed"] += 1
            return orig(x, *args, **kwargs)
        return wrapped

    def make_mid(orig):
        def wrapped(x, *args, **kwargs):
            if ctx["skip"]:
                return x
            return orig(x, *args, **kwargs)
        return wrapped

    def make_last(orig, idx):
        def wrapped(x, *args, **kwargs):
            if ctx["skip"]:
                return x
            out = orig(x, *args, **kwargs)
            st["residual"] = (out - ctx["in0"]).detach()
            return out
        return wrapped

    for i, l in enumerate(layers):
        if i == 0 and n > 1:
            l.forward = make_first(orig_forwards[i])
        elif i == n - 1:
            base = make_last(orig_forwards[i], i)
            l.forward = base if n > 1 else make_first(orig_forwards[i])
        else:
            l.forward = make_mid(orig_forwards[i])

    orig_tf = transformer.forward
    tf_had_own = "forward" in transformer.__dict__

    def tf_forward(*args, **kwargs):
        out = orig_tf(*args, **kwargs)
        st["cnt"] += 1
        return out

    transformer.forward = tf_forward
    try:
        yield st
    finally:
        st["residual"] = None
        st["prev_mod"] = None
        for l, f, own in zip(layers, orig_forwards, had_own):
            if own:
                l.forward = f
            else:
                del l.forward
        if tf_had_own:
            transformer.forward = orig_tf
        else:
            del transformer.forward
