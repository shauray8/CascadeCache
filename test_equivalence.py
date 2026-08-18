from __future__ import annotations

import argparse

import numpy as np
import torch

from .cache import cascade_cache, boundaries
from .models import ADAPTERS, get_adapter, load_pipeline
from .schedules import schedule_alt, skipped_steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--prompt", default="a red fox in fresh snow, photorealistic")
    ap.add_argument("--negative-prompt", default="blurry, low quality")
    args = ap.parse_args()

    a = get_adapter(args.model)
    pipe = load_pipeline(a)
    blocks = a.blocks(pipe)
    nb = len(blocks)
    print(f"{a.name}: {nb} blocks, cfg_style={a.cfg_style}, "
          f"expect streams={a.n_streams}")

    orig_tf = pipe.transformer.forward
    orig_b0 = blocks[0].forward

    def call():
        g = torch.Generator(device="cuda").manual_seed(0)
        return pipe(**a.call_kwargs(args.prompt, args.negative_prompt, args.steps,
                                    a.guidance, args.size, args.size, g)).images[0]

    print("\n[1] stock pipeline (no patch)...")
    ref = np.asarray(call())

    print("[2] dense pass with cache installed...")
    with cascade_cache(pipe.transformer, blocks, args.steps,
                     lambda s: set(), stream_key_fn=a.stream_key_fn) as st:
        got = np.asarray(call())
    print(f"    streams={st['n_streams']} forwards={st['forwards']} "
          f"skipped_blocks={st['skipped']} invalidations={st['invalidations']}")

    md = int(np.abs(ref.astype(int) - got.astype(int)).max())
    print(f"    max|Δ| vs stock = {md}")
    assert md == 0, "dense cached pass diverged from the stock pipeline!"

    assert st["n_streams"] == a.n_streams, (st["n_streams"], a.n_streams)
    expect_fwd = args.steps * (a.n_streams if a.cfg_style == "two_call" else 1)
    assert st["forwards"] == expect_fwd, (st["forwards"], expect_fwd)
    assert st["skipped"] == 0, "dense pass skipped blocks!"

    print("[3] alt2 schedule...")
    bp, bq = boundaries(args.steps)
    sched = schedule_alt(2, 0, nb, args.steps, bp, bq)
    n_skip_steps = len(skipped_steps(sched, args.steps))
    with cascade_cache(pipe.transformer, blocks, args.steps, sched,
                     stream_key_fn=a.stream_key_fn) as st2:
        alt = np.asarray(call())
    exp_skipped = n_skip_steps * nb * (a.n_streams if a.cfg_style == "two_call" else 1)
    print(f"    skip steps={n_skip_steps} skipped block-evals={st2['skipped']} "
          f"(expect {exp_skipped})")
    assert st2["skipped"] == exp_skipped, (st2["skipped"], exp_skipped)
    d = np.abs(alt.astype(int) - ref.astype(int))
    print(f"    mean|Δ| vs dense = {d.mean():.3f}  max|Δ| = {d.max()}")
    assert d.mean() > 0, "schedule had no effect"

    print("[4] restoration...")
    assert pipe.transformer.forward == orig_tf
    assert blocks[0].forward == orig_b0
    assert "forward" not in pipe.transformer.__dict__
    assert "forward" not in blocks[0].__dict__
    print("    forwards restored, no lingering instance attributes")

    print(f"\nALL OK — {a.name}")

if __name__ == "__main__":
    main()
