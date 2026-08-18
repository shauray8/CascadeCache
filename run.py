from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .cache import block_cache, boundaries, step_sigma
from .models import ADAPTERS, get_adapter, load_pipeline
from .schedules import (
    describe,
    schedule_alt,
    schedule_threshold,
    skipped_steps,
    threshold_for_realized_skips,
)
from .kernels.sigma import backend_for

def _slug(t: str, n: int = 6) -> str:
    import re
    return "-".join(re.sub(r"[^a-z0-9\s-]", "", t.lower()).split()[:n]) or "prompt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--prompts-json", default="")
    ap.add_argument("--negative-prompt", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--late-start", type=int, default=0,
                    help="first skip-eligible block. Output-neutral (cascade "
                         "rule); 0 = maximum speedup.")
    ap.add_argument("--schedule", default="thr",
                    choices=["dense", "alt2", "alt3", "thr", "all"])
    ap.add_argument("--out-dir", default="./output/cascade_cache")
    args = ap.parse_args()

    a = get_adapter(args.model)
    steps = args.steps or a.steps
    guidance = a.guidance if args.guidance is None else args.guidance
    height, width = args.height or a.height, args.width or a.width

    blob = json.loads(Path(args.prompts_json).read_text())
    neg = args.negative_prompt if args.negative_prompt is not None \
        else blob.get("negative_prompt", "")
    items = [{"name": p.get("name") or _slug(p["prompt"]), "prompt": p["prompt"]}
             for p in blob["prompts"]]

    out_dir = Path(args.out_dir) / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = (f"{a.name}__{width}x{height}__steps{steps}_cfg{guidance:g}"
               f"_seed{args.seed}")

    print(f"Loading {a.repo_id} via diffusers.{a.pipeline_cls} ...")
    pipe = load_pipeline(a)
    blocks = a.blocks(pipe)
    n_blocks = len(blocks)
    ls = args.late_start
    bp, bq = boundaries(steps)
    print(f"  {n_blocks} blocks ({a.blocks_attr}) | cfg_style={a.cfg_style} "
          f"expect_streams={a.n_streams}")
    print(f"  steps={steps} {a.guidance_kwarg}={guidance} {width}x{height} "
          f"boundaries=({bp},{bq}) late_start={ls}")
    print(f"  negative_prompt={neg[:60]!r}...")
    print(f"  GPU {torch.cuda.memory_allocated()/1e9:.1f} GB | sigma backend "
          f"{backend_for(torch.zeros(1, dtype=torch.bfloat16, device='cuda'))}")
    print(f"  run tag: {run_tag}")

    def gen(prompt, sched, capture=False):
        gencuda = torch.Generator(device="cuda").manual_seed(args.seed)
        with block_cache(pipe.transformer, blocks, steps, sched,
                         stream_key_fn=a.stream_key_fn, capture_sigma=capture,
                         late_start=ls) as st:
            torch.cuda.synchronize()
            t0 = time.time()
            img = pipe(**a.call_kwargs(prompt, neg, steps, guidance,
                                       height, width, gencuda)).images[0]
            torch.cuda.synchronize()
            dt = time.time() - t0
        return img, dt, st

    print("\nWarmup...", end=" ", flush=True)
    t0 = time.time()
    _, _, stw = gen(items[0]["prompt"], lambda s: set())
    print(f"{time.time()-t0:.1f}s  streams={stw['n_streams']} "
          f"forwards={stw['forwards']} invalidations={stw['invalidations']}")
    if stw["n_streams"] != a.n_streams:
        raise SystemExit(
            f"stream keying wrong: saw {stw['n_streams']}, expected {a.n_streams}")

    speedups: dict[str, list[float]] = {}
    for pi, item in enumerate(items):
        prompt, name = item["prompt"], item["name"]
        print(f"\n{'='*78}\n[{pi+1}/{len(items)}] {name}\n{'='*78}")

        img_d, t_d, _ = gen(prompt, lambda s: set())
        img_d.save(out_dir / f"{run_tag}__p{pi:02d}-{name}__dense.png")
        print(f"  {'dense':10s} {t_d:6.2f}s  (reference)")

        runs = []
        if args.schedule in ("alt2", "all"):
            runs.append(("alt2", schedule_alt(2, ls, n_blocks, steps, bp, bq)))
        if args.schedule in ("alt3", "all"):
            runs.append(("alt3", schedule_alt(3, ls, n_blocks, steps, bp, bq)))

        if args.schedule in ("thr", "all"):
            _, _, st_sig = gen(prompt, lambda s: set(), capture=True)
            sig = step_sigma(st_sig, stream=0, late_start=ls)
            cap = -(-(steps - bp - bq) // 2)
            targets = [("thr-quality", max(1, round(cap * 0.4))),
                       ("thr-balanced", max(1, round(cap * 0.7))),
                       ("thr-fast", cap)]
            for label, target in targets:
                tau, realized = threshold_for_realized_skips(sig, target, steps, bp, bq)
                if not np.isfinite(tau) or realized == 0:
                    print(f"  !! {label}: no usable τ for {target} skips")
                    continue
                print(f"  {label:14s} τ={tau:.4f}  realizes {realized}/{target} skips")
                runs.append((label, schedule_threshold(sig, tau, ls, n_blocks, steps, bp, bq)))

        for label, sched in runs:
            d = describe(sched, steps, n_blocks, a.n_streams)
            img, t, _ = gen(prompt, sched)
            img.save(out_dir / f"{run_tag}__p{pi:02d}-{name}__{label}.png")
            speedup = t_d / t
            speedups.setdefault(label, []).append(speedup)
            print(f"  {label:10s} {t:6.2f}s  {speedup:.2f}x  "
                  f"budget={d['fraction']*100:4.1f}%  "
                  f"skips={len(skipped_steps(sched, steps))}")

    if speedups:
        print(f"\n\n{'schedule':<12s} {'mean speedup':>12s}")
        print("-" * 26)
        for label, vals in speedups.items():
            print(f"{label:<12s} {np.mean(vals):11.2f}x")

    print(f"\nImages in {out_dir}/")

if __name__ == "__main__":
    main()
