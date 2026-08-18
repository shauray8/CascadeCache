from __future__ import annotations

from typing import Callable
import numpy as np

SkipSchedule = Callable[[int], set]

def schedule_alt(
    period: int, late_start: int, n_blocks: int, n_steps: int,
    boundary_pre: int = 2, boundary_post: int = 1,
) -> SkipSchedule:
    """Skip every (period-1) out of `period` steps on blocks [late_start, n_blocks).

    period=2 -> skip 1/2 ("alt2");  period=3 -> skip 2/3 ("alt3").
    """
    def fn(step: int) -> set:
        if step < boundary_pre or step >= n_steps - boundary_post:
            return set()
        if step % period == 0:
            return set()
        return set(range(late_start, n_blocks))
    return fn

def schedule_threshold(
    step_sigma: np.ndarray, threshold: float,
    late_start: int, n_blocks: int, n_steps: int,
    boundary_pre: int = 2, boundary_post: int = 1, cap: int | None = 1,
) -> SkipSchedule:
    """Skip step t only when σ_t > threshold.

    ``cap=1`` (default) forbids two consecutive skips; ``cap=None`` disables the
    constraint entirely, which is needed to push the schedule past its natural
    ceiling of ``ceil(eligible/2)`` skips — e.g. when stress-testing to failure.
    """
    skip_set: set[int] = set()
    prev = -10
    for s in range(boundary_pre, n_steps - boundary_post):
        if not np.isfinite(step_sigma[s]) or step_sigma[s] <= threshold:
            continue
        if cap is not None and s - 1 == prev:   # cap=1 — no two in a row
            continue
        skip_set.add(s)
        prev = s

    def fn(step: int) -> set:
        return set(range(late_start, n_blocks)) if step in skip_set else set()
    return fn

def threshold_for_realized_skips(
    step_sigma: np.ndarray, target: int, n_steps: int,
    boundary_pre: int = 2, boundary_post: int = 1,
) -> tuple[float, int]:
    lo, hi = boundary_pre, n_steps - boundary_post
    cands = sorted({float(step_sigma[s]) for s in range(lo, hi)
                    if np.isfinite(step_sigma[s])}, reverse=True)
    best = (float("inf"), 0)
    for c in cands:
        tau = c - 1e-9
        realized, prev = 0, -10
        for s in range(lo, hi):
            if not np.isfinite(step_sigma[s]) or step_sigma[s] <= tau:
                continue
            if s - 1 == prev:
                continue
            realized += 1
            prev = s
        if realized <= target and realized > best[1]:
            best = (tau, realized)
        if realized >= target:
            break
    return best

def normalize_sigma(step_sigma: np.ndarray) -> np.ndarray:
    """σ / mean(σ). Rescales τ for cross-prompt transfer only — dividing by a
    positive constant leaves argsort unchanged, so this does NOT change which
    steps a threshold or top-k rule selects."""
    mean = np.nanmean(step_sigma)
    if not np.isfinite(mean) or mean == 0:
        return step_sigma
    return step_sigma / mean

def describe(
    schedule_fn: SkipSchedule, n_steps: int, n_blocks: int, n_streams: int = 1,
) -> dict:
    from .cache import as_stream_schedule

    fn = as_stream_schedule(schedule_fn)
    total = n_steps * n_blocks * n_streams
    skipped = 0
    per_step = []
    for s in range(n_steps):
        n = sum(len(fn(s, b)) for b in range(n_streams))
        per_step.append(n)
        skipped += n
    return {
        "total_block_evals": total,
        "skipped": skipped,
        "fraction": skipped / total if total else 0.0,
        "per_step": per_step,
    }

def skipped_steps(schedule_fn: SkipSchedule, n_steps: int) -> list[int]:
    from .cache import as_stream_schedule
    fn = as_stream_schedule(schedule_fn)
    return [s for s in range(n_steps) if fn(s, 0)]
