"""Paired statistics for treatment schedules scored on fresh futures of the true patient."""

from __future__ import annotations

import numpy as np

from twin import metrics
from twin.config import HORIZON_STEPS
from twin.engine import Result


def outcome_arrays(results: dict[str, Result], patient_id: str, name: str, n_futures: int) -> dict:
    """Per-future outcomes of one schedule: dead by day 28, and steps survived (capped at 28 d)."""
    runs = [results[f"{patient_id}|{name}|{r}"] for r in range(n_futures)]
    died = np.array([r.died for r in runs], dtype=float)
    steps = np.array([r.death_step if r.died else HORIZON_STEPS for r in runs], dtype=float)
    return {"died": died, "steps_survived": steps}


def paired(a: np.ndarray, b: np.ndarray, seed: int = 0, n_boot: int = 4000) -> dict:
    """mean(a), mean(b) and mean(a - b) with a 95% bootstrap interval over futures, plus the
    discordant pairs (a=1, b=0) and (a=0, b=1) for binary outcomes."""

    def mean(x):
        return float(np.mean(x))

    est, lo, hi = metrics.paired_bootstrap_diff(mean, (a,), (b,), n_boot=n_boot, seed=seed)
    return {
        "mean_a": mean(a),
        "mean_b": mean(b),
        "diff": est,
        "ci95": [lo, hi],
        "a_only": int(np.sum((a > 0) & (b == 0)))
        if set(np.unique(np.r_[a, b])) <= {0.0, 1.0}
        else None,
        "b_only": int(np.sum((a == 0) & (b > 0)))
        if set(np.unique(np.r_[a, b])) <= {0.0, 1.0}
        else None,
    }


def pooled_difference(
    per_patient: list[tuple[np.ndarray, np.ndarray]], seed: int = 0, n_boot: int = 4000
) -> dict:
    """Equal-weight mean over patients of (mean(a) - mean(b)); the bootstrap resamples futures
    within each patient and keeps the patients fixed (n is the number of chosen patients, so no
    claim about other patients is made)."""
    rng = np.random.default_rng(seed)
    est = float(np.mean([a.mean() - b.mean() for a, b in per_patient]))
    draws = []
    for _ in range(n_boot):
        diffs = []
        for a, b in per_patient:
            idx = rng.integers(0, len(a), size=len(a))
            diffs.append(a[idx].mean() - b[idx].mean())
        draws.append(np.mean(diffs))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"diff": est, "ci95": [float(lo), float(hi)], "n_patients": len(per_patient)}
