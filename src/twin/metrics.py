"""Pure statistics for validation: scores, bootstrap, coverage and rank checks.

No truth is read here; callers pass arrays. Patients are the unit of resampling.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import chisquare


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> float:
    """Mean negative log-likelihood; probabilities are clipped to [eps, 1 - eps]."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney; ties count one half). NaN if one class is empty."""
    p, y = np.asarray(p, float), np.asarray(y, bool)
    pos, neg = p[y], p[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def bootstrap_ci(
    stat: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """(estimate, lower, upper): percentile bootstrap over PATIENTS (rows of every array)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    est = stat(*arrays)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat(*[a[idx] for a in arrays])
        if not np.isnan(v):
            draws.append(v)
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float(est), float(lo), float(hi)


def paired_bootstrap_diff(
    stat: Callable[..., float], a: tuple, b: tuple, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """stat(a) - stat(b) with a 95% CI, resampling the same patients for both methods."""
    a = [np.asarray(x) for x in a]
    b = [np.asarray(x) for x in b]
    n = len(a[0])
    rng = np.random.default_rng(seed)
    est = stat(*a) - stat(*b)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat(*[x[idx] for x in a]) - stat(*[x[idx] for x in b])
        if not np.isnan(v):
            d.append(v)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return float(est), float(lo), float(hi)


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 5) -> list[dict]:
    """Equal-count bins of predicted probability: mean prediction, observed frequency, count."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            out.append(
                {
                    "mean_predicted": float(p[chunk].mean()),
                    "observed": float(y[chunk].mean()),
                    "n": int(len(chunk)),
                }
            )
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (treats trials as independent)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def rank_uniformity(ranks: np.ndarray, bins: int = 10) -> dict:
    """Histogram of ranks in [0, 1] and a chi-square test of uniformity (simulation-based
    calibration, Talts et al. 2018). Small p-values mean the posterior is mis-calibrated."""
    counts, _ = np.histogram(ranks, bins=bins, range=(0.0, 1.0))
    stat, p = chisquare(counts) if counts.sum() > 0 else (float("nan"), float("nan"))
    return {"counts": counts.tolist(), "chi2": float(stat), "p_value": float(p)}


def interval_coverage(lower: np.ndarray, upper: np.ndarray, value: np.ndarray) -> float:
    lower, upper, value = (np.asarray(x, float) for x in (lower, upper, value))
    return float(np.mean((lower <= value) & (value <= upper)))
