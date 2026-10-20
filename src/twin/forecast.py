"""Forecasts from weighted library runs: 28-day mortality and predictive intervals.

A forecast is a set of weights over the library's runs (twin.calibrate.Posterior.run_w). The
weights already condition on the patient being alive at the decision time. Quantities at a
target day are summarised among the runs that are alive at that day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twin.calibrate import Posterior
from twin.librarydata import TARGET_DAYS, TARGET_VARS, LibraryData

LEVELS = (0.5, 0.8, 0.95)  # central predictive intervals


@dataclass(frozen=True)
class Forecast:
    mortality: float  # P(dead by the end of the 28-day horizon | alive at the decision time)
    p_alive: dict[int, float]  # day -> P(alive at that day | alive at the decision time)
    median: dict[tuple[str, int], float]
    # (variable, day) -> {level: (lower, upper)}, among runs alive at that day
    intervals: dict[tuple[str, int], dict[float, tuple[float, float]]]


class ForecastEngine:
    """Presorts the library once so weighted quantiles are cheap for many patients."""

    def __init__(self, lib: LibraryData):
        self.lib = lib
        self.died = lib.died.reshape(-1)
        n_t, n_v = lib.targets.shape[2], lib.targets.shape[3]
        flat = lib.targets.reshape(-1, n_t, n_v)
        self.valid = ~np.isnan(flat[:, :, 0])  # (R, n_t): run has a record at the target step
        self.order: dict[tuple[int, int], np.ndarray] = {}
        self.sorted_vals: dict[tuple[int, int], np.ndarray] = {}
        for ti in range(n_t):
            idx = np.flatnonzero(self.valid[:, ti])
            for vi in range(n_v):
                vals = flat[idx, ti, vi]
                o = np.argsort(vals, kind="stable")
                self.order[(ti, vi)] = idx[o]
                self.sorted_vals[(ti, vi)] = vals[o].astype(np.float64)

    def _quantiles(self, w: np.ndarray, ti: int, vi: int, qs: np.ndarray) -> np.ndarray:
        ws = w[self.order[(ti, vi)]]
        total = ws.sum()
        if total <= 0:
            return np.full(len(qs), np.nan)
        c = np.cumsum(ws) / total
        i = np.minimum(np.searchsorted(c, qs, side="left"), len(c) - 1)
        return self.sorted_vals[(ti, vi)][i]

    def forecast(self, post: Posterior, variables: list[str] | None = None) -> Forecast:
        w = post.run_w.reshape(-1)
        p_alive = {d: float(w[self.valid[:, ti]].sum()) for ti, d in enumerate(TARGET_DAYS)}
        mortality = float(w[self.died].sum())
        qs = np.array(
            sorted({(1 - lv) / 2 for lv in LEVELS} | {(1 + lv) / 2 for lv in LEVELS} | {0.5})
        )
        median, intervals = {}, {}
        for vi, var in enumerate(TARGET_VARS):
            if variables is not None and var not in variables:
                continue
            for ti, day in enumerate(TARGET_DAYS):
                q = dict(zip(qs.round(4), self._quantiles(w, ti, vi, qs), strict=True))
                median[(var, day)] = float(q[0.5])
                intervals[(var, day)] = {
                    lv: (float(q[round((1 - lv) / 2, 4)]), float(q[round((1 + lv) / 2, 4)]))
                    for lv in LEVELS
                }
        return Forecast(mortality, p_alive, median, intervals)


def theta_summary(post: Posterior, lib: LibraryData, levels=LEVELS) -> dict[str, dict]:
    """Posterior mean and central intervals of each parameter from the design-point weights."""
    from twin.params import PARAM_NAMES

    out = {}
    for j, name in enumerate(PARAM_NAMES):
        v = lib.theta[:, j]
        o = np.argsort(v, kind="stable")
        c = np.cumsum(post.theta_w[o])
        mean = float(np.sum(post.theta_w * v))
        iv = {}
        for lv in levels:
            lo = v[o][min(int(np.searchsorted(c, (1 - lv) / 2)), len(c) - 1)]
            hi = v[o][min(int(np.searchsorted(c, (1 + lv) / 2)), len(c) - 1)]
            iv[lv] = (float(lo), float(hi))
        var = float(np.sum(post.theta_w * (v - mean) ** 2))
        out[name] = {"mean": mean, "var": var, "intervals": iv}
    return out
