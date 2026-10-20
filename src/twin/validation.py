"""Validation of the twin against the hidden truth: parameter recovery, forecast scores and
coverage, measurement-design and misspecification studies. Evaluation code: it reads the truth
through twin.evaluate and is never imported by calibration, forecasting or treatment code.

Everything runs on a list of patient ids with the settings frozen in config/calibration.yaml.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np

from twin import metrics
from twin.calibrate import (
    CalibrationConfig,
    Posterior,
    compute_terms,
    ga_posterior,
    posterior_from_terms,
    prior_only,
)
from twin.evaluate import (
    prior_variance,
    regenerate_observations,
    score_patient,
)
from twin.forecast import LEVELS, ForecastEngine
from twin.jobs.calibrate_ga import load_fit
from twin.librarydata import TARGET_DAYS, LibraryData
from twin.observe import MeasurementModel, Observations, load_observations, subset_observations
from twin.params import PARAM_NAMES
from twin.truth import read_truth

SMALL_PANEL = ["TNF", "IL10", "IL8"]  # fixed before any result was seen
BOOT = 2000


def _lv(d: dict, level: float):
    """Coverage dicts have float keys in memory and string keys after a JSON round trip."""
    return d[level] if level in d else d[str(level)]


# --------------------------------------------------------------------------------------------
# Per-patient scores for one method
# --------------------------------------------------------------------------------------------
def score_library_method(
    ids: list[str],
    lib: LibraryData,
    engine: ForecastEngine,
    cfg: CalibrationConfig,
    obs_for: Callable[[str], Observations],
    lib_for: Callable[[], LibraryData] | None = None,
) -> list[dict]:
    """Score the library posterior with `cfg` on each patient (observations may be altered)."""
    pv = prior_variance(lib)
    out = []
    for pid in ids:
        obs = obs_for(pid)
        use_lib = lib_for() if lib_for else lib
        post = posterior_from_terms(compute_terms(obs, use_lib, cfg), use_lib, cfg)
        out.append(score_patient(read_truth(pid), post, engine.forecast(post), lib, pv))
    return out


def score_prior_only(ids: list[str], lib: LibraryData, engine: ForecastEngine) -> list[dict]:
    post = prior_only(lib)
    fc = engine.forecast(post)
    pv = prior_variance(lib)
    return [score_patient(read_truth(pid), post, fc, lib, pv) for pid in ids]


def score_ga_method(
    ids: list[str],
    fits_dir: Path,
    cfg: CalibrationConfig,
    delta: float,
    kappa: float,
    lib: LibraryData,
) -> list[dict]:
    pv = prior_variance(lib)
    out = []
    for pid in ids:
        fit = load_fit(fits_dir / f"{pid}.npz", pid)
        flib, post = ga_posterior(fit, delta, cfg, kappa)
        out.append(
            score_patient(read_truth(pid), post, ForecastEngine(flib).forecast(post), flib, pv)
        )
    return out


# --------------------------------------------------------------------------------------------
# Summaries with uncertainty
# --------------------------------------------------------------------------------------------
def mortality_arrays(scores: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([s["mortality_p"] for s in scores]),
        np.array([s["died"] for s in scores], dtype=float),
    )


def summarise_method(scores: list[dict], seed: int = 0) -> dict:
    p, y = mortality_arrays(scores)
    out: dict = {"n_patients": len(scores)}
    out["mortality"] = {
        "brier": metrics.bootstrap_ci(metrics.brier, p, y, n_boot=BOOT, seed=seed),
        "log_loss": metrics.bootstrap_ci(metrics.log_loss, p, y, n_boot=BOOT, seed=seed),
        "auroc": metrics.bootstrap_ci(metrics.auroc, p, y, n_boot=BOOT, seed=seed),
        "mean_forecast": float(p.mean()),
        "observed": float(y.mean()),
        "reliability": metrics.reliability(p, y, bins=5),
    }
    cov: dict = {}
    for lv in LEVELS:
        hits = [h[lv] for s in scores for h in s["forecast"].values()]
        k, n = int(np.sum(hits)), len(hits)
        cov[str(lv)] = {"coverage": k / max(n, 1), "n": n, "wilson95": metrics.wilson(k, n)}
    out["forecast_coverage"] = cov
    by_day, by_var = {}, {}
    for day in TARGET_DAYS:
        hits = [h[0.8] for s in scores for (v, d), h in s["forecast"].items() if d == day]
        by_day[str(day)] = {
            "coverage80": float(np.mean(hits)) if hits else float("nan"),
            "n": len(hits),
        }
    for var in sorted({v for s in scores for (v, _) in s["forecast"]}):
        hits = [h[0.8] for s in scores for (v, d), h in s["forecast"].items() if v == var]
        by_var[var] = {"coverage80": float(np.mean(hits)), "n": len(hits)}
    out["forecast_coverage_by_day"], out["forecast_coverage_by_variable"] = by_day, by_var
    theta = {}
    for name in PARAM_NAMES:
        covered = {lv: [s["theta"][name]["covered"][lv] for s in scores] for lv in LEVELS}
        ranks = np.array([s["theta"][name]["rank"] for s in scores])
        theta[name] = {
            "coverage": {
                str(lv): {
                    "coverage": float(np.mean(v)),
                    "wilson95": metrics.wilson(int(np.sum(v)), len(v)),
                }
                for lv, v in covered.items()
            },
            "rank_test": metrics.rank_uniformity(ranks, bins=10),
            "contraction_mean": float(np.mean([s["theta"][name]["contraction"] for s in scores])),
            "posterior_mean": [s["theta"][name]["mean"] for s in scores],
        }
    out["theta"] = theta
    out["ess_theta_median"] = float(np.median([s["ess_theta"] for s in scores]))
    out["ess_theta_min"] = float(np.min([s["ess_theta"] for s in scores]))
    return out


def paired_vs_baseline(scores: list[dict], baseline: list[dict], seed: int = 0) -> dict:
    """Brier, log loss and AUROC difference (method - baseline) with paired bootstrap CIs."""
    p, y = mortality_arrays(scores)
    pb, _ = mortality_arrays(baseline)

    def diff(stat):
        return metrics.paired_bootstrap_diff(stat, (p, y), (pb, y), n_boot=BOOT, seed=seed)

    return {
        "brier_diff": diff(metrics.brier),
        "log_loss_diff": diff(metrics.log_loss),
        "auroc_diff": diff(metrics.auroc),
    }


def truth_vs_mean(ids: list[str], summary: dict) -> dict:
    """Posterior means and true values per parameter (for scatter plots and error summaries)."""
    out = {}
    for name in PARAM_NAMES:
        true = np.array([getattr(read_truth(i).theta, name) for i in ids], dtype=float)
        mean = np.array(summary["theta"][name]["posterior_mean"], dtype=float)
        out[name] = {
            "truth": true.tolist(),
            "mean": mean.tolist(),
            "rmse": float(np.sqrt(np.mean((mean - true) ** 2))),
            "bias": float(np.mean(mean - true)),
            "corr": float(np.corrcoef(true, mean)[0, 1]) if np.std(mean) > 0 else float("nan"),
        }
    return out


# --------------------------------------------------------------------------------------------
# Measurement design and misspecification (library re-weighting only)
# --------------------------------------------------------------------------------------------
def design_variants(panel: tuple[str, ...]) -> list[dict]:
    """Pre-specified measurement designs. Readings are indexed 0..11 (steps 51..612)."""
    full = list(range(12))
    base = {"readings": full, "panel": list(panel), "data_cv": 0.25, "assumed_cv": 0.25}
    v = [dict(base, name="baseline: every 6 h, last 72 h, CV 25%, 8 cytokines")]
    v.append(dict(base, name="sampling every 12 h", readings=[1, 3, 5, 7, 9, 11]))
    v.append(dict(base, name="sampling every 24 h", readings=[3, 7, 11]))
    v.append(dict(base, name="window: last 48 h", readings=list(range(4, 12))))
    v.append(dict(base, name="window: last 24 h", readings=list(range(8, 12))))
    for cv in (0.10, 0.50):
        v.append(dict(base, name=f"assay noise CV {int(cv * 100)}%", data_cv=cv, assumed_cv=cv))
    for c in panel:
        v.append(dict(base, name=f"leave out {c}", panel=[x for x in panel if x != c]))
    v.append(dict(base, name=f"small panel ({', '.join(SMALL_PANEL)})", panel=SMALL_PANEL))
    return v


def observations_for_design(
    pid: str, design: dict, lod: dict, full_panel: tuple[str, ...]
) -> Observations:
    model = MeasurementModel(lod=lod, noise_cv=design["data_cv"], panel=full_panel)
    obs = regenerate_observations(pid, model)
    obs = subset_observations(obs, design["readings"], design["panel"])
    return dataclasses.replace(obs, noise_cv=design["assumed_cv"])  # what the twin assumes


def score_design(
    ids: list[str], lib: LibraryData, engine: ForecastEngine, cfg: CalibrationConfig, design: dict,
    lod: dict,
) -> list[dict]:  # fmt: skip
    small = lib.restrict(design["readings"], design["panel"])
    return score_library_method(
        ids, lib, engine, cfg,
        obs_for=lambda pid: observations_for_design(pid, design, lod, lib.panel),
        lib_for=lambda: small,
    )  # fmt: skip


def design_row(scores: list[dict], baseline: list[dict]) -> dict:
    p, y = mortality_arrays(scores)
    pb, _ = mortality_arrays(baseline)
    cov = {n: float(np.mean([s["theta"][n]["covered"][0.8] for s in scores])) for n in PARAM_NAMES}
    fcov = float(np.mean([h[0.8] for s in scores for h in s["forecast"].values()]))
    return {
        "brier": metrics.brier(p, y),
        "brier_diff_vs_baseline": metrics.paired_bootstrap_diff(
            metrics.brier, (p, y), (pb, y), n_boot=500
        ),
        "forecast_coverage80": fcov,
        "theta_coverage80": cov,
        "contraction": {
            n: float(np.mean([s["theta"][n]["contraction"] for s in scores])) for n in PARAM_NAMES
        },
        "ess_median": float(np.median([s["ess_theta"] for s in scores])),
    }  # fmt: skip


__all__ = [
    "Posterior", "design_variants", "load_observations", "score_design", "score_ga_method",
    "score_library_method", "score_prior_only", "summarise_method", "truth_vs_mean",
]  # fmt: skip
