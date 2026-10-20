"""Scores calibration and forecasts against the hidden truth. The ONLY module (with the cohort
builder and tests) that reads twin.truth.

Guards keep development and test results apart (CLAUDE.md rule 4):
- `assert_development_only` is called by anything that tunes: test patients are refused;
- `evaluate_cohort` is for the final, frozen evaluation: development patients are refused unless
  `allow_development=True` is passed explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from twin.calibrate import Posterior
from twin.forecast import LEVELS, Forecast, ForecastEngine, theta_summary
from twin.librarydata import TARGET_DAYS, TARGET_STEPS, TARGET_VARS, LibraryData
from twin.observe import Observations, make_observations
from twin.params import INT_PARAMS, PARAM_NAMES
from twin.truth import PatientTruth, read_truth


def _tie_rng(patient_id: str, name: str) -> np.random.Generator:
    import hashlib

    h = hashlib.blake2b(f"rank/{patient_id}/{name}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def regenerate_observations(patient_id: str, model) -> Observations:
    """Draw a patient's readings again from the stored truth under a different measurement model
    (noise level, panel, missingness). With the default model this equals the stored observations;
    the same standard normals are used, so a different noise level rescales the same draws."""
    t = read_truth(patient_id)
    return make_observations(patient_id, t.series, t.record_steps, model)


def assert_development_only(patient_ids: list[str]) -> None:
    bad = [p for p in patient_ids if not p.startswith("dev-")]
    if bad:
        raise ValueError(f"tuning may only use development patients; refused: {bad[:3]}")


def assert_frozen() -> str:
    """Refuse to score the test cohort unless the calibration, prior and measurement settings are
    exactly the committed ones; returns the commit hash they are frozen at."""
    import subprocess

    files = ["config/calibration.yaml", "config/prior.yaml", "config/measurement.yaml"]
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *files], cwd=root).returncode
    if dirty:
        raise RuntimeError(f"settings are not frozen: uncommitted changes in {files}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _check_cohort(patient_ids: list[str], allow_development: bool) -> None:
    dev = [p for p in patient_ids if p.startswith("dev-")]
    if dev and not allow_development:
        raise ValueError(
            "refusing to evaluate development patients as if they were test patients "
            f"({dev[:3]} ...); pass allow_development=True to do it on purpose"
        )


def prior_variance(lib: LibraryData) -> np.ndarray:
    return lib.theta.var(axis=0)


def score_patient(
    truth: PatientTruth,
    post: Posterior,
    fc: Forecast,
    lib: LibraryData,
    prior_var: np.ndarray | None = None,
) -> dict:
    """Coverage indicators, mortality forecast and contraction for one patient.

    `lib` holds the runs the weights refer to (the full library, or a GA calibration's own
    runs); `prior_var` is the prior variance of each parameter (default: that of `lib`).
    """
    th = theta_summary(post, lib)
    prior_var = prior_variance(lib) if prior_var is None else prior_var
    theta = {}
    for j, name in enumerate(PARAM_NAMES):
        true = getattr(truth.theta, name)
        cov = {lv: bool(lo <= true <= hi) for lv, (lo, hi) in th[name]["intervals"].items()}
        v = lib.theta[:, j]
        # rank of the truth among posterior draws in [0, 1]; ties (integer parameters) are broken
        # at random, with a generator seeded by the patient and parameter so it is reproducible
        below = post.theta_w[v < true].sum()
        equal = post.theta_w[v == true].sum() if name in INT_PARAMS else 0.0
        u = _tie_rng(truth.patient_id, name).random()
        theta[name] = {
            "covered": cov,
            "rank": float(below + u * equal),
            "mean": th[name]["mean"],
            "contraction": th[name]["var"] / prior_var[j],
        }
    forecast = {}
    for (var, day), iv in fc.intervals.items():
        step = TARGET_STEPS[TARGET_DAYS.index(day)]
        hit = np.flatnonzero(truth.record_steps == step)
        if len(hit) == 0:  # the patient is dead (or the run ended) before this day
            continue
        value = float(truth.series[var][hit[0]])
        forecast[(var, day)] = {lv: bool(lo <= value <= hi) for lv, (lo, hi) in iv.items()}
    return {
        "patient_id": truth.patient_id,
        "theta": theta,
        "forecast": forecast,
        "mortality_p": fc.mortality,
        "died": bool(truth.died),
        "ess_theta": post.ess_theta,
        "ess_run": post.ess_run,
    }


def summarise(scores: list[dict]) -> dict:
    """Pooled coverage, Brier score and mean contraction over patients."""
    out: dict = {"n_patients": len(scores)}
    out["theta_coverage"] = {
        lv: float(np.mean([s["theta"][p]["covered"][lv] for s in scores for p in PARAM_NAMES]))
        for lv in LEVELS
    }
    out["theta_coverage_by_param"] = {
        p: {lv: float(np.mean([s["theta"][p]["covered"][lv] for s in scores])) for lv in LEVELS}
        for p in PARAM_NAMES
    }
    out["contraction_by_param"] = {
        p: float(np.mean([s["theta"][p]["contraction"] for s in scores])) for p in PARAM_NAMES
    }
    cov = {lv: [] for lv in LEVELS}
    for s in scores:
        for per_level in s["forecast"].values():
            for lv, hit in per_level.items():
                cov[lv].append(hit)
    out["forecast_coverage"] = {lv: float(np.mean(v)) for lv, v in cov.items()}
    out["forecast_n"] = len(cov[LEVELS[0]])
    p = np.array([s["mortality_p"] for s in scores])
    y = np.array([s["died"] for s in scores], dtype=float)
    out["brier"] = float(np.mean((p - y) ** 2))
    out["mortality_mean_forecast"] = float(p.mean())
    out["mortality_observed"] = float(y.mean())
    out["ess_theta_median"] = float(np.median([s["ess_theta"] for s in scores]))
    out["ess_theta_min"] = float(np.min([s["ess_theta"] for s in scores]))
    return out


def evaluate_cohort(
    patient_ids: list[str],
    posts: dict[str, Posterior],
    engine: ForecastEngine,
    lib: LibraryData,
    allow_development: bool = False,
) -> tuple[list[dict], dict]:
    _check_cohort(patient_ids, allow_development)
    scores = []
    for pid in patient_ids:
        scores.append(score_patient(read_truth(pid), posts[pid], engine.forecast(posts[pid]), lib))
    return scores, summarise(scores)


__all__ = [
    "TARGET_VARS",
    "assert_development_only",
    "evaluate_cohort",
    "score_patient",
    "summarise",
]
