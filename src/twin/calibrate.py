"""Calibrate the twin to one patient's observations, using the simulation library.

`library_posterior` turns readings into weights over the library's design points (theta level)
and over its individual runs (run level). The design points are a scrambled Sobol sample of the
prior, so the prior weights are equal and the posterior is just likelihood weights.

Only an `Observations` object goes in; nothing else about the patient is available here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from twin import seeds
from twin.config import DECISION_STEP, HORIZON_STEPS, STEPS_PER_READING
from twin.engine import Result, run_batch
from twin.ga import GA, GeneSpec
from twin.librarydata import LibraryData, from_results
from twin.likelihood import (
    LikelihoodConfig,
    Prepared,
    prepare,
    run_reading_loglik,
    synthetic_terms,
)
from twin.observe import Observations
from twin.params import INT_PARAMS, PARAM_NAMES, Prior


@dataclass(frozen=True)
class CalibrationConfig:
    """How readings become weights. Chosen on the development cohort (config/calibration.yaml).

    theta_method  "synthetic": Gaussian synthetic likelihood per design point (tempered by tau);
                  "mixture": run-mixture likelihood (tempered by tau, applied to each run).
    forecast_level "theta": mix a design point's alive runs equally; "run": weight each run by its
                  own match to the readings (tempered by tau_run) within the design point.
    """

    theta_method: str = "synthetic"
    forecast_level: str = "run"
    tau: float = 1.0
    tau_run: float = 1.0
    var_floor: float = 0.25
    survival_floor: float = 1e-3

    def likelihood(self) -> LikelihoodConfig:
        return LikelihoodConfig(
            tau=self.tau, var_floor=self.var_floor, survival_floor=self.survival_floor
        )


@dataclass(frozen=True)
class Posterior:
    theta_w: np.ndarray  # (N,) weights over design points, sum 1
    run_w: np.ndarray  # (N, K) weights over library runs for forecasting, sum 1
    config: CalibrationConfig

    @property
    def ess_theta(self) -> float:
        return float(1.0 / np.sum(self.theta_w**2))

    @property
    def ess_run(self) -> float:
        return float(1.0 / np.sum(self.run_w**2))


def _softmax(logw: np.ndarray, axis=None) -> np.ndarray:
    return np.exp(logw - logsumexp(logw, axis=axis, keepdims=True))


def prior_only(lib: LibraryData) -> Posterior:
    """The population forecast: every library run alive at the decision time, equally weighted."""
    run_w = lib.alive_td.astype(float)
    run_w /= run_w.sum()
    return Posterior(run_w.sum(axis=1), run_w, CalibrationConfig(theta_method="prior"))


@dataclass(frozen=True)
class Terms:
    """The expensive, tau-independent parts of the likelihood for one patient."""

    surv: np.ndarray  # (N,) survival term of the synthetic likelihood
    reading: np.ndarray  # (N,) synthetic reading sum (untempered)
    run_reading: np.ndarray  # (N, K) each run's own reading log-likelihood (untempered)


def compute_terms(obs: Observations | Prepared, lib: LibraryData, cfg: CalibrationConfig) -> Terms:
    """Only cfg.var_floor and cfg.survival_floor matter here; tau and tau_run are applied later."""
    prep = obs if isinstance(obs, Prepared) else prepare(obs)
    lik = cfg.likelihood()
    surv, reading = synthetic_terms(prep, lib.panel_pre, lib.alive_td, lik)
    return Terms(surv, reading, run_reading_loglik(prep, lib.panel_pre, lik))


def posterior_from_terms(terms: Terms, lib: LibraryData, cfg: CalibrationConfig) -> Posterior:
    alive = lib.alive_td
    if cfg.theta_method == "mixture":
        per_run = np.where(alive, cfg.tau * terms.run_reading, -np.inf)
        run_w = _softmax(per_run)
        return Posterior(run_w.sum(axis=1), run_w, cfg)
    if cfg.theta_method != "synthetic":
        raise ValueError(f"unknown theta_method {cfg.theta_method!r}")
    theta_w = _softmax(terms.surv + cfg.tau * terms.reading)
    if cfg.forecast_level == "theta":
        n_alive = alive.sum(axis=1, keepdims=True)
        within = np.where(n_alive > 0, alive / np.maximum(n_alive, 1), 0.0)
    elif cfg.forecast_level == "run":
        per_run = np.where(alive, cfg.tau_run * terms.run_reading, -np.inf)
        has = alive.any(axis=1)
        within = np.zeros(per_run.shape)
        within[has] = _softmax(per_run[has], axis=1)
    else:
        raise ValueError(f"unknown forecast_level {cfg.forecast_level!r}")
    run_w = theta_w[:, None] * within
    run_w /= run_w.sum()
    return Posterior(theta_w, run_w, cfg)


def library_posterior(
    obs: Observations | Prepared, lib: LibraryData, cfg: CalibrationConfig
) -> Posterior:
    return posterior_from_terms(compute_terms(obs, lib, cfg), lib, cfg)


# ---------------------------------------------------------------------------------------------
# GA calibration: the same objective, but every fitness evaluation runs fresh simulations.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 40
    max_generations: int = 25
    k_seeds: int = 8  # fresh simulations per candidate (common random numbers: same seeds always)
    patience: int = 5
    tol: float = 0.5  # log-likelihood units
    elite: int = 2
    forecast_top: int = 64  # evaluated thetas (highest likelihood) simulated for the full horizon
    forecast_seeds: int = 8
    seed: int = 0  # the GA's own random generator


@dataclass
class GAFit:
    """Everything a GA calibration produced, enough to choose Delta afterwards without new runs."""

    patient_id: str
    evaluated_theta: np.ndarray  # (E, 5) every distinct theta evaluated, in order
    evaluated_ll: np.ndarray  # (E,) tempered synthetic log-likelihood
    best_by_generation: list[float]
    generations: int
    stopped_early: bool
    n_sims_search: int
    forecast_theta: np.ndarray  # (M, 5) the top-M thetas simulated for the full horizon
    forecast_ll: np.ndarray  # (M,)
    forecast_lib: LibraryData  # full-horizon runs of those thetas

    @property
    def n_sims(self) -> int:
        return self.n_sims_search + self.forecast_lib.n * self.forecast_lib.k


def gene_specs(prior: Prior) -> list[GeneSpec]:
    specs = []
    for name in PARAM_NAMES:
        lo, hi = prior.bounds[name]
        if name in INT_PARAMS:
            specs.append(GeneSpec(name, "int", lo, hi, 1))
        else:
            specs.append(GeneSpec(name, "float", lo, hi, (hi - lo) / 27.0))
    return specs


def _panel_array(results: list[Result], panel: tuple[str, ...]) -> np.ndarray:
    """(K, T, P) panel values at readings 51..612 from a candidate's results; NaN if dead."""
    n_pre = DECISION_STEP // STEPS_PER_READING
    out = np.full((len(results), n_pre, len(panel)), np.nan)
    for k, r in enumerate(results):
        m = min(len(r.step), n_pre)
        for j, c in enumerate(panel):
            out[k, :m, j] = r.series[c][:m]
    return out


def _alive_at_decision(results: list[Result]) -> np.ndarray:
    return np.array([(not r.died) or (r.death_step or 0) > DECISION_STEP for r in results])


def ga_calibrate(
    obs: Observations,
    prior: Prior,
    cal: CalibrationConfig,
    ga_cfg: GAConfig | None = None,
    workers: int | None = None,
    bin_path: str | None = None,
) -> GAFit:
    """Search for parameter sets that explain the readings, with fresh simulations.

    Fitness of a candidate = the tempered synthetic log-likelihood from `ga_cfg.k_seeds` runs of
    the first 72 h with fixed calibration seeds (common random numbers). The best
    `forecast_top` evaluated candidates are then simulated for the full horizon.
    """
    ga_cfg = ga_cfg or GAConfig()
    prep = prepare(obs)
    lik = cal.likelihood()
    pid = obs.patient_id
    calib_seeds = [seeds.calibration_seed(pid, k) for k in range(ga_cfg.k_seeds)]
    n_sims = 0

    def evaluate(batch: list[tuple]) -> list[float]:
        nonlocal n_sims
        cfgs = []
        for i, ind in enumerate(batch):
            th = dict(zip(PARAM_NAMES, ind, strict=True))
            for k, sd in enumerate(calib_seeds):
                cfgs.append(
                    dict(th, id=f"{i}/{k}", seed=sd, num_steps=DECISION_STEP,
                         output_stride=STEPS_PER_READING, outputs=list(obs.panel))
                )  # fmt: skip
        res = run_batch(cfgs, n_workers=workers, chunk_size=1, bin_path=bin_path)
        n_sims += len(cfgs)
        out = []
        for i in range(len(batch)):
            rs = res[i * ga_cfg.k_seeds : (i + 1) * ga_cfg.k_seeds]
            surv, reading = synthetic_terms(
                prep, _panel_array(rs, obs.panel)[None], _alive_at_decision(rs)[None], lik
            )
            out.append(float(surv[0] + cal.tau * reading[0]))
        return out

    ga = GA(gene_specs(prior), pop_size=ga_cfg.pop_size, elite=ga_cfg.elite, seed=ga_cfg.seed)
    res = ga.run(evaluate, ga_cfg.max_generations, ga_cfg.patience, ga_cfg.tol)
    thetas = np.array([[float(v) for v in ind] for ind, _ in res.evaluated])
    lls = np.array([f for _, f in res.evaluated])

    top = np.argsort(lls)[::-1][: ga_cfg.forecast_top]
    cfgs = []
    for a, i in enumerate(top):
        th = dict(zip(PARAM_NAMES, res.evaluated[i][0], strict=True))
        for k in range(ga_cfg.forecast_seeds):
            cfgs.append(
                dict(th, id=f"f{a}/{k}", num_steps=HORIZON_STEPS, output_stride=STEPS_PER_READING,
                     seed=seeds.calibration_seed(pid, 1000 + a * ga_cfg.forecast_seeds + k))
            )  # fmt: skip
    fres = run_batch(cfgs, n_workers=workers, chunk_size=1, bin_path=bin_path)
    grid = [
        fres[a * ga_cfg.forecast_seeds : (a + 1) * ga_cfg.forecast_seeds] for a in range(len(top))
    ]
    return GAFit(
        patient_id=pid,
        evaluated_theta=thetas,
        evaluated_ll=lls,
        best_by_generation=res.best_by_generation,
        generations=res.generations,
        stopped_early=res.stopped_early,
        n_sims_search=n_sims,
        forecast_theta=thetas[top],
        forecast_ll=lls[top],
        forecast_lib=from_results(thetas[top], grid, obs.panel),
    )


def ga_posterior(
    fit: GAFit, delta: float, cfg: CalibrationConfig, kappa: float = 1.0
) -> tuple[LibraryData, Posterior]:
    """Weights over the GA's full-horizon runs: candidates within `delta` log-likelihood units of
    the best, weighted by exp(kappa x log-likelihood difference). The GA's samples are not a draw
    from the prior, so these are likelihood weights only and tend to be too narrow; kappa <= 1
    tempers them further. delta and kappa are chosen on the development cohort."""
    lib = fit.forecast_lib
    ll = fit.forecast_ll
    keep = ll >= ll.max() - delta
    theta_w = np.where(keep, np.exp(kappa * (ll - ll.max())), 0.0)
    theta_w /= theta_w.sum()
    alive = lib.alive_td
    n_alive = alive.sum(axis=1, keepdims=True)
    within = np.where(n_alive > 0, alive / np.maximum(n_alive, 1), 0.0)
    run_w = theta_w[:, None] * within
    run_w /= run_w.sum()
    return lib, Posterior(theta_w, run_w, cfg)
