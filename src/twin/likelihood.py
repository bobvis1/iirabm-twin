"""Likelihoods of one patient's readings given candidate parameter sets, in log1p space.

Two objects, both computed from simulated panel values (K runs per candidate, T readings, P
cytokines; NaN where a run had died before a reading):

* `synthetic_loglik`: a Gaussian synthetic likelihood (Wood, 2010). For every reading time and
  cytokine, the mean and variance of log1p(value) across the K runs, plus the measurement-noise
  variance, give a normal density for the observed reading (a normal CDF for a censored one).
  Readings are treated as independent given theta, which they are not: a run's seed makes its
  deviations from the mean persist over time. The reading term is therefore tempered by tau and
  the across-run variance has a floor (a fraction of the noise variance).
* `mixture_loglik`: the run-mixture likelihood (1/K) sum_k prod_t p(y_t | run k), which respects
  that correlation and uses only the measurement noise. It also returns each run's own
  log-likelihood, which are the run-level weights used for forecasting.

Both include the survival term: the patient is observed alive at the decision time.

Measurement model (twin.observe): reading = value x exp(e), e ~ Normal(-s^2/2, s^2). In log1p
space, to first order, log1p(reading) ~ Normal(log1p(x) - (x/(1+x)) s^2/2, (x/(1+x))^2 s^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, logsumexp

from twin.observe import Observations

LOG_2PI = np.log(2 * np.pi)


@dataclass(frozen=True)
class LikelihoodConfig:
    tau: float = 1.0  # tempering exponent on the reading term, in (0, 1]
    var_floor: float = 0.25  # floor on the across-run variance, as a fraction of the noise variance
    survival_floor: float = 1e-3  # smallest alive-at-decision fraction used in the survival term
    chunk: int = 256  # candidates processed at a time (bounds memory)


@dataclass(frozen=True)
class Prepared:
    """Observations in log1p space."""

    y: np.ndarray  # (T, P) log1p of the readings, NaN if censored or missing
    censored: np.ndarray  # (T, P)
    missing: np.ndarray
    thr: np.ndarray  # (P,) log1p of the limit of detection
    sigma2: float  # variance of log(noise)


def prepare(obs: Observations) -> Prepared:
    return Prepared(
        y=np.log1p(np.where(np.isnan(obs.values), 0.0, obs.values)),
        censored=obs.censored,
        missing=obs.missing,
        thr=np.log1p(obs.lod),
        sigma2=obs.sigma**2,
    )


def _noise_moments(x: np.ndarray, sigma2: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean offset and variance of the log1p reading around log1p(x), for true value x >= 0."""
    w = x / (1.0 + x)
    return -w * sigma2 / 2.0, w * w * sigma2


def _reading_loglik(prep: Prepared, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log-density of each reading under Normal(mu, var); broadcasts over leading axes.

    Observed readings: normal log-density. Censored: log P(reading < limit). Missing: 0.
    """
    sd = np.sqrt(var)
    z = (prep.y - mu) / sd
    dens = -0.5 * (LOG_2PI + 2 * np.log(sd) + z * z)
    cens = log_ndtr((prep.thr - mu) / sd)
    out = np.where(prep.censored, cens, dens)
    return np.where(prep.missing, 0.0, out)


def synthetic_terms(
    obs_or_prep: Observations | Prepared,
    panel: np.ndarray,
    alive_td: np.ndarray,
    cfg: LikelihoodConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """The two parts of the synthetic log-likelihood for N candidates: (survival term, reading sum).

    panel (N, K, T, P): simulated true panel values; alive_td (N, K): run alive at the
    decision time. The log-likelihood is survival + tau * reading, so tau can be swept without
    recomputing the reading sums.
    """
    prep = obs_or_prep if isinstance(obs_or_prep, Prepared) else prepare(obs_or_prep)
    N = panel.shape[0]
    surv = np.empty(N)
    reading = np.empty(N)
    for lo in range(0, N, cfg.chunk):
        sl = slice(lo, min(lo + cfg.chunk, N))
        x = panel[sl].astype(np.float64)  # (n, K, T, P)
        valid = ~np.isnan(x)
        cnt = valid.sum(axis=1)  # (n, T, P)
        z = np.where(valid, np.log1p(np.where(valid, x, 0.0)), 0.0)
        safe = np.maximum(cnt, 1)
        mean_z = z.sum(axis=1) / safe
        mean_x = np.where(valid, x, 0.0).sum(axis=1) / safe
        dev = np.where(valid, z - mean_z[:, None], 0.0)
        var_across = (dev**2).sum(axis=1) / np.maximum(cnt - 1, 1)
        off, nvar = _noise_moments(mean_x, prep.sigma2)
        var = np.maximum(np.maximum(var_across, cfg.var_floor * nvar) + nvar, 1e-12)
        ll = _reading_loglik(prep, mean_z + off, var)  # (n, T, P)
        ll = np.where(cnt > 0, ll, 0.0)  # no run alive at this reading: no information from it
        frac = alive_td[sl].mean(axis=1)
        surv[sl] = np.log(np.maximum(frac, cfg.survival_floor))
        reading[sl] = ll.sum(axis=(1, 2))
    return surv, reading


def synthetic_loglik(
    obs_or_prep: Observations | Prepared,
    panel: np.ndarray,
    alive_td: np.ndarray,
    cfg: LikelihoodConfig,
) -> np.ndarray:
    """Synthetic log-likelihood of the readings for N candidates (see synthetic_terms)."""
    surv, reading = synthetic_terms(obs_or_prep, panel, alive_td, cfg)
    return surv + cfg.tau * reading


def run_reading_loglik(
    obs_or_prep: Observations | Prepared, panel: np.ndarray, cfg: LikelihoodConfig
) -> np.ndarray:
    """Each run's own reading log-likelihood (N, K), measurement noise only, untempered.

    Runs that died before a reading have NaN there; callers mask runs not alive at the decision
    time (their value here is meaningless).
    """
    prep = obs_or_prep if isinstance(obs_or_prep, Prepared) else prepare(obs_or_prep)
    N, K = panel.shape[:2]
    out = np.empty((N, K))
    for lo in range(0, N, cfg.chunk):
        sl = slice(lo, min(lo + cfg.chunk, N))
        x = np.nan_to_num(panel[sl].astype(np.float64), nan=0.0)
        off, nvar = _noise_moments(x, prep.sigma2)
        ll = _reading_loglik(prep, np.log1p(x) + off, np.maximum(nvar, 1e-12))  # (n, K, T, P)
        out[sl] = ll.sum(axis=(2, 3))
    return out


def mixture_loglik(
    obs_or_prep: Observations | Prepared,
    panel: np.ndarray,
    alive_td: np.ndarray,
    cfg: LikelihoodConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run-mixture log-likelihood per candidate, and each run's own log-likelihood (N, K).

    Per-run: tau x log p(readings | run k), counting only the measurement noise; -inf if the run
    had died before the decision time. Per candidate: logsumexp over runs minus log K, which
    already contains the survival term (runs that died contribute zero probability).
    """
    K = panel.shape[1]
    per_run = np.where(alive_td, cfg.tau * run_reading_loglik(obs_or_prep, panel, cfg), -np.inf)
    return logsumexp(per_run, axis=1) - np.log(K), per_run
