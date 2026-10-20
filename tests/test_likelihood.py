"""Likelihood tests on a toy simulator with a known answer.

Toy: one parameter theta; a run's panel value is exp(a + theta + b z_k) times a small
independent wobble per reading, where z_k is a persistent per-run effect (so readings within a
run are correlated). Observed = truth x log-normal noise. Values are large so log1p ~ log.
"""

import numpy as np

from twin.likelihood import LikelihoodConfig, mixture_loglik, prepare, synthetic_loglik
from twin.observe import Observations

T, P, K = 12, 4, 64
NOISE_CV = 0.25
SIGMA = float(np.sqrt(np.log1p(NOISE_CV**2)))


def simulate(theta, rng, b):
    """(K, T, P) true values for one candidate theta (fresh random effects)."""
    z = rng.normal(size=(K, 1, 1))
    wobble = rng.normal(scale=0.05, size=(K, T, P))
    return np.exp(6.0 + theta + b * z + wobble)


def simulate_crn(thetas, b, seed):
    """Candidates sharing the same random effects (common random numbers): smooth in theta."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(K, 1, 1))
    wobble = rng.normal(scale=0.05, size=(K, T, P))
    return np.stack([np.exp(6.0 + t + b * z + wobble) for t in thetas])


def make_obs(theta0, rng, b):
    truth = simulate(theta0, rng, b)[0]  # one run plays the patient
    noisy = truth * np.exp(rng.normal(-0.5 * SIGMA**2, SIGMA, size=truth.shape))
    return Observations(
        patient_id="toy", panel=tuple("abcd"), steps=np.arange(1, T + 1) * 51, values=noisy,
        censored=np.zeros((T, P), bool), missing=np.zeros((T, P), bool), lod=np.full(P, 1e-3),
        noise_cv=NOISE_CV, decision_step=T * 51,
    )  # fmt: skip


def posterior_sd(loglik, grid):
    w = np.exp(loglik - loglik.max())
    w /= w.sum()
    m = (w * grid).sum()
    return float(np.sqrt((w * (grid - m) ** 2).sum()))


def test_likelihood_peaks_near_the_truth_and_shapes():
    rng = np.random.default_rng(0)
    grid = np.linspace(-1, 1, 41)
    obs = make_obs(0.2, rng, b=0.3)
    panel = np.stack([simulate(t, rng, 0.3) for t in grid])  # (N, K, T, P)
    alive = np.ones(panel.shape[:2], bool)
    cfg = LikelihoodConfig(tau=1.0)
    syn = synthetic_loglik(obs, panel, alive, cfg)
    mix, per_run = mixture_loglik(obs, panel, alive, cfg)
    assert syn.shape == mix.shape == (41,) and per_run.shape == (41, K)
    assert abs(grid[np.argmax(syn)] - 0.2) < 0.35
    assert abs(grid[np.argmax(mix)] - 0.2) < 0.45


def test_independence_version_is_overconfident_by_a_known_factor_and_the_remedies_fix_it():
    """Persistent run effect b, n = T x P readings. The patient's own effect b z_0 shifts every
    reading together, so the exact posterior sd of theta is sqrt(b^2 + v_r / n) with v_r the
    independent per-reading variance (wobble + measurement noise). Treating the n readings as
    independent gives sqrt((b^2 + v_r) / n): overconfident by a factor of about 6 here."""
    rng = np.random.default_rng(1)
    b, n = 0.4, T * P
    grid = np.linspace(-1.6, 1.6, 321)
    obs = make_obs(0.0, rng, b)
    panel = simulate_crn(grid, b, seed=5)
    alive = np.ones(panel.shape[:2], bool)
    v_r = 0.05**2 + SIGMA**2
    sd_exact = np.sqrt(b**2 + v_r / n)
    sd_independent = np.sqrt((b**2 + v_r) / n)
    assert sd_independent < sd_exact / 4

    sd_syn = posterior_sd(synthetic_loglik(obs, panel, alive, LikelihoodConfig(tau=1.0)), grid)
    assert (
        abs(sd_syn - sd_independent) / sd_independent < 0.25
    )  # it returns the overconfident width

    sd_temp = posterior_sd(synthetic_loglik(obs, panel, alive, LikelihoodConfig(tau=1 / n)), grid)
    assert (
        abs(sd_temp - sd_exact) / sd_exact < 0.25
    )  # tempering by 1/n restores about the right width

    mix, _ = mixture_loglik(obs, panel, alive, LikelihoodConfig(tau=1.0))
    sd_mix = posterior_sd(mix, grid)
    assert abs(sd_mix - sd_exact) / sd_exact < 0.3  # the run-mixture respects the correlation


def test_survival_term_and_dead_runs():
    rng = np.random.default_rng(2)
    obs = make_obs(0.0, rng, b=0.3)
    panel = np.stack([simulate(0.0, rng, 0.3), simulate(0.0, rng, 0.3)])
    alive = np.ones((2, K), bool)
    alive[1, : K // 2] = False  # half the runs of candidate 1 died before the decision time
    panel2 = panel.copy()
    panel2[1, : K // 2, 6:] = np.nan  # died at reading 7
    cfg = LikelihoodConfig(tau=0.0)  # survival term only
    ll = synthetic_loglik(obs, panel2, alive, cfg)
    assert abs((ll[0] - ll[1]) - np.log(2)) < 1e-9
    mix, per_run = mixture_loglik(obs, panel2, alive, LikelihoodConfig())
    assert np.isneginf(per_run[1, : K // 2]).all() and np.isfinite(per_run[1, K // 2 :]).all()
    # all runs dead: finite floor for the synthetic version, -inf for the mixture
    dead = np.zeros((2, K), bool)
    assert np.isfinite(synthetic_loglik(obs, panel, dead, cfg)).all()
    assert np.isneginf(mixture_loglik(obs, panel, dead, cfg)[0]).all()


def test_censored_readings_use_the_cdf_not_a_density_at_zero():
    rng = np.random.default_rng(3)
    base = make_obs(0.0, rng, b=0.2)
    panel = simulate(0.0, rng, 0.2)[None]
    alive = np.ones((1, K), bool)
    cens = base.censored.copy()
    cens[:, 0] = True
    values = np.where(cens, np.nan, base.values)
    obs_c = Observations(
        **{**base.__dict__, "values": values, "censored": cens, "lod": np.full(P, 1e5)}
    )
    # the value is far below the limit of detection of 1e5 only if the model value is: it is not,
    # so a censored reading at a very high limit is *likely*: log P(reading < 1e5) ~ 0
    ll_c = synthetic_loglik(obs_c, panel, alive, LikelihoodConfig())
    assert np.isfinite(ll_c).all()
    obs_low = Observations(**{**obs_c.__dict__, "lod": np.full(P, 1.0)})
    ll_low = synthetic_loglik(obs_low, panel, alive, LikelihoodConfig())
    assert ll_c[0] > ll_low[0]  # being below 1e5 is far more probable than being below 1.0


def test_prepare_converts_to_log1p():
    rng = np.random.default_rng(4)
    obs = make_obs(0.0, rng, 0.2)
    prep = prepare(obs)
    assert np.allclose(prep.y, np.log1p(obs.values))
    assert np.isclose(prep.sigma2, SIGMA**2)
