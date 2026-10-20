"""Library posterior and GA calibration, on the real library when present and the real engine."""

import os
from pathlib import Path

import numpy as np
import pytest

from twin.calibrate import (
    CalibrationConfig,
    GAConfig,
    ga_calibrate,
    ga_posterior,
    library_posterior,
    prior_only,
)
from twin.librarydata import LibraryData
from twin.observe import MeasurementModel, make_observations
from twin.params import Prior

PRIOR = Prior(
    {
        "inj_number": (10, 50),
        "oxyHeal": (0.03, 0.30),
        "infectSpread": (2, 12),
        "numRecurInj": (1, 6),
        "numInfectRepeat": (1, 6),
    }
)
LOD = {c: 0.5 for c in ["TNF", "IL1", "IL1ra", "IL10", "IL8", "GCSF", "IL12", "IFNg"]}


def synthetic_library(n=40, k=4, seed=0) -> LibraryData:
    """A tiny made-up library: panel values depend on theta[0] (injury), a few runs die."""
    rng = np.random.default_rng(seed)
    theta = np.stack(
        [
            np.linspace(10, 50, n),
            np.full(n, 0.1),
            np.full(n, 5.0),
            np.full(n, 2.0),
            np.full(n, 2.0),
        ],
        1,
    )
    base = 5 + 0.05 * theta[:, 0]
    panel_pre = np.exp(base[:, None, None, None] + rng.normal(0, 0.1, (n, k, 12, 8))).astype(
        np.float32
    )
    died = np.zeros((n, k), bool)
    death_step = np.full((n, k), -1, np.int32)
    died[-5:, 0] = True
    death_step[-5:, 0] = 300
    panel_pre[-5:, 0, 6:] = np.nan
    alive = ~died | (death_step > 612)
    targets = np.exp(base[:, None, None, None] + rng.normal(0, 0.1, (n, k, 4, 9))).astype(
        np.float32
    )
    return LibraryData(theta, tuple(LOD), panel_pre, targets, alive, died, death_step)


def obs_at(lib: LibraryData, idx: int):
    truth = {c: lib.panel_pre[idx, 1, :, j] for j, c in enumerate(lib.panel)}
    steps = np.arange(1, 13) * 51
    return make_observations("dev-001", truth, steps, MeasurementModel(lod=LOD, noise_cv=0.1))


def test_posterior_concentrates_near_the_true_design_point_and_weights_are_normalised():
    lib = synthetic_library()
    obs = obs_at(lib, 25)
    post = library_posterior(
        obs, lib, CalibrationConfig(tau=1.0, forecast_level="run", tau_run=0.3)
    )
    assert abs(post.theta_w.sum() - 1) < 1e-12 and abs(post.run_w.sum() - 1) < 1e-12
    assert (post.run_w[~lib.alive_td] == 0).all()  # runs dead before T_d carry no weight
    assert abs(lib.theta[np.argmax(post.theta_w), 0] - lib.theta[25, 0]) < 3
    assert post.ess_theta < 20  # concentrated
    tempered = library_posterior(obs, lib, CalibrationConfig(tau=0.01))
    assert tempered.ess_theta > post.ess_theta  # tempering widens


def test_mixture_and_theta_level_variants_run_and_prior_only_is_uniform_over_alive_runs():
    lib = synthetic_library()
    obs = obs_at(lib, 10)
    for cfg in (
        CalibrationConfig(theta_method="mixture", tau=0.1),
        CalibrationConfig(forecast_level="theta"),
    ):
        post = library_posterior(obs, lib, cfg)
        assert abs(post.run_w.sum() - 1) < 1e-12
    p = prior_only(lib)
    assert (
        np.allclose(p.run_w[lib.alive_td], 1 / lib.alive_td.sum())
        and (p.run_w[~lib.alive_td] == 0).all()
    )


@pytest.mark.skipif(not os.environ.get("IIRABM_BIN"), reason="no engine binary")
def test_ga_calibration_end_to_end_tiny():
    from twin.engine import run_batch
    from twin.observe import MeasurementModel as MM

    th = dict(inj_number=25, oxyHeal=0.12, infectSpread=5, numRecurInj=2, numInfectRepeat=2)
    (res,) = run_batch(
        [dict(th, id="t", seed=1_234_567, num_steps=5760, output_stride=51)], n_workers=1
    )
    obs = make_observations("dev-001", res.series, res.step, MM(lod=LOD))
    cfg = GAConfig(
        pop_size=6, max_generations=2, k_seeds=2, forecast_top=3, forecast_seeds=2, patience=2
    )
    fit = ga_calibrate(obs, PRIOR, CalibrationConfig(tau=0.1), cfg, workers=4)
    assert fit.n_sims_search == len(fit.evaluated_ll) * 2
    assert fit.forecast_lib.n == 3 and fit.forecast_lib.k == 2
    assert np.all(np.diff(np.sort(fit.forecast_ll)[::-1]) <= 0)
    lib, post = ga_posterior(fit, delta=5.0, cfg=CalibrationConfig(tau=0.1))
    assert lib is fit.forecast_lib and abs(post.run_w.sum() - 1) < 1e-9


REAL = Path(__file__).resolve().parent.parent / "data" / "library" / "condensed.npz"


@pytest.mark.skipif(not REAL.exists(), reason="the real library has not been built (data/library)")
def test_real_library_loads_and_has_the_expected_shape():
    lib = LibraryData.load(REAL)
    assert lib.n == 4096 and lib.k == 16 and abs(lib.alive_td.mean() - 0.80) < 0.01
    assert np.isnan(lib.panel_pre[lib.alive_td]).sum() == 0  # runs alive at T_d have every reading
