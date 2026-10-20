import numpy as np

from twin.config import DECISION_STEP
from twin.observe import MeasurementModel, Observations, make_observations, measurement_rng

PANEL = ("TNF", "IL1", "IL1ra", "IL10", "IL8", "GCSF", "IL12", "IFNg")
LOD = dict.fromkeys(PANEL, 1.0)
STEPS = np.arange(51, 5760 + 1, 51)


def series(fn):
    return {c: fn(STEPS.astype(float)) for c in PANEL}


def model(**kw):
    return MeasurementModel(lod=LOD, **kw)


def test_reproducible_from_patient_id():
    s = series(lambda t: 5 + 0 * t)
    a = make_observations("dev-001", s, STEPS, model())
    b = make_observations("dev-001", s, STEPS, model())
    c = make_observations("dev-002", s, STEPS, model())
    assert np.array_equal(a.values, b.values, equal_nan=True)
    assert not np.array_equal(a.values, c.values, equal_nan=True)


def test_twelve_readings_up_to_the_decision_time():
    obs = make_observations("p", series(lambda t: 5 + 0 * t), STEPS, model())
    assert list(obs.steps) == list(range(51, 612 + 1, 51)) and len(obs.steps) == 12
    assert obs.steps[-1] == DECISION_STEP == 612
    assert obs.values.shape == (12, 8)


def test_reading_at_step_s_comes_from_the_truth_labelled_s():
    # With negligible noise a reading equals the truth value at its own label, not a neighbour.
    obs = make_observations("p", series(lambda t: t + 1000), STEPS, model(noise_cv=1e-9))
    for i, s in enumerate(obs.steps):
        assert np.allclose(obs.values[i], s + 1000, rtol=1e-6)


def test_twin_is_given_nothing_after_the_decision_time():
    obs = make_observations("p", series(lambda t: t + 1000), STEPS, model())
    assert obs.steps.max() <= obs.decision_step
    assert not any("future" in k or "truth" in k for k in obs.to_json())
    assert set(obs.to_json()) == {
        "patient_id", "panel", "steps", "values", "censored", "missing", "lod",
        "noise_cv", "decision_step",
    }  # fmt: skip


def test_censored_readings_are_flagged_never_zeros():
    s = series(lambda t: np.where(t < 300, 0.2, 50.0))  # below the LOD of 1.0 early on
    obs = make_observations("p", s, STEPS, model(noise_cv=0.05))
    early = obs.steps < 300
    assert obs.censored[early].all() and not obs.censored[~early].any()
    assert np.isnan(obs.values[obs.censored]).all()
    assert not np.any(obs.values == 0)
    assert np.all(obs.values[~obs.censored & ~obs.missing] >= obs.lod.min())


def test_noise_is_mean_preserving_lognormal_with_the_requested_cv():
    s = series(lambda t: 100 + 0 * t)
    vals = []
    for i in range(400):
        vals.append(make_observations(f"p{i}", s, STEPS, model(noise_cv=0.25)).values.ravel())
    v = np.concatenate(vals)
    assert abs(v.mean() / 100 - 1) < 0.01
    assert abs(v.std() / v.mean() - 0.25) < 0.01


def test_missing_readings_are_flagged_and_independent_of_censoring():
    s = series(lambda t: 50 + 0 * t)
    obs = make_observations("p", s, STEPS, model(missing_fraction=0.3))
    assert 0.15 < obs.missing.mean() < 0.45
    assert np.isnan(obs.values[obs.missing]).all()
    assert not (obs.censored & obs.missing).any()


def test_json_roundtrip():
    s = series(lambda t: np.where(t < 300, 0.2, 50.0))
    obs = make_observations("p", s, STEPS, model(missing_fraction=0.1))
    back = Observations.from_json(obs.to_json())
    assert np.array_equal(back.values, obs.values, equal_nan=True)
    assert np.array_equal(back.censored, obs.censored) and np.array_equal(back.missing, obs.missing)


def test_measurement_rng_is_independent_of_engine_seeds():
    a = measurement_rng("dev-001").random()
    assert a == measurement_rng("dev-001").random()
