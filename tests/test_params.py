import numpy as np

from twin.params import INT_PARAMS, PARAM_NAMES, Prior, Theta

PRIOR = Prior(
    {
        "inj_number": (10, 50),
        "oxyHeal": (0.03, 0.30),
        "infectSpread": (2, 12),
        "numRecurInj": (1, 6),
        "numInfectRepeat": (1, 6),
    }
)


def test_samples_are_inside_the_prior_and_cover_every_integer():
    th = PRIOR.sample(np.random.default_rng(1), 5000)
    assert all(PRIOR.contains(t) for t in th)
    for name in INT_PARAMS:
        lo, hi = PRIOR.bounds[name]
        vals = {getattr(t, name) for t in th}
        assert vals == set(range(int(lo), int(hi) + 1)), name  # inclusive ends are reachable


def test_sobol_extension_keeps_earlier_points():
    a = PRIOR.sobol(32, scramble_seed=7)
    b = PRIOR.sobol(64, scramble_seed=7)
    assert a == b[:32]
    c = PRIOR.sobol(32, scramble_seed=7, start=32)
    assert c == b[32:]


def test_sobol_oxyheal_values_are_distinct_and_cover_the_range():
    th = PRIOR.sobol(512, scramble_seed=1)
    ox = np.array([t.oxyHeal for t in th])
    assert len(set(ox)) == 512
    assert ox.min() < 0.04 and ox.max() > 0.29


def test_theta_roundtrip_and_engine_fields():
    t = Theta(33, 0.1, 5, 2, 2)
    assert Theta.from_array(t.as_array()) == t
    assert set(t.to_engine()) == set(PARAM_NAMES)


def test_patient_theta_is_never_a_design_point():
    design = {t.oxyHeal for t in PRIOR.sobol(2000, scramble_seed=1)}
    patients = {t.oxyHeal for t in PRIOR.sample(np.random.default_rng(9), 2000)}
    assert not (design & patients)
