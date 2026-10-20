"""Build a tiny cohort with the real engine (scratch directories; see conftest.py)."""

import os

import pytest

from twin import seeds
from twin.config import DECISION_STEP
from twin.observe import MeasurementModel, load_observations
from twin.params import Prior
from twin.patients import alive_at_decision, build_cohorts, candidate
from twin.truth import list_patients, read_truth

pytestmark = pytest.mark.skipif(not os.environ.get("IIRABM_BIN"), reason="no engine binary")

PRIOR = Prior(
    {
        "inj_number": (10, 50),
        "oxyHeal": (0.03, 0.30),
        "infectSpread": (2, 12),
        "numRecurInj": (1, 6),
        "numInfectRepeat": (1, 6),
    }
)
LOD = {c: 1.0 for c in ["TNF", "IL1", "IL1ra", "IL10", "IL8", "GCSF", "IL12", "IFNg"]}


def test_candidates_are_deterministic_and_in_the_patient_seed_range():
    a, b = candidate(7, PRIOR), candidate(7, PRIOR)
    assert a == b and a != candidate(8, PRIOR)
    assert seeds.role_of(a[1]) == "patient"


def test_small_cohorts_are_valid_and_reproducible():
    ids = build_cohorts(2, 2, prior=PRIOR, model=MeasurementModel(lod=LOD), workers=4, batch=4)
    assert ids == {"development": ["dev-001", "dev-002"], "test": ["test-001", "test-002"]}
    assert list_patients("development") == ids["development"]
    assert list_patients("test") == ids["test"]
    first = {}
    for pid in ids["development"] + ids["test"]:
        t = read_truth(pid)
        first[pid] = (t.theta, t.seed)
        assert seeds.role_of(t.seed) == "patient"  # never a library, calibration or future seed
        assert not t.died or t.death_step > DECISION_STEP  # alive at the decision time
        assert t.record_steps[0] == 51 and PRIOR.contains(t.theta)
        obs = load_observations(pid)
        assert obs.steps[-1] == DECISION_STEP == 612 and len(obs.steps) == 12
    assert len({s for _, s in first.values()}) == 4  # distinct seeds
    assert len({t.oxyHeal for t, _ in first.values()}) == 4  # continuous, so never on a design grid
    # Rebuilding gives the same patients.
    build_cohorts(2, 2, prior=PRIOR, model=MeasurementModel(lod=LOD), workers=4, batch=4)
    for pid, (theta, seed) in first.items():
        t = read_truth(pid)
        assert (t.theta, t.seed) == (theta, seed)


def test_alive_at_decision_rule():
    class R:
        def __init__(self, died, death_step):
            self.died, self.death_step = died, death_step

    assert alive_at_decision(R(False, None))
    assert alive_at_decision(R(True, 613))
    assert not alive_at_decision(R(True, 612))  # no record is taken at the death step
    assert not alive_at_decision(R(True, 300))
