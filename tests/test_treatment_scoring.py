import numpy as np
import pytest

from twin import evaluate, seeds
from twin.engine import SERIES_NAMES, Result
from twin.params import Theta
from twin.treatment_scoring import outcome_arrays, paired, pooled_difference
from twin.truth import write_truth

THETA = Theta(30, 0.1, 5, 2, 2)


def fake_result(rid, died=False, death_step=None, tweak=None, steps=None):
    steps = np.arange(51, 5761, 51, dtype=np.int32) if steps is None else steps
    series = {n: (np.arange(len(steps), dtype=np.float32) + i) for i, n in enumerate(SERIES_NAMES)}
    if tweak:
        tweak(series)
    return Result(rid, died, death_step, 5760, 0.0, 0.0, True, steps, series)


@pytest.fixture
def truth_patient():
    write_truth("test-901", "test", THETA, seeds.patient_seed(901), fake_result("truth"))
    return "test-901"


def test_scoring_configs_use_true_theta_history_seed_reseed_and_shared_futures(truth_patient):
    win = [{"start_step": 612, "end_step": 711, "multipliers": {"TNF": 0.5}}]
    cfgs = evaluate.scoring_configs(truth_patient, {"none": [], "twin": win}, n_futures=5)
    assert len(cfgs) == 10
    none = [c for c in cfgs if "|none|" in c["id"]]
    twin = [c for c in cfgs if "|twin|" in c["id"]]
    assert [c["reseed_seed"] for c in none] == [c["reseed_seed"] for c in twin]  # same futures
    assert len({c["reseed_seed"] for c in none}) == 5
    assert all(seeds.role_of(c["reseed_seed"]) == "future" for c in cfgs)
    assert all(c["seed"] == seeds.patient_seed(901) and c["reseed_step"] == 612 for c in cfgs)
    assert all("interventions" not in c for c in none) and all("interventions" in c for c in twin)


def test_history_assertion_passes_for_shared_history_and_fails_on_a_difference(truth_patient):
    def after_td(series):  # differs only AFTER the decision time: allowed
        for v in series.values():
            v[13:] += 100

    ok = {f"{truth_patient}|none|0": fake_result("a", tweak=after_td)}
    assert evaluate.assert_history_matches_truth(truth_patient, ok, ["none"]) == 1

    def before_td(series):
        series["TNF"][3] += 1.0

    bad = {f"{truth_patient}|none|0": fake_result("a", tweak=before_td)}
    with pytest.raises(AssertionError, match="history of TNF"):
        evaluate.assert_history_matches_truth(truth_patient, bad, ["none"])
    wrong_steps = {
        f"{truth_patient}|none|0": fake_result("a", steps=np.arange(60, 5761, 60, dtype=np.int32))
    }
    with pytest.raises(AssertionError):
        evaluate.assert_history_matches_truth(truth_patient, wrong_steps, ["none"])


def test_paired_statistics_on_a_known_case():
    rng = np.random.default_rng(0)
    a = (rng.random(64) < 0.7).astype(float)  # dead under no treatment
    b = a.copy()
    b[:20] = 0.0  # the treatment saves some of those who died
    p = paired(a, b)
    assert p["diff"] == pytest.approx(a.mean() - b.mean()) and p["ci95"][0] > 0
    assert p["a_only"] == int(np.sum((a > 0) & (b == 0))) and p["b_only"] == 0
    null = paired(a, a.copy())
    assert null["diff"] == 0 and null["ci95"] == [0.0, 0.0]
    pooled = pooled_difference([(a, b), (a, a)])
    assert pooled["diff"] == pytest.approx((a.mean() - b.mean()) / 2) and pooled["n_patients"] == 2


def test_outcome_arrays_cap_survival_at_the_horizon():
    res = {
        f"p|s|{r}": fake_result("x", died=(r == 1), death_step=3000 if r == 1 else None)
        for r in range(3)
    }
    o = outcome_arrays(res, "p", "s", 3)
    assert o["died"].tolist() == [0, 1, 0] and o["steps_survived"].tolist() == [5760, 3000, 5760]
