import os

import pytest

from twin import treat
from twin.ga import GA


def test_schedule_maps_to_the_four_windows_of_99_steps_every_100():
    wins = treat.schedule_to_windows(treat.NO_TREATMENT)
    assert [(w["start_step"], w["end_step"]) for w in wins] == [
        (612, 711), (712, 811), (812, 911), (912, 1011)
    ]  # fmt: skip
    assert all(v == 1 for w in wins for v in w["multipliers"].values())
    sch = tuple(range(4)) * 4  # level index 0,1,2,3 for IL1ra, TNF, GCSF, IFNg in every window
    m = treat.schedule_to_windows(sch)[2]["multipliers"]
    assert m["IL1ra"] == 1
    assert (m["TNF"], m["GCSF"], m["IFNg"]) == (0.75, 5, 10)


def test_levels_and_search_space_size():
    assert treat.DIALS["IL1ra"] == [1, 2, 5, 10] and treat.DIALS["TNF"] == [1, 0.75, 0.5, 0.25]
    assert treat.N_GENES == 16 and len(treat.gene_specs()) == 16
    assert all(g.kind == "int" and (g.lo, g.hi) == (0, 3) for g in treat.gene_specs())
    assert "IL1ra x10" in treat.describe((3,) + (0,) * 15) and "no change" in treat.describe(
        treat.NO_TREATMENT
    )


def test_the_ga_starts_from_the_no_treatment_schedule():
    ga = GA(treat.gene_specs(), pop_size=6, seed=0)
    seen = []

    def ev(batch):
        seen.extend(batch)
        return [0.0] * len(batch)

    ga.run(ev, max_generations=1, patience=1, initial=[treat.NO_TREATMENT])
    assert treat.NO_TREATMENT in seen


def test_fitness_is_lexicographic_mortality_first():
    class R:
        def __init__(self, died, o, i):
            self.died, self.final_oxy_deficit, self.final_total_infection = died, o, i

    low_mort_high_oxy = treat.cost_of([R(False, 9000, 100)] * 31 + [R(False, 9000, 100)])
    high_mort_low_oxy = treat.cost_of([R(True, 8200, 0)] + [R(False, 0, 0)] * 31)
    f = lambda c: -(c[0] + c[1] / treat.SECONDARY_SCALE)  # noqa: E731
    assert f(low_mort_high_oxy) > f(high_mort_low_oxy)  # one death outweighs any oxy difference


@pytest.mark.skipif(not os.environ.get("IIRABM_BIN"), reason="no engine binary")
def test_search_end_to_end_tiny_and_deterministic():
    th = dict(inj_number=20, oxyHeal=0.15, infectSpread=4, numRecurInj=2, numInfectRepeat=2)
    ens = treat.build_ensemble([th], "unit-test", seeds_per_theta=2, workers=2)
    assert len(ens) == 2 and all(m.seed >= 750_000 for m in ens)
    cfg = treat.SearchConfig(pop_size=4, max_generations=2, patience=2, horizon_steps=700)
    a = treat.search(ens, cfg, workers=2)
    b = treat.search(ens, cfg, workers=2)
    assert a.best == b.best and a.best_cost == b.best_cost  # common random numbers
    assert a.n_sims == len(a.evaluated) * 2
    assert (
        a.best_cost[0] <= a.no_treatment_cost[0]
    )  # never worse than doing nothing on the ensemble
