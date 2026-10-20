import numpy as np

from twin.ga import GA, GeneSpec

GENES = [
    GeneSpec("a", "int", 10, 50, 1),
    GeneSpec("b", "float", 0.03, 0.30, 0.01),
    GeneSpec("c", "int", 2, 12, 1),
]
TARGET = (31, 0.11, 7)


def fitness(ind):
    return (
        -(
            ((ind[0] - TARGET[0]) / 40) ** 2
            + ((ind[1] - TARGET[1]) / 0.27) ** 2
            + ((ind[2] - TARGET[2]) / 10) ** 2
        )
        * 100
    )


def run_ga(seed=0, **kw):
    ga = GA(GENES, pop_size=30, seed=seed)
    calls = []

    def ev(batch):
        calls.append(len(batch))
        return [fitness(i) for i in batch]

    return ga.run(ev, **kw), calls


def test_finds_the_optimum_of_a_toy_function():
    res, _ = run_ga(0, max_generations=40, patience=40)
    best, f = res.best()
    assert abs(best[0] - 31) <= 2 and abs(best[1] - 0.11) < 0.03 and abs(best[2] - 7) <= 1
    assert f > fitness((29, 0.14, 9))


def test_deterministic_given_the_seed():
    a, _ = run_ga(3, max_generations=10)
    b, _ = run_ga(3, max_generations=10)
    assert a.evaluated == b.evaluated
    c, _ = run_ga(4, max_generations=10)
    assert a.evaluated != c.evaluated


def test_individuals_respect_bounds_and_types_and_nothing_is_evaluated_twice():
    res, calls = run_ga(1, max_generations=15, patience=15)
    keys = [
        tuple(round(v, 9) if isinstance(v, float) else v for v in ind) for ind, _ in res.evaluated
    ]
    assert len(keys) == len(set(keys))  # cache works
    assert sum(calls) == len(keys)
    for ind, _ in res.evaluated:
        assert isinstance(ind[0], int) and 10 <= ind[0] <= 50
        assert isinstance(ind[1], float) and 0.03 <= ind[1] < 0.30
        assert isinstance(ind[2], int) and 2 <= ind[2] <= 12


def test_mutation_moves_exactly_one_gene_by_one_step():
    ga = GA(GENES, seed=0, reset_prob=0.0)
    ind = (30, 0.15, 6)
    for _ in range(200):
        m = ga.mutate(ind)
        changed = [i for i in range(3) if m[i] != ind[i]]
        assert len(changed) == 1
        j = changed[0]
        assert abs(m[j] - ind[j]) <= GENES[j].step + 1e-12


def test_crossover_takes_each_gene_from_a_parent():
    ga = GA(GENES, seed=0)
    a, b = (10, 0.05, 2), (50, 0.25, 12)
    kids = {ga.crossover(a, b) for _ in range(100)}
    for k in kids:
        assert all(k[i] in (a[i], b[i]) for i in range(3))
    assert len(kids) > 3


def test_tournament_prefers_the_fitter_individual():
    ga = GA(GENES, seed=0)
    pop = [(i, 0.1, 5) for i in range(10, 20)]
    fit = list(range(10))
    wins = np.mean([ga.select(pop, fit)[0] - 10 for _ in range(2000)])
    assert wins > 5.5  # tournament of 2 favours high fitness (random choice would give 4.5)


def test_early_stop_on_a_flat_fitness():
    ga = GA(GENES, pop_size=10, seed=0)
    res = ga.run(lambda batch: [0.0] * len(batch), max_generations=25, patience=3)
    assert res.stopped_early and res.generations == 4
