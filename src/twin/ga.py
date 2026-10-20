"""A small genetic algorithm with mixed integer and continuous genes.

Design follows the genetic algorithm the An-Cockrell lab used on this model (Cockrell and An,
PLoS Computational Biology, 2018): tournament selection of size 2, uniform crossover, and a
mutation that moves one gene a step up or down. Additions, for stability with small populations:
elitism (the best few survive unchanged), a small chance of resetting one gene to a random value,
a cache so no individual is evaluated twice, and early stopping when the best fitness stops
improving. Fitness is maximised and is computed by a caller-supplied batch function, so the
expensive evaluations (fresh simulations) can run in parallel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GeneSpec:
    name: str
    kind: str  # "int" (inclusive bounds) or "float" (lo <= x < hi)
    lo: float
    hi: float
    step: float = 1.0  # size of a mutation step

    def random(self, rng: np.random.Generator):
        if self.kind == "int":
            return int(rng.integers(int(self.lo), int(self.hi) + 1))
        return float(rng.uniform(self.lo, self.hi))

    def clip(self, v):
        if self.kind == "int":
            return int(min(max(round(v), int(self.lo)), int(self.hi)))
        eps = (self.hi - self.lo) * 1e-9
        return float(min(max(v, self.lo), self.hi - eps))


Individual = tuple


@dataclass
class GAResult:
    evaluated: list[tuple[Individual, float]] = field(default_factory=list)  # unique, in order
    best_by_generation: list[float] = field(default_factory=list)
    generations: int = 0
    stopped_early: bool = False

    @property
    def n_evaluations(self) -> int:
        return len(self.evaluated)

    def best(self) -> tuple[Individual, float]:
        return max(self.evaluated, key=lambda t: t[1])


def _key(ind: Individual) -> tuple:
    return tuple(round(v, 9) if isinstance(v, float) else v for v in ind)


class GA:
    def __init__(
        self,
        genes: Sequence[GeneSpec],
        pop_size: int = 40,
        tournament: int = 2,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.6,
        reset_prob: float = 0.05,
        elite: int = 2,
        seed: int = 0,
    ):
        self.genes = list(genes)
        self.pop_size = pop_size
        self.tournament = tournament
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.reset_prob = reset_prob
        self.elite = elite
        self.rng = np.random.default_rng(seed)

    # --- operators (public so they can be tested) ---
    def random_individual(self) -> Individual:
        return tuple(g.random(self.rng) for g in self.genes)

    def select(self, pop: list[Individual], fit: list[float]) -> Individual:
        idx = self.rng.integers(0, len(pop), size=self.tournament)
        return pop[max(idx, key=lambda i: fit[i])]

    def crossover(self, a: Individual, b: Individual) -> Individual:
        take_a = self.rng.random(len(self.genes)) < 0.5
        return tuple(x if t else y for x, y, t in zip(a, b, take_a, strict=True))

    def mutate(self, ind: Individual) -> Individual:
        out = list(ind)
        j = int(self.rng.integers(len(self.genes)))
        g = self.genes[j]
        if self.rng.random() < self.reset_prob:
            out[j] = g.random(self.rng)
        else:
            out[j] = g.clip(out[j] + (g.step if self.rng.random() < 0.5 else -g.step))
        return tuple(out)

    def _next_generation(self, pop: list[Individual], fit: list[float]) -> list[Individual]:
        order = np.argsort(fit)[::-1]
        nxt = [pop[i] for i in order[: self.elite]]
        while len(nxt) < self.pop_size:
            a, b = self.select(pop, fit), self.select(pop, fit)
            child = self.crossover(a, b) if self.rng.random() < self.crossover_prob else a
            if self.rng.random() < self.mutation_prob or child == a:
                child = self.mutate(child)
            nxt.append(child)
        return nxt

    def run(
        self,
        evaluate: Callable[[list[Individual]], list[float]],
        max_generations: int = 25,
        patience: int = 5,
        tol: float = 0.5,
    ) -> GAResult:
        """Run the GA. `evaluate` maps a list of new individuals to their fitness (maximised)."""
        res = GAResult()
        cache: dict[tuple, float] = {}
        pop = [self.random_individual() for _ in range(self.pop_size)]
        best_so_far, stale = -np.inf, 0
        for gen in range(max_generations):
            new = []
            for ind in pop:
                if _key(ind) not in cache and ind not in new:
                    new.append(ind)
            if new:
                for ind, f in zip(new, evaluate(new), strict=True):
                    cache[_key(ind)] = float(f)
                    res.evaluated.append((ind, float(f)))
            fit = [cache[_key(ind)] for ind in pop]
            res.generations = gen + 1
            gen_best = max(fit)
            res.best_by_generation.append(gen_best)
            if gen_best > best_so_far + tol:
                best_so_far, stale = gen_best, 0
            else:
                best_so_far = max(best_so_far, gen_best)
                stale += 1
            if stale >= patience:
                res.stopped_early = gen + 1 < max_generations
                break
            pop = self._next_generation(pop, fit)
        return res
