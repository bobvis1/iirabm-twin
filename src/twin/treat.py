"""Treatment-schedule search on a twin (or on the prior): action space, ensembles and the GA.

A schedule is 16 genes: 4 windows x 4 dials, each gene the index (0-3) of a level. Window w acts in
the agent passes of steps 612 + 100 w + 1 .. 612 + 100 w + 99 (engine window {start, end}). The
search sees only simulations; it never reads a patient's truth (tests/test_truth_isolation.py).

Fitness (minimised, lexicographic): mortality by the fitness horizon across a fixed ensemble of
(theta, seed) runs that are alive at the decision time, then the mean of final oxy_deficit +
total_infection (V1's fitness) as a tie-breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from twin import seeds
from twin.calibrate import Posterior
from twin.config import DECISION_STEP
from twin.engine import Result, run_batch
from twin.ga import GA, GeneSpec
from twin.librarydata import LibraryData
from twin.params import PARAM_NAMES

DIALS: dict[str, list[float]] = {
    "IL1ra": [1, 2, 5, 10],  # resembles an IL-1 receptor antagonist (anakinra)
    "TNF": [1, 0.75, 0.5, 0.25],  # resembles anti-TNF antibodies
    "GCSF": [1, 2, 5, 10],  # resembles G-CSF (filgrastim)
    "IFNg": [1, 2, 5, 10],  # resembles interferon gamma-1b
}
DIAL_NAMES = list(DIALS)
N_WINDOWS = 4
WINDOW_SPACING = 100
WINDOW_LENGTH = 99
N_GENES = N_WINDOWS * len(DIALS)
NO_TREATMENT: tuple = (0,) * N_GENES
SECONDARY_SCALE = 1e6  # secondary term < 1/32 of one death: the fitness is lexicographic


def schedule_to_windows(schedule: tuple, t0: int = DECISION_STEP) -> list[dict]:
    """Engine intervention windows for a 16-gene schedule."""
    wins = []
    for w in range(N_WINDOWS):
        mult = {d: DIALS[d][int(schedule[w * len(DIALS) + k])] for k, d in enumerate(DIAL_NAMES)}
        start = t0 + WINDOW_SPACING * w
        wins.append({"start_step": start, "end_step": start + WINDOW_LENGTH, "multipliers": mult})
    return wins


def describe(schedule: tuple) -> str:
    """Readable schedule, one line per window; dials at level 1 are omitted."""
    lines = []
    for w, win in enumerate(schedule_to_windows(schedule)):
        active = {d: v for d, v in win["multipliers"].items() if v != 1}
        label = ", ".join(f"{d} x{v:g}" for d, v in active.items()) or "no change"
        lines.append(f"window {w + 1} (steps {win['start_step']}-{win['end_step']}): {label}")
    return "\n".join(lines)


def gene_specs() -> list[GeneSpec]:
    return [
        GeneSpec(f"w{w}_{d}", "int", 0, len(DIALS[d]) - 1, 1)
        for w in range(N_WINDOWS)
        for d in DIAL_NAMES
    ]


@dataclass(frozen=True)
class SearchConfig:
    pop_size: int = 24
    max_generations: int = 12
    patience: int = 4
    elite: int = 2
    horizon_steps: int = 3703  # H_f = 18 days
    n_theta: int = 8
    seeds_per_theta: int = 4
    seed: int = 0  # the GA's own generator


@dataclass
class Member:
    theta: dict
    seed: int


def draw_thetas(post: Posterior, lib: LibraryData, m: int, rng: np.random.Generator) -> list[dict]:
    """m parameter sets drawn from the posterior's design-point weights (with replacement)."""
    idx = rng.choice(lib.n, size=m, p=post.theta_w)
    return [dict(zip(PARAM_NAMES, lib.theta[i], strict=True)) for i in idx]


def _theta_cfg(theta: dict) -> dict:
    out = {}
    for k in PARAM_NAMES:
        v = theta[k]
        out[k] = float(v) if k == "oxyHeal" else int(round(v))
    return out


def build_ensemble(
    thetas: list[dict],
    label: str,
    seeds_per_theta: int = 4,
    workers: int | None = None,
    bin_path: str | None = None,
    max_tries: int = 400,
) -> list[Member]:
    """For every theta, the first `seeds_per_theta` search-range seeds whose run is alive at the
    decision time (a run dead before T_d is replaced by the next seed). A theta whose runs are so
    rarely alive that `max_tries` seeds do not yield enough is skipped (the patient is alive at
    T_d, so such a theta is not a plausible twin); if every theta is skipped it is an error."""
    members: list[Member] = []
    per = max(seeds_per_theta * 4, 16)
    for ti, th in enumerate(thetas):
        found: list[int] = []
        j = 0
        while len(found) < seeds_per_theta and j < max_tries:
            cands = [seeds.search_seed(f"{label}/theta{ti}", j + k) for k in range(per)]
            cfgs = [
                dict(_theta_cfg(th), id=str(c), seed=c, num_steps=DECISION_STEP,
                     output_stride=DECISION_STEP, outputs=["oxy_deficit"])
                for c in cands
            ]  # fmt: skip
            for c, r in zip(
                cands, run_batch(cfgs, n_workers=workers, bin_path=bin_path), strict=True
            ):
                if not r.died and len(found) < seeds_per_theta:
                    found.append(c)
            j += per
        if len(found) < seeds_per_theta:
            continue
        members += [Member(_theta_cfg(th), s) for s in found]
    if not members:
        raise RuntimeError(f"{label}: no theta had enough runs alive at the decision time")
    return members


def cost_of(results: list[Result]) -> tuple[float, float]:
    """(mortality, mean final oxy_deficit + total_infection) over the ensemble's runs."""
    mort = float(np.mean([r.died for r in results]))
    sec = float(np.mean([r.final_oxy_deficit + r.final_total_infection for r in results]))
    return mort, sec


@dataclass
class SearchResult:
    best: tuple
    best_cost: tuple[float, float]
    no_treatment_cost: tuple[float, float]
    history: list[float]  # best fitness (negative cost) per generation
    evaluated: list[tuple[tuple, float]] = field(default_factory=list)
    generations: int = 0
    n_sims: int = 0


def search(
    ensemble: list[Member],
    cfg: SearchConfig = SearchConfig(),  # noqa: B008 - frozen dataclass default
    workers: int | None = None,
    bin_path: str | None = None,
) -> SearchResult:
    """GA over schedules on a fixed ensemble (common random numbers: deterministic fitness)."""
    n_sims = 0
    costs: dict[tuple, tuple[float, float]] = {}

    def evaluate(batch: list[tuple]) -> list[float]:
        nonlocal n_sims
        cfgs = []
        for si, sch in enumerate(batch):
            wins = schedule_to_windows(sch)
            for ei, m in enumerate(ensemble):
                cfgs.append(
                    dict(
                        m.theta,
                        id=f"{si}/{ei}",
                        seed=m.seed,
                        num_steps=cfg.horizon_steps,
                        output_stride=cfg.horizon_steps,
                        outputs=["oxy_deficit"],
                        interventions=wins,
                    )
                )
        res = run_batch(cfgs, n_workers=workers, chunk_size=1, bin_path=bin_path)
        n_sims += len(cfgs)
        out = []
        for si, sch in enumerate(batch):
            c = cost_of(res[si * len(ensemble) : (si + 1) * len(ensemble)])
            costs[sch] = c
            out.append(-(c[0] + c[1] / SECONDARY_SCALE))
        return out

    ga = GA(gene_specs(), pop_size=cfg.pop_size, elite=cfg.elite, seed=cfg.seed)
    res = ga.run(evaluate, cfg.max_generations, cfg.patience, tol=1e-9, initial=[NO_TREATMENT])
    best, _ = res.best()
    best = tuple(int(g) for g in best)
    return SearchResult(
        best=best,
        best_cost=costs[best],
        no_treatment_cost=costs[NO_TREATMENT],
        history=res.best_by_generation,
        evaluated=[(tuple(int(g) for g in i), f) for i, f in res.evaluated],
        generations=res.generations,
        n_sims=n_sims,
    )
