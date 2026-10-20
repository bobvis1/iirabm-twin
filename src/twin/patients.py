"""Virtual patients: sample a hidden theta and history seed, simulate the full horizon, and keep
only patients alive at the decision time (like ICU patients at enrollment).

Two cohorts are built from one deterministic candidate stream: the first `n_dev` accepted
candidates are the development cohort (used to tune anything that is tuned), the next `n_test`
the test cohort (scored once). The builder writes the hidden truth (twin.truth) and the
observations the twin is allowed to see (twin.observe).

Run:  python -m twin.patients [--dev 40] [--test 100]
"""

from __future__ import annotations

import argparse

import numpy as np

from twin import seeds
from twin.config import DECISION_STEP, HORIZON_STEPS, STEPS_PER_READING
from twin.engine import run_batch
from twin.observe import MeasurementModel, make_observations, save_observations
from twin.params import Prior
from twin.truth import write_truth

STREAM_KEY = 20261020  # fixed: the candidate stream is part of the experiment definition


def candidate(c: int, prior: Prior) -> tuple:
    """Candidate c: a theta from the continuous prior and a history seed. Deterministic in c."""
    rng = np.random.default_rng([STREAM_KEY, c])
    return prior.sample(rng, 1)[0], seeds.patient_seed(c)


def alive_at_decision(result) -> bool:
    return (not result.died) or (
        result.death_step is not None and result.death_step > DECISION_STEP
    )


def build_cohorts(
    n_dev: int = 40,
    n_test: int = 100,
    prior: Prior | None = None,
    model: MeasurementModel | None = None,
    workers: int = 8,
    batch: int = 40,
) -> dict[str, list[str]]:
    prior = prior or Prior.load()
    model = model or MeasurementModel.load()
    need = {"development": n_dev, "test": n_test}
    ids: dict[str, list[str]] = {"development": [], "test": []}
    prefix = {"development": "dev", "test": "test"}
    c = 0
    tried = accepted = 0
    while len(ids["development"]) < n_dev or len(ids["test"]) < n_test:
        cands = [candidate(c + i, prior) for i in range(batch)]
        configs = [
            dict(
                th.to_engine(),
                id=f"cand-{c + i}",
                seed=sd,
                num_steps=HORIZON_STEPS,
                output_stride=STEPS_PER_READING,
            )
            for i, (th, sd) in enumerate(cands)
        ]
        results = run_batch(configs, n_workers=workers, chunk_size=1)
        for (th, sd), res in zip(cands, results, strict=True):
            tried += 1
            if not alive_at_decision(res):
                continue
            cohort = "development" if len(ids["development"]) < n_dev else "test"
            if len(ids[cohort]) >= need[cohort]:
                continue
            accepted += 1
            pid = f"{prefix[cohort]}-{len(ids[cohort]) + 1:03d}"
            ids[cohort].append(pid)
            write_truth(pid, cohort, th, sd, res)
            save_observations(make_observations(pid, res.series, res.step, model))
        c += batch
    print(
        f"{accepted} patients accepted from {tried} candidates "
        f"({accepted / tried:.0%} alive at T_d)"
    )
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=40)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    build_cohorts(args.dev, args.test, workers=args.workers)


if __name__ == "__main__":
    main()
