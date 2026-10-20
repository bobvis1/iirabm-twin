"""Check and summarise the simulation library (Parquet from the SageMaker job).

    uv run python scripts/analyze_library.py data/library/results/<job> [--n 4096 --k 16]

- completeness: every (theta_id, replicate) pair present exactly once, seeds as expected;
- thetas.parquet (design point -> theta), written next to the results;
- docs/results/library_summary.json and docs/figures/mortality_landscape.png (28-day mortality
  over pairs of parameters, binned; in the spirit of the 2017 landscape analysis, not a
  reproduction of it);
- the treatment fitness horizon H_f for Phase 6: the shortest whole number of days by which at
  least 90% of 28-day deaths, among runs alive at the decision time, have happened.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from twin import seeds  # noqa: E402
from twin.config import DECISION_STEP, DOCS_DIR, HORIZON_STEPS  # noqa: E402
from twin.library import design  # noqa: E402
from twin.params import INT_PARAMS, PARAM_NAMES, Prior  # noqa: E402

STEPS_PER_DAY = HORIZON_STEPS / 28  # 205.7; V1's own schedule uses 205


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument(
        "--job-minutes", type=float, help="wall minutes of the SageMaker job (for cost)"
    )
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--price-per-hour", type=float, default=1.80)
    ap.add_argument(
        "--summary-out", type=Path, default=DOCS_DIR / "results" / "library_summary.json"
    )
    ap.add_argument(
        "--figure-out", type=Path, default=DOCS_DIR / "figures" / "mortality_landscape.png"
    )
    args = ap.parse_args()

    runs = pa.concat_tables([pq.read_table(f) for f in sorted(args.results.glob("runs-*.parquet"))])
    ids = runs.column("id").to_pylist()
    theta_id = np.asarray(runs.column("theta_id"))
    rep = np.asarray(runs.column("replicate"))
    seed = np.asarray(runs.column("seed"))
    died = np.asarray(runs.column("died"))
    death_step = np.asarray(runs.column("death_step"))

    expected = {(t, r) for t in range(args.n) for r in range(args.k)}
    got = list(zip(theta_id.tolist(), rep.tolist(), strict=True))
    complete = len(got) == len(expected) and set(got) == expected and len(set(ids)) == len(ids)
    seeds_ok = all(
        s == seeds.library_seed(t, r)
        for t, r, s in zip(theta_id.tolist(), rep.tolist(), seed.tolist(), strict=True)
    )
    print(
        f"runs: {len(got)}; every (theta, seed) pair exactly once: {complete}; seeds as expected: {seeds_ok}"
    )

    thetas = design(Prior.load(), args.n)
    table = pa.table(
        {"theta_id": np.arange(args.n, dtype=np.int32)}
        | {n: np.array([getattr(t, n) for t in thetas]) for n in PARAM_NAMES}
    )
    pq.write_table(table, args.results / "thetas.parquet")
    th = np.array([[getattr(t, n) for n in PARAM_NAMES] for t in thetas])

    alive_td = (~died) | (death_step > DECISION_STEP)
    dead_by_28 = died & alive_td  # alive at T_d, dead by day 28
    ds = death_step[dead_by_28]
    days = np.arange(1, 29)
    cum = np.array([(ds <= d * STEPS_PER_DAY).mean() for d in days]) if len(ds) else np.zeros(28)
    h_f = int(days[np.argmax(cum >= 0.90)]) if len(ds) else None
    summary = {
        "n_design_points": args.n,
        "seeds_per_point": args.k,
        "runs": len(got),
        "complete": bool(complete),
        "seeds_as_expected": bool(seeds_ok),
        "alive_at_72h": float(alive_td.mean()),
        "mortality_28d_all_runs": float(died.mean()),
        "mortality_28d_given_alive_at_72h": float(dead_by_28.sum() / max(alive_td.sum(), 1)),
        "mean_steps_per_run": float(np.where(died, death_step, HORIZON_STEPS).mean()),
        "fitness_horizon_days": h_f,
        "deaths_after_72h_cumulative_fraction_by_day": {
            int(d): float(c) for d, c in zip(days, cum, strict=True)
        },
        "steps_per_day": STEPS_PER_DAY,
    }
    total_steps = float(np.asarray(runs.column("steps_run")).sum())
    summary["total_steps"] = total_steps
    if args.job_minutes:
        inst_h = args.instances * args.job_minutes / 60
        summary["job"] = {
            "wall_minutes": args.job_minutes,
            "instances": args.instances,
            "instance_hours": inst_h,
            "price_per_instance_hour_usd": args.price_per_hour,
            "estimated_cost_usd": inst_h * args.price_per_hour,
            "steps_per_second_per_instance": total_steps / (args.job_minutes * 60 * args.instances),
        }
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if "cumulative" not in k}, indent=1))

    # Mortality landscape: 28-day mortality of all runs, per parameter set, binned over pairs.
    per_theta = np.array([died[theta_id == t].mean() for t in range(args.n)])
    pairs = list(itertools.combinations(range(len(PARAM_NAMES)), 2))
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.6), gridspec_kw={"wspace": 0.5, "hspace": 0.4})

    def edges_for(col: int) -> np.ndarray:
        """One bin per integer value for small integer ranges, otherwise 8 equal-width bins."""
        lo, hi = th[:, col].min(), th[:, col].max()
        if PARAM_NAMES[col] in INT_PARAMS and hi - lo <= 12:
            return np.arange(lo - 0.5, hi + 1.0, 1.0)
        return np.linspace(lo, hi, 9)

    for ax, (i, j) in zip(axes.ravel(), pairs, strict=True):
        ei, ej = edges_for(i), edges_for(j)
        bi = np.clip(np.searchsorted(ei, th[:, i], side="right") - 1, 0, len(ei) - 2)
        bj = np.clip(np.searchsorted(ej, th[:, j], side="right") - 1, 0, len(ej) - 2)
        grid = np.full((len(ej) - 1, len(ei) - 1), np.nan)
        for a_ in range(len(ei) - 1):
            for b_ in range(len(ej) - 1):
                m = (bi == a_) & (bj == b_)
                if m.any():
                    grid[b_, a_] = per_theta[m].mean()
        im = ax.pcolormesh(ei, ej, grid, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel(PARAM_NAMES[i], fontsize=8)
        ax.set_ylabel(PARAM_NAMES[j], fontsize=8)
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="fraction of runs dead by day 28")
    fig.suptitle(
        f"28-day mortality over parameter pairs ({args.n} design points x {args.k} seeds, in this model)"
    )
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.summary_out} and {args.figure_out}")


if __name__ == "__main__":
    main()
