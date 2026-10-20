"""SageMaker Processing entry point: treatment-schedule searches, one task file each.

    python -m twin.jobs.treat_search [--input DIR] [--output DIR] [--workers N]

A task file: {"label", "kind", "thetas": [{...}, ...], "seeds_per_theta", "search": {...}}. The
ensemble (theta x alive-at-T_d search-range seeds) is built from it, the GA runs on that ensemble,
and <label>.json (best schedule, costs, per-generation history, every evaluated schedule, the
ensemble, counts and timing) is written to the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from twin import treat
from twin.jobs.simulate import IN_DIR, OUT_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_DIR)
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input.rglob("*.json"))
    if not files:
        raise SystemExit(f"no *.json task files under {args.input}")
    for f in files:
        task = json.loads(f.read_text())
        cfg = treat.SearchConfig(**task["search"])
        t0 = time.time()
        ens = treat.build_ensemble(
            task["thetas"], task["label"], task["seeds_per_theta"], workers=args.workers
        )
        t1 = time.time()
        res = treat.search(ens, cfg, workers=args.workers)
        out = {
            "label": task["label"],
            "kind": task["kind"],
            "search_config": asdict(cfg),
            "ensemble": [asdict(m) for m in ens],
            "n_theta_used": len({json.dumps(m.theta, sort_keys=True) for m in ens}),
            "n_theta_requested": len(task["thetas"]),
            "best": list(res.best),
            "best_cost": {
                "mortality": res.best_cost[0],
                "mean_oxy_plus_infection": res.best_cost[1],
            },
            "no_treatment_cost": {
                "mortality": res.no_treatment_cost[0],
                "mean_oxy_plus_infection": res.no_treatment_cost[1],
            },
            "history": res.history,
            "generations": res.generations,
            "n_evaluated": len(res.evaluated),
            "n_sims": res.n_sims,
            "evaluated": [{"genes": list(g), "fitness": fit} for g, fit in res.evaluated],
            "wall_s_ensemble": t1 - t0,
            "wall_s_search": time.time() - t1,
            "described": treat.describe(res.best),
        }
        (args.output / f"{task['label']}.json").write_text(json.dumps(out, indent=1))
        keys = ("label", "best_cost", "no_treatment_cost", "generations", "n_sims", "wall_s_search")
        print(json.dumps({k: out[k] for k in keys}), flush=True)


if __name__ == "__main__":
    main()
