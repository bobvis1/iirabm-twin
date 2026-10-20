"""SageMaker Processing entry point: GA calibration of patients, one after another.

    python -m twin.jobs.calibrate_ga [--input DIR] [--output DIR] [--workers N]

Each input file is one patient: {"calibration": {...}, "ga": {...}, "prior": {...},
"observations": {...}} (the settings travel with every file because the S3 prefix is sharded
across instances). Writes <patient>.npz (every evaluated theta with its log-likelihood, the
full-horizon runs of the best few) and <patient>.json (counts, timing) to the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from twin.calibrate import CalibrationConfig, GAConfig, GAFit, ga_calibrate
from twin.jobs.simulate import IN_DIR, OUT_DIR
from twin.librarydata import LibraryData
from twin.observe import Observations
from twin.params import PARAM_NAMES, Prior


def save_fit(fit: GAFit, path: Path) -> None:
    lib = fit.forecast_lib
    np.savez_compressed(
        path,
        evaluated_theta=fit.evaluated_theta, evaluated_ll=fit.evaluated_ll,
        best_by_generation=np.array(fit.best_by_generation),
        scalars=np.array([fit.generations, int(fit.stopped_early), fit.n_sims_search]),
        forecast_theta=fit.forecast_theta, forecast_ll=fit.forecast_ll,
        lib_theta=lib.theta, lib_panel=np.array(lib.panel), lib_panel_pre=lib.panel_pre,
        lib_targets=lib.targets, lib_alive=lib.alive_td, lib_died=lib.died,
        lib_death_step=lib.death_step,
    )  # fmt: skip


def load_fit(path: Path, patient_id: str) -> GAFit:
    with np.load(path) as z:
        lib = LibraryData(
            theta=z["lib_theta"], panel=tuple(str(x) for x in z["lib_panel"]),
            panel_pre=z["lib_panel_pre"], targets=z["lib_targets"], alive_td=z["lib_alive"],
            died=z["lib_died"], death_step=z["lib_death_step"],
        )  # fmt: skip
        gens, stopped, n_sims = (int(v) for v in z["scalars"])
        return GAFit(
            patient_id=patient_id, evaluated_theta=z["evaluated_theta"],
            evaluated_ll=z["evaluated_ll"],
            best_by_generation=[float(v) for v in z["best_by_generation"]], generations=gens,
            stopped_early=bool(stopped), n_sims_search=n_sims, forecast_theta=z["forecast_theta"],
            forecast_ll=z["forecast_ll"], forecast_lib=lib,
        )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_DIR)
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input.rglob("*.json"))
    if not files:
        raise SystemExit(f"no *.json patient files under {args.input}")
    for f in files:
        spec = json.loads(f.read_text())
        obs = Observations.from_json(spec["observations"])
        prior = Prior({n: tuple(spec["prior"][n]) for n in PARAM_NAMES})
        t0 = time.time()
        fit = ga_calibrate(
            obs, prior, CalibrationConfig(**spec["calibration"]), GAConfig(**spec["ga"]),
            workers=args.workers,
        )  # fmt: skip
        wall = time.time() - t0
        save_fit(fit, args.output / f"{obs.patient_id}.npz")
        info = {
            "patient_id": obs.patient_id, "wall_s": wall, "n_sims": fit.n_sims,
            "n_sims_search": fit.n_sims_search, "generations": fit.generations,
            "stopped_early": fit.stopped_early, "best_ll": float(fit.evaluated_ll.max()),
            "n_evaluated": int(len(fit.evaluated_ll)),
        }  # fmt: skip
        (args.output / f"{obs.patient_id}.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
