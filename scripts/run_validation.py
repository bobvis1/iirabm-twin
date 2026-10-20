"""Phase 5: score the twin against the hidden truth.

Final run (test cohort, settings frozen at a commit, scored once):
    uv run python scripts/run_validation.py --ga-dir data/ga-test/results/<job>
Dry run on the development cohort (to debug; never reported as the validation):
    uv run python scripts/run_validation.py --cohort development --allow-development \
        --ga-dir data/ga-dev/results/<job> --out build/validation_dev.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from twin import validation as V
from twin.calibrate import CalibrationConfig
from twin.config import CONFIG_DIR, DOCS_DIR, ROOT
from twin.evaluate import _check_cohort, assert_frozen
from twin.forecast import ForecastEngine
from twin.librarydata import load_or_build
from twin.observe import load_observations
from twin.truth import list_patients


def measurement_design(ids, lib, engine, cal_theta, baseline, lod, log) -> dict:
    rows = {}
    full_n = len(lib.panel) * 12
    for design in V.design_variants(lib.panel):
        log(f"design: {design['name']}")
        sc = V.score_design(ids, lib, engine, cal_theta, design, lod)
        # Second view: tau scaled so the total weight on the readings equals the baseline's
        # (tau was tuned for 96 readings; fewer readings otherwise get relatively more weight).
        n_read = len(design["readings"]) * len(design["panel"])
        cal_scaled = dataclasses.replace(cal_theta, tau=cal_theta.tau * full_n / n_read)
        sc2 = V.score_design(ids, lib, engine, cal_scaled, design, lod)
        rows[design["name"]] = {
            "n_readings": n_read,
            "fixed_tau": V.design_row(sc, baseline),
            "scaled_tau": V.design_row(sc2, baseline),
            "tau_scaled_value": cal_scaled.tau,
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["test", "development"], default="test")
    ap.add_argument("--allow-development", action="store_true")
    ap.add_argument("--ga-dir", type=Path)
    ap.add_argument("--out", type=Path, default=DOCS_DIR / "results" / "validation.json")
    ap.add_argument("--rescore", action="store_true", help="overwrite an existing final result")
    ap.add_argument("--skip-designs", action="store_true")
    ap.add_argument(
        "--designs-only",
        action="store_true",
        help="recompute only the measurement-design study and merge it into the existing --out file",
    )
    args = ap.parse_args()

    ids = list_patients(args.cohort)
    _check_cohort(ids, args.allow_development)
    frozen = None
    if args.cohort == "test":
        frozen = assert_frozen()
        if args.out.exists() and not args.rescore and not args.designs_only:
            sys.exit(
                f"{args.out} exists: the test cohort is scored once (use --rescore to overwrite)"
            )

    raw = yaml.safe_load((CONFIG_DIR / "calibration.yaml").read_text())
    cal_theta = CalibrationConfig(**raw["library_posterior"])
    cal_run = dataclasses.replace(cal_theta, forecast_level="run")
    ga_cfg = raw["ga"]
    lod = yaml.safe_load((CONFIG_DIR / "measurement.yaml").read_text())["lod"]
    lib = load_or_build()
    engine = ForecastEngine(lib)
    log = lambda m: print(m, flush=True)  # noqa: E731

    log(f"{len(ids)} {args.cohort} patients; frozen at {frozen}")
    if args.designs_only:
        out = json.loads(args.out.read_text())
        assert (
            out["settings_sha256"]
            == hashlib.sha256((CONFIG_DIR / "calibration.yaml").read_bytes()).hexdigest()
        )
        assert out["cohort"] == args.cohort and out["patients"] == ids
        baseline = V.score_library_method(ids, lib, engine, cal_theta, load_observations)
        out["measurement_design"] = measurement_design(
            ids, lib, engine, cal_theta, baseline, lod, log
        )
        out["measurement_design_recomputed"] = (
            "recomputed after a bug dropped it from the first save; same frozen settings, deterministic"
        )
        args.out.write_text(json.dumps(out, indent=1, default=str))
        log(f"merged measurement design into {args.out}")
        return
    stored = load_observations
    methods: dict[str, list[dict]] = {}
    methods["prior_only"] = V.score_prior_only(ids, lib, engine)
    methods["library_theta"] = V.score_library_method(ids, lib, engine, cal_theta, stored)
    methods["library_run"] = V.score_library_method(ids, lib, engine, cal_run, stored)
    ga_ids = []
    if args.ga_dir:
        ga_ids = [i for i in ids if (args.ga_dir / f"{i}.npz").exists()]
        methods["ga"] = V.score_ga_method(
            ga_ids, args.ga_dir, cal_theta, ga_cfg["delta"], ga_cfg["kappa"], lib
        )
    log("methods scored")

    summaries = {name: V.summarise_method(sc) for name, sc in methods.items()}
    paired = {
        n: V.paired_vs_baseline(sc, methods["prior_only"])
        for n, sc in methods.items()
        if n not in ("prior_only", "ga")
    }
    # The GA covers a subset: compare every method on exactly those patients.
    subset = {}
    if ga_ids:
        pos = [ids.index(i) for i in ga_ids]
        for name in ("prior_only", "library_theta", "library_run"):
            sub = [methods[name][k] for k in pos]
            subset[name] = V.summarise_method(sub)
        subset["ga"] = summaries["ga"]
        subset["paired_ga_vs_library"] = V.paired_vs_baseline(
            methods["ga"], [methods["library_theta"][k] for k in pos]
        )
        subset["paired_ga_vs_prior"] = V.paired_vs_baseline(
            methods["ga"], [methods["prior_only"][k] for k in pos]
        )
        subset["patients"] = ga_ids

    out = {
        "cohort": args.cohort, "patients": ids, "frozen_at_commit": frozen,
        "settings": {"library_posterior": raw["library_posterior"], "ga": ga_cfg},
        "settings_sha256": hashlib.sha256((CONFIG_DIR / "calibration.yaml").read_bytes()).hexdigest(),
        "date": datetime.date.today().isoformat(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "methods": summaries, "paired_vs_prior_only": paired, "on_ga_subset": subset,
        "truth_vs_posterior_mean": V.truth_vs_mean(ids, summaries["library_theta"]),
    }  # fmt: skip

    if not args.skip_designs:
        baseline = methods["library_theta"]
        rows = measurement_design(ids, lib, engine, cal_theta, baseline, lod, log)
        out["measurement_design"] = rows
        log("misspecification: data CV 40%, twin assumes 25%")
        mis = dict(
            V.design_variants(lib.panel)[0],
            name="misspecified: data CV 40%, twin assumes 25%",
            data_cv=0.40,
            assumed_cv=0.25,
        )
        sc = V.score_design(ids, lib, engine, cal_theta, mis, lod)
        out["misspecification"] = {"well_specified": V.design_row(baseline, baseline), "misspecified": V.design_row(sc, baseline),
                                   "misspecified_summary": V.summarise_method(sc)}  # fmt: skip

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    b = summaries["library_theta"]["mortality"]
    log(
        f"wrote {args.out}; library theta-level Brier {b['brier'][0]:.3f}, prior-only {summaries['prior_only']['mortality']['brier'][0]:.3f}"
    )


if __name__ == "__main__":
    main()
