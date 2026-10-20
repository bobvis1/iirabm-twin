"""Score treatment schedules on the TRUE patients (fresh futures), per docs/treatment_protocol.md.

# 1. after the search job: build the scoring configs (reads the truth through twin.evaluate)
uv run python scripts/score_treatment.py prepare data/ts-main/results/<job> [--oracle data/ts-oracle/results/<job>] --upload
# 2. run them:  scripts/run_job.py ts-score --module twin.jobs.simulate ...
# 3. score:
uv run python scripts/score_treatment.py score data/ts-score/results/<job> data/ts-main/results/<job> [--oracle ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twin import evaluate, treat
from twin.config import DATA_DIR, DOCS_DIR
from twin.library import write_shards
from twin.results_io import load_results
from twin.treatment_scoring import outcome_arrays, paired, pooled_difference

N_FUTURES = 64


def load_schedules(search_dir: Path, oracle_dir: Path | None) -> tuple[list[str], dict[str, dict]]:
    chosen = json.loads((DATA_DIR / "treatment" / "patients.json").read_text())["patients"]
    found = {p.stem: json.loads(p.read_text()) for p in search_dir.glob("*.json")}
    if oracle_dir:
        found |= {p.stem: json.loads(p.read_text()) for p in oracle_dir.glob("*.json")}
    return chosen, found


def schedules_for(pid: str, found: dict) -> dict[str, list[dict]]:
    out = {
        "none": [],
        "one_size": treat.schedule_to_windows(tuple(found["one-size-fits-all"]["best"])),
        "twin": treat.schedule_to_windows(tuple(found[f"twin-{pid}"]["best"])),
    }
    if f"oracle-{pid}" in found:
        out["oracle"] = treat.schedule_to_windows(tuple(found[f"oracle-{pid}"]["best"]))
    return out


def prepare(search_dir: Path, oracle_dir: Path | None, upload: bool) -> None:
    chosen, found = load_schedules(search_dir, oracle_dir)
    cfgs = []
    for pid in chosen:
        cfgs += evaluate.scoring_configs(pid, schedules_for(pid, found), N_FUTURES)
    out = DATA_DIR / "ts-score" / "configs"
    paths = write_shards(cfgs, out, 8)
    print(f"{len(cfgs)} scoring runs in {len(paths)} shards under {out}")
    if upload:
        from twin.config import boto_session, load_infra

        infra, s3 = load_infra(), boto_session().client("s3")
        for p in paths:
            s3.upload_file(str(p), infra.bucket, f"ts-score/configs/{p.name}")
        print("uploaded")


def score(results_dir: Path, search_dir: Path, oracle_dir: Path | None) -> None:
    chosen, found = load_schedules(search_dir, oracle_dir)
    results = load_results(results_dir)
    out: dict = {"patients": {}, "futures": N_FUTURES, "schedules": {}}
    arrays: dict[str, dict] = {}
    for pid in chosen:
        names = list(schedules_for(pid, found))
        checked = evaluate.assert_history_matches_truth(pid, results, names)
        per = {n: outcome_arrays(results, pid, n, N_FUTURES) for n in names}
        arrays[pid] = per
        rec = {
            "history_runs_checked": checked,
            "mortality": {n: float(per[n]["died"].mean()) for n in names},
            "mean_days_survived": {n: float(per[n]["steps_survived"].mean() / (5760 / 28)) for n in names},
            "twin_schedule": treat.describe(tuple(found[f"twin-{pid}"]["best"])),
            "twin_search": {k: found[f"twin-{pid}"][k] for k in ("best_cost", "no_treatment_cost", "generations", "n_sims")},
            "diffs": {},
        }  # fmt: skip
        for a, b in (
            ("twin", "none"),
            ("one_size", "none"),
            ("twin", "one_size"),
            ("oracle", "none"),
            ("oracle", "twin"),
        ):
            if a in per and b in per:
                rec["diffs"][f"{a}_minus_{b}"] = {
                    "mortality": paired(per[a]["died"], per[b]["died"]),
                    "steps_survived": paired(per[a]["steps_survived"], per[b]["steps_survived"]),
                }
        out["patients"][pid] = rec
    for a, b in (("twin", "none"), ("one_size", "none"), ("twin", "one_size")):
        out["schedules"][f"{a}_minus_{b}"] = pooled_difference(
            [(arrays[p][a]["died"], arrays[p][b]["died"]) for p in chosen]
        )
    if all("oracle" in arrays[p] for p in chosen):
        out["schedules"]["oracle_minus_none"] = pooled_difference(
            [(arrays[p]["oracle"]["died"], arrays[p]["none"]["died"]) for p in chosen]
        )
    out["one_size_fits_all"] = {
        "schedule": treat.describe(tuple(found["one-size-fits-all"]["best"])),
        "best_cost": found["one-size-fits-all"]["best_cost"],
        "no_treatment_cost": found["one-size-fits-all"]["no_treatment_cost"],
    }
    (DOCS_DIR / "results" / "treatment.json").write_text(json.dumps(out, indent=1))
    for pid, r in out["patients"].items():
        print(pid, {k: round(v, 2) for k, v in r["mortality"].items()})
    print(
        {
            k: (round(v["diff"], 3), [round(x, 3) for x in v["ci95"]])
            for k, v in out["schedules"].items()
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("search_dir", type=Path)
    p.add_argument("--oracle", type=Path)
    p.add_argument("--upload", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("results_dir", type=Path)
    s.add_argument("search_dir", type=Path)
    s.add_argument("--oracle", type=Path)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.search_dir, a.oracle, a.upload)
    else:
        score(a.results_dir, a.search_dir, a.oracle)


if __name__ == "__main__":
    main()
