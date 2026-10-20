"""Write the treatment-search task files (and optionally upload them to S3).

    uv run python scripts/make_treatment_tasks.py ts-main --upload         # 5 twin searches + the one-size-fits-all
    uv run python scripts/make_treatment_tasks.py ts-oracle --oracle --upload   # optional upper bound (reads the truth)

Twin tasks use only the patient's observations; the oracle tasks need the true parameters and are
written by the evaluation side (twin.evaluate).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json

import numpy as np
import yaml

from twin import treat
from twin.calibrate import CalibrationConfig, library_posterior, prior_only
from twin.config import CONFIG_DIR, DATA_DIR
from twin.librarydata import load_or_build
from twin.observe import load_observations


def rng_for(label: str) -> np.random.Generator:
    h = hashlib.blake2b(f"treat/{label}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    chosen = json.loads((DATA_DIR / "treatment" / "patients.json").read_text())["patients"]
    cfg = treat.SearchConfig()
    search = dataclasses.asdict(cfg)
    tasks = []
    if args.oracle:
        from twin.evaluate import true_theta_task

        for pid in chosen:
            tasks.append({"label": f"oracle-{pid}", "kind": "oracle", "thetas": [true_theta_task(pid)],
                          "seeds_per_theta": cfg.n_theta * cfg.seeds_per_theta, "search": search})  # fmt: skip
    else:
        lib = load_or_build()
        cal = CalibrationConfig(
            **yaml.safe_load((CONFIG_DIR / "calibration.yaml").read_text())["library_posterior"]
        )
        for pid in chosen:
            post = library_posterior(load_observations(pid), lib, cal)
            thetas = treat.draw_thetas(post, lib, cfg.n_theta, rng_for(pid))
            tasks.append({"label": f"twin-{pid}", "kind": "twin", "thetas": thetas,
                          "seeds_per_theta": cfg.seeds_per_theta, "search": search})  # fmt: skip
        thetas = treat.draw_thetas(prior_only(lib), lib, cfg.n_theta, rng_for("one-size-fits-all"))
        tasks.append({"label": "one-size-fits-all", "kind": "prior", "thetas": thetas,
                      "seeds_per_theta": cfg.seeds_per_theta, "search": search})  # fmt: skip
    out = DATA_DIR / args.name / "configs"
    out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        (out / f"{t['label']}.json").write_text(json.dumps(t))
    print(f"{len(tasks)} task files in {out}")
    if args.upload:
        from twin.config import boto_session, load_infra

        infra, s3 = load_infra(), boto_session().client("s3")
        for t in tasks:
            s3.upload_file(
                str(out / f"{t['label']}.json"),
                infra.bucket,
                f"{args.name}/configs/{t['label']}.json",
            )
        print(f"uploaded to s3://{infra.bucket}/{args.name}/configs/")


if __name__ == "__main__":
    main()
