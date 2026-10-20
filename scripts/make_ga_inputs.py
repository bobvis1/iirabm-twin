"""Write per-patient GA input files (observations + settings) for the DEVELOPMENT cohort subset
and optionally upload them to S3 for a SageMaker job. Test patients are refused.

    uv run python scripts/make_ga_inputs.py ga-dev --n 20 [--upload]
"""

from __future__ import annotations

import argparse
import json

import yaml

from twin.config import CONFIG_DIR, DATA_DIR
from twin.evaluate import assert_development_only
from twin.observe import load_observations
from twin.params import PARAM_NAMES, Prior


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n", type=int, default=20, help="first n development patients by id")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--cohort", choices=["development", "test"], default="development")
    ap.add_argument("--pop-size", type=int)
    ap.add_argument("--max-generations", type=int)
    args = ap.parse_args()
    prefix = "dev" if args.cohort == "development" else "test"
    ids = [f"{prefix}-{i:03d}" for i in range(1, args.n + 1)]
    if args.cohort == "development":
        assert_development_only(ids)
    else:
        from twin.evaluate import assert_frozen

        print("test cohort: settings frozen at", assert_frozen())
    cfg = yaml.safe_load((CONFIG_DIR / "calibration.yaml").read_text())
    cal = {k: v for k, v in cfg["library_posterior"].items()}
    ga = {k: v for k, v in cfg["ga"].items() if k not in ("delta", "kappa")}
    if args.pop_size:
        ga["pop_size"] = args.pop_size
    if args.max_generations:
        ga["max_generations"] = args.max_generations
    prior = Prior.load()
    out = DATA_DIR / args.name / "configs"
    out.mkdir(parents=True, exist_ok=True)
    for pid in ids:
        spec = {
            "calibration": cal,
            "ga": ga,
            "prior": {n: list(prior.bounds[n]) for n in PARAM_NAMES},
            "observations": load_observations(pid).to_json(),
        }
        (out / f"{pid}.json").write_text(json.dumps(spec))
    print(f"{len(ids)} patient files in {out}")
    if args.upload:
        from twin.config import boto_session, load_infra

        infra, s3 = load_infra(), boto_session().client("s3")
        for pid in ids:
            s3.upload_file(
                str(out / f"{pid}.json"), infra.bucket, f"{args.name}/configs/{pid}.json"
            )
        print(f"uploaded to s3://{infra.bucket}/{args.name}/configs/")


if __name__ == "__main__":
    main()
