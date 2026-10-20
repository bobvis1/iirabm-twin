"""Write library or smoke-test config shards locally and optionally upload them to S3.

# smoke test: 100 runs (25 design points x 4 seeds)
uv run python scripts/make_library_configs.py smoke --n 25 --k 4 --shards 4
# the library
uv run python scripts/make_library_configs.py library --n 4096 --k 16 --shards 64 --upload
"""

from __future__ import annotations

import argparse

from twin.config import DATA_DIR
from twin.library import design, library_configs, write_shards
from twin.params import Prior


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="smoke, timing, library, ... (prefix in S3 and under data/)")
    ap.add_argument("--n", type=int, required=True, help="design points")
    ap.add_argument("--k", type=int, default=16, help="seeds per design point")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--start", type=int, default=0, help="first Sobol index (to extend a design)")
    ap.add_argument("--upload", action="store_true", help="upload to s3://<bucket>/<name>/configs/")
    args = ap.parse_args()

    prior = Prior.load()
    thetas = design(prior, args.n, start=args.start)
    configs = library_configs(thetas, args.start, args.k)
    out = DATA_DIR / args.name / "configs"
    paths = write_shards(configs, out, args.shards)
    print(f"{len(configs)} configs in {len(paths)} shards under {out}")
    if args.upload:
        from twin.config import boto_session, load_infra

        infra, s3 = load_infra(), boto_session().client("s3")
        for p in paths:
            s3.upload_file(str(p), infra.bucket, f"{args.name}/configs/{p.name}")
        print(f"uploaded to s3://{infra.bucket}/{args.name}/configs/")


if __name__ == "__main__":
    main()
