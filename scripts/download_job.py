"""Wait for a SageMaker Processing job and download its results.

uv run python scripts/download_job.py <job name> <prefix under the bucket, e.g. ts-oracle/results/<job>> <dest>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from twin.config import boto_session, load_infra


def main(job: str, prefix: str, dest: Path) -> None:
    session, infra = boto_session(), load_infra()
    sm, s3 = session.client("sagemaker"), session.client("s3")
    while True:
        d = sm.describe_processing_job(ProcessingJobName=job)
        status = d["ProcessingJobStatus"]
        if status != "InProgress":
            break
        time.sleep(60)
    print(status, d.get("FailureReason", ""))
    if status != "Completed":
        raise SystemExit(1)
    start, end = d["ProcessingStartTime"], d["ProcessingEndTime"]
    mins = (end - start).total_seconds() / 60
    n = d["ProcessingResources"]["ClusterConfig"]["InstanceCount"]
    print(
        f"{mins:.1f} min x {n} instances = about ${mins / 60 * n * 1.80:.2f} at $1.80 per instance-hour"
    )
    dest.mkdir(parents=True, exist_ok=True)
    prefix = prefix.rstrip("/") + "/"
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=infra.bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            target = dest / Path(obj["Key"]).relative_to(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(infra.bucket, obj["Key"], str(target))
    print("downloaded to", dest)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
