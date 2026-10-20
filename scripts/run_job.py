"""Launch a SageMaker Processing job that runs engine configs (twin.jobs.simulate).

Custom image, private VPC (no NAT), network isolation, KMS-encrypted volume and output. Input is a
prefix of JSON Lines shards in S3, sharded across instances with ShardedByS3Key.

    uv run python scripts/run_job.py smoke --instance-count 1 --price-per-hour 1.84 --download
    uv run python scripts/run_job.py library --instance-count 4 --price-per-hour 1.84 --approved

State the cost before launching: the script prints an upper bound (instances x max runtime x price)
and refuses to start a job whose upper bound is above --max-cost-usd (default 10) unless
--approved is given (CLAUDE.md: ask the user before any single job estimated above $10).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sagemaker.core.resources import ProcessingJob
from sagemaker.core.shapes import (
    AppSpecification,
    NetworkConfig,
    ProcessingClusterConfig,
    ProcessingInput,
    ProcessingOutput,
    ProcessingOutputConfig,
    ProcessingResources,
    ProcessingS3Input,
    ProcessingS3Output,
    ProcessingStoppingCondition,
    Tag,
    VpcConfig,
)

from twin.config import DATA_DIR, boto_session, load_aws_settings, load_infra

IN, OUT = "/opt/ml/processing/input", "/opt/ml/processing/output"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "name", help="prefix under the bucket holding <name>/configs/ (smoke, library, ...)"
    )
    ap.add_argument("--module", default="twin.jobs.simulate", help="entry point module")
    ap.add_argument("--image-uri", help="default: <ecr repo>:latest (prefer the git-sha tag)")
    ap.add_argument("--instance-type", default="ml.c5.9xlarge")
    ap.add_argument("--instance-count", type=int, default=1)
    ap.add_argument("--max-runtime-s", type=int, default=3600)
    ap.add_argument(
        "--price-per-hour", type=float, required=True, help="USD per instance-hour (look it up)"
    )
    ap.add_argument("--max-cost-usd", type=float, default=10.0)
    ap.add_argument(
        "--approved", action="store_true", help="the user approved a job above --max-cost-usd"
    )
    ap.add_argument(
        "--download", action="store_true", help="download results to data/<name>/results/"
    )
    args = ap.parse_args()

    upper = args.instance_count * args.max_runtime_s / 3600 * args.price_per_hour
    print(f"cost upper bound: {args.instance_count} x {args.instance_type} x "
          f"{args.max_runtime_s / 3600:.2f} h x ${args.price_per_hour:.2f}/h = ${upper:.2f}")  # fmt: skip
    if upper > args.max_cost_usd and not args.approved:
        raise SystemExit(f"upper bound ${upper:.2f} is above ${args.max_cost_usd:.2f}: get the user's "
                         "approval, then pass --approved")  # fmt: skip

    s, infra, session = load_aws_settings(), load_infra(), boto_session()
    job_name = f"twin-{args.name}-{int(time.time())}"
    out_uri = f"s3://{infra.bucket}/{args.name}/results/{job_name}"
    job = ProcessingJob.create(
        processing_job_name=job_name,
        role_arn=infra.role_arn,
        app_specification=AppSpecification(
            image_uri=args.image_uri or f"{infra.ecr_repository_url}:latest",
            container_entrypoint=["python", "-m", args.module],
        ),
        processing_inputs=[
            ProcessingInput(
                input_name="configs",
                s3_input=ProcessingS3Input(
                    s3_uri=f"s3://{infra.bucket}/{args.name}/configs/",
                    local_path=IN,
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="ShardedByS3Key",
                ),
            )
        ],
        processing_resources=ProcessingResources(
            cluster_config=ProcessingClusterConfig(
                instance_count=args.instance_count,
                instance_type=args.instance_type,
                volume_size_in_gb=30,
                volume_kms_key_id=infra.kms_key_arn,
            )
        ),
        processing_output_config=ProcessingOutputConfig(
            kms_key_id=infra.kms_key_arn,
            outputs=[
                ProcessingOutput(
                    output_name="results",
                    s3_output=ProcessingS3Output(
                        s3_uri=out_uri, local_path=OUT, s3_upload_mode="EndOfJob"
                    ),
                )
            ],
        ),
        network_config=NetworkConfig(
            enable_network_isolation=True,
            vpc_config=VpcConfig(
                security_group_ids=[infra.security_group_id], subnets=infra.subnet_ids
            ),
        ),
        stopping_condition=ProcessingStoppingCondition(max_runtime_in_seconds=args.max_runtime_s),
        tags=[
            Tag(key="project", value="iirabm-twin"),
            Tag(key="owner", value=s.owner),
            Tag(key="env", value="demo"),
        ],
        session=session,
    )
    print(f"started {job_name}; waiting...", flush=True)
    t0 = time.time()
    job.wait()
    job.refresh()
    elapsed = time.time() - t0
    status = job.processing_job_status
    print(f"{status} after {elapsed / 60:.1f} min; "
          f"billed about {args.instance_count} x {elapsed / 3600:.2f} h = "
          f"${args.instance_count * elapsed / 3600 * args.price_per_hour:.2f} (plus startup)")  # fmt: skip
    if status != "Completed":
        raise SystemExit(f"job {status}: {getattr(job, 'failure_reason', '')}")
    print(f"results: {out_uri}/")
    if args.download:
        dest = DATA_DIR / args.name / "results" / job_name
        dest.mkdir(parents=True, exist_ok=True)
        s3 = session.client("s3")
        prefix = f"{args.name}/results/{job_name}/"
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=infra.bucket, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                target = dest / Path(obj["Key"]).relative_to(prefix)
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(infra.bucket, obj["Key"], str(target))
        print(f"downloaded to {dest}")


if __name__ == "__main__":
    main()
