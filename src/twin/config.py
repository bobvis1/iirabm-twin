"""Settings and paths (config/settings.yaml is git-ignored; some values are environment-set)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Overridable so tests never touch the real data/ and truth/ directories.
DATA_DIR = Path(os.environ.get("TWIN_DATA_DIR", ROOT / "data"))
TRUTH_DIR = Path(os.environ.get("TWIN_TRUTH_DIR", ROOT / "truth"))
CONFIG_DIR = ROOT / "config"
DOCS_DIR = ROOT / "docs"

# Fixed by the plan: decision time and observation grid, in engine step labels (51 steps = 6 h).
STEPS_PER_READING = 51
DECISION_STEP = 612  # 72 h, the 12th reading
HORIZON_STEPS = 5760  # 28 days


def load_settings() -> dict:
    """Read config/settings.yaml (or settings.example.yaml if it does not exist)."""
    for name in ("settings.yaml", "settings.example.yaml"):
        path = CONFIG_DIR / name
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    return {}


def engine_bin() -> str:
    """Command that starts the engine CLI (IIRABM_BIN), e.g. a path or scripts/iirabm_docker.sh."""
    value = os.environ.get("IIRABM_BIN")
    if not value:
        raise RuntimeError(
            "IIRABM_BIN is not set. Locally: export IIRABM_BIN=scripts/iirabm_docker.sh "
            "(see that script); on SageMaker it is the baked-in binary."
        )
    return value


@dataclass(frozen=True)
class Settings:
    aws_profile: str
    region: str
    owner: str
    name_prefix: str
    budget_usd: float


@dataclass(frozen=True)
class Infra:
    """Resource ids written by `scripts/infra.sh outputs` (config/outputs.json, git-ignored)."""

    bucket: str
    kms_key_arn: str
    role_arn: str
    subnet_ids: list[str]
    security_group_id: str
    ecr_repository_url: str


def load_aws_settings() -> Settings:
    raw = load_settings()
    return Settings(
        aws_profile=os.environ.get("TWIN_AWS_PROFILE", raw["aws_profile"]),
        region=os.environ.get("TWIN_REGION", raw["region"]),
        owner=raw["owner"],
        name_prefix=raw.get("name_prefix", "twin"),
        budget_usd=float(raw.get("budget_usd", 50)),
    )


def load_infra() -> Infra:
    raw = json.loads((CONFIG_DIR / "outputs.json").read_text())
    v = {k: x["value"] for k, x in raw.items()}
    return Infra(
        bucket=v["bucket"],
        kms_key_arn=v["kms_key_arn"],
        role_arn=v["role_arn"],
        subnet_ids=v["subnet_ids"],
        security_group_id=v["security_group_id"],
        ecr_repository_url=v["ecr_repository_url"],
    )


def boto_session():
    import boto3

    s = load_aws_settings()
    return boto3.Session(profile_name=s.aws_profile, region_name=s.region)
