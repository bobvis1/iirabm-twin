"""Settings and paths (config/settings.yaml is git-ignored; some values are environment-set)."""

from __future__ import annotations

import os
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
