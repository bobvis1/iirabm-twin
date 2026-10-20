"""Test session setup: scratch data/truth directories and a local engine, if available."""

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Must happen before twin.config is imported.
_scratch = tempfile.mkdtemp(prefix="twin-tests-")
os.environ["TWIN_DATA_DIR"] = str(Path(_scratch) / "data")
os.environ["TWIN_TRUTH_DIR"] = str(Path(_scratch) / "truth")

if not os.environ.get("IIRABM_BIN"):
    wrapper = ROOT / "scripts" / "iirabm_docker.sh"
    built = list((ROOT / "engine" / "build").glob("Linux-*/iirabm_cli"))
    if shutil.which("docker") and built:
        os.environ["IIRABM_BIN"] = str(wrapper)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_scratch, ignore_errors=True)
