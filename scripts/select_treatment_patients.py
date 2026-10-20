"""Apply the selection rule of docs/treatment_protocol.md: the first 5 test patients (by id) whose
twin forecasts a 28-day mortality in [0.3, 0.7]. Uses observations only, never the truth.

    uv run python scripts/select_treatment_patients.py
"""

from __future__ import annotations

import json

import yaml

from twin.calibrate import CalibrationConfig, library_posterior
from twin.config import CONFIG_DIR, DATA_DIR
from twin.forecast import ForecastEngine
from twin.librarydata import load_or_build
from twin.observe import load_observations

LOW, HIGH, N = 0.3, 0.7, 5


def main() -> None:
    lib = load_or_build()
    fe = ForecastEngine(lib)
    cfg = CalibrationConfig(
        **yaml.safe_load((CONFIG_DIR / "calibration.yaml").read_text())["library_posterior"]
    )
    chosen, table = [], []
    for i in range(1, 101):
        pid = f"test-{i:03d}"
        p = fe.forecast(library_posterior(load_observations(pid), lib, cfg)).mortality
        table.append({"patient": pid, "twin_mortality": p})
        if LOW <= p <= HIGH and len(chosen) < N:
            chosen.append(pid)
    out = DATA_DIR / "treatment"
    out.mkdir(parents=True, exist_ok=True)
    (out / "patients.json").write_text(json.dumps({"rule": f"first {N} test patients with twin mortality in [{LOW}, {HIGH}]",
                                                   "patients": chosen, "forecasts": table}, indent=1))  # fmt: skip
    print("chosen:", chosen)
    print({t["patient"]: round(t["twin_mortality"], 2) for t in table if t["patient"] in chosen})
    print("qualifying among all 100:", sum(LOW <= t["twin_mortality"] <= HIGH for t in table))


if __name__ == "__main__":
    main()
