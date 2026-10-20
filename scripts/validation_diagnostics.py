"""Post-hoc sanity checks on the validation (run AFTER scoring; nothing here is tuned or selected).

The test-cohort Brier score was much better than the development one. A result that looks too good
is a reason to suspect leakage of the truth or reuse of seeds, so:
 1. shuffled observations: calibrate each patient with ANOTHER patient's readings; the forecast
    must fall back to about prior-only skill (no leak through anything but the patient's own readings);
 2. the "oracle" forecast: mortality among the 64 library design points nearest the patient's TRUE
    parameters, a rough upper bound on what knowing theta would give, per cohort;
 3. seed ranges: no patient seed lies in the library, calibration, search or future ranges.
Writes docs/results/validation_diagnostics.json.
"""

from __future__ import annotations

import json

import numpy as np
import yaml

from twin import metrics, seeds
from twin.calibrate import CalibrationConfig, library_posterior
from twin.config import CONFIG_DIR, DOCS_DIR
from twin.forecast import ForecastEngine
from twin.librarydata import load_or_build
from twin.observe import load_observations
from twin.params import PARAM_NAMES
from twin.truth import list_patients, read_truth


def forecasts(ids, fe, lib, cfg, shift: int) -> np.ndarray:
    """28-day mortality forecast of each patient using the readings of the patient `shift` places on."""
    return np.array(
        [
            fe.forecast(
                library_posterior(load_observations(ids[(i + shift) % len(ids)]), lib, cfg)
            ).mortality
            for i in range(len(ids))
        ]
    )


def main() -> None:
    lib = load_or_build()
    fe = ForecastEngine(lib)
    cfg = CalibrationConfig(
        **yaml.safe_load((CONFIG_DIR / "calibration.yaml").read_text())["library_posterior"]
    )
    sd = lib.theta.std(0)
    out: dict = {}
    for cohort in ("development", "test"):
        ids = list_patients(cohort)
        truths = [read_truth(i) for i in ids]
        y = np.array([t.died for t in truths], float)

        p, ps = forecasts(ids, fe, lib, cfg, 0), forecasts(ids, fe, lib, cfg, 1)
        oracle = []
        for t in truths:
            th = np.array([getattr(t.theta, n) for n in PARAM_NAMES], float)
            nn = np.argsort((((lib.theta - th) / sd) ** 2).sum(1))[:64]
            alive = lib.alive_td[nn]
            oracle.append((lib.died[nn] & alive).sum() / max(alive.sum(), 1))
        oracle = np.array(oracle)
        out[cohort] = {
            "n": len(ids),
            "observed_mortality": float(y.mean()),
            "matched": {"brier": metrics.brier(p, y), "auroc": metrics.auroc(p, y)},
            "shuffled_observations": {"brier": metrics.brier(ps, y), "auroc": metrics.auroc(ps, y)},
            "oracle_true_theta_neighbours": {
                "brier": metrics.brier(oracle, y),
                "auroc": metrics.auroc(oracle, y),
            },
            "patient_seed_ranges": sorted({seeds.role_of(t.seed) for t in truths}),
        }
    (DOCS_DIR / "results" / "validation_diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
