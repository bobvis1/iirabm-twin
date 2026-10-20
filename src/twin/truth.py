"""Hidden truth of virtual patients: true parameters, seeds and full trajectories.

Stored under truth/ (git-ignored). ONLY this module reads those files, and only twin.evaluate
(Phase 5), the cohort builder and tests may import it. Calibration, forecasting, likelihood and
treatment code receive an Observations object and nothing else (tests/test_truth_isolation.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from twin.config import TRUTH_DIR
from twin.engine import Result
from twin.params import Theta


@dataclass(frozen=True)
class PatientTruth:
    patient_id: str
    cohort: str  # "development" or "test"
    theta: Theta
    seed: int  # history seed
    record_steps: np.ndarray
    series: dict[str, np.ndarray]  # true trajectories at record_steps, full horizon
    died: bool
    death_step: int | None


def _paths(patient_id: str):
    return TRUTH_DIR / f"{patient_id}.json", TRUTH_DIR / f"{patient_id}.npz"


def write_truth(patient_id: str, cohort: str, theta: Theta, seed: int, result: Result) -> None:
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    meta, arrays = _paths(patient_id)
    meta.write_text(
        json.dumps(
            {
                "patient_id": patient_id,
                "cohort": cohort,
                "theta": theta.to_engine(),
                "seed": seed,
                "died": result.died,
                "death_step": result.death_step,
            }
        )
    )
    np.savez_compressed(arrays, step=result.step, **result.series)


def read_truth(patient_id: str) -> PatientTruth:
    meta_path, arrays_path = _paths(patient_id)
    meta = json.loads(meta_path.read_text())
    with np.load(arrays_path) as z:
        series = {k: z[k] for k in z.files if k != "step"}
        steps = z["step"]
    return PatientTruth(
        patient_id=meta["patient_id"],
        cohort=meta["cohort"],
        theta=Theta(**meta["theta"]),
        seed=meta["seed"],
        record_steps=steps,
        series=series,
        died=meta["died"],
        death_step=meta["death_step"],
    )


def list_patients(cohort: str | None = None) -> list[str]:
    ids = sorted(p.stem for p in TRUTH_DIR.glob("*.json")) if TRUTH_DIR.exists() else []
    if cohort is None:
        return ids
    return [i for i in ids if json.loads(_paths(i)[0].read_text())["cohort"] == cohort]
