"""Measurement model: noisy, censored cytokine readings every 6 h.

An `Observations` object is everything the twin may know about a patient: reading times, values,
which readings are censored (below the assay's lower limit of detection) or missing, the panel
and the decision time. Nothing else about the patient (no parameters, no seeds, no later data).

Model cytokine totals are in arbitrary units; downstream code works in log1p space.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import yaml

from twin.config import CONFIG_DIR, DATA_DIR, DECISION_STEP, STEPS_PER_READING

DEFAULT_PANEL = ["TNF", "IL1", "IL1ra", "IL10", "IL8", "GCSF", "IL12", "IFNg"]


@dataclass(frozen=True)
class MeasurementModel:
    """Assay noise (multiplicative log-normal), detection limits and missingness."""

    noise_cv: float = 0.25
    missing_fraction: float = 0.0
    lod: dict[str, float] | None = None  # per cytokine; values below are censored
    panel: tuple[str, ...] = tuple(DEFAULT_PANEL)
    interval_steps: int = STEPS_PER_READING
    decision_step: int = DECISION_STEP

    @classmethod
    def load(cls, path=None, **overrides) -> MeasurementModel:
        path = path or CONFIG_DIR / "measurement.yaml"
        raw = yaml.safe_load(open(path)) if path.exists() else {}
        kw = dict(
            noise_cv=raw.get("noise_cv", 0.25),
            missing_fraction=raw.get("missing_fraction", 0.0),
            lod=raw.get("lod"),
            panel=tuple(raw.get("panel", DEFAULT_PANEL)),
        )
        kw.update(overrides)
        return cls(**kw)

    @property
    def sigma(self) -> float:
        """Standard deviation of log(noise): CV c gives sigma = sqrt(log(1 + c^2))."""
        return float(np.sqrt(np.log1p(self.noise_cv**2)))

    def reading_steps(self) -> np.ndarray:
        return np.arange(self.interval_steps, self.decision_step + 1, self.interval_steps)


@dataclass(frozen=True)
class Observations:
    patient_id: str
    panel: tuple[str, ...]
    steps: np.ndarray  # (T,) step labels of the readings, all <= decision_step
    values: np.ndarray  # (T, P) readings; NaN where censored or missing
    censored: np.ndarray  # (T, P) bool: below the limit of detection (value not reported)
    missing: np.ndarray  # (T, P) bool: reading not taken
    lod: np.ndarray  # (P,) limit of detection per cytokine (arbitrary units)
    noise_cv: float
    decision_step: int

    def __post_init__(self):
        if self.steps.max() > self.decision_step:
            raise ValueError("observations after the decision time")
        if np.any(np.isnan(self.values) & ~self.censored & ~self.missing):
            raise ValueError("NaN reading that is neither censored nor missing")

    @property
    def sigma(self) -> float:
        return float(np.sqrt(np.log1p(self.noise_cv**2)))

    def to_json(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "panel": list(self.panel),
            "steps": self.steps.tolist(),
            "values": [[None if np.isnan(v) else float(v) for v in row] for row in self.values],
            "censored": self.censored.tolist(),
            "missing": self.missing.tolist(),
            "lod": self.lod.tolist(),
            "noise_cv": self.noise_cv,
            "decision_step": self.decision_step,
        }

    @classmethod
    def from_json(cls, obj: dict) -> Observations:
        values = np.array(
            [[np.nan if v is None else v for v in row] for row in obj["values"]], dtype=float
        )
        return cls(
            patient_id=obj["patient_id"],
            panel=tuple(obj["panel"]),
            steps=np.array(obj["steps"], dtype=int),
            values=values,
            censored=np.array(obj["censored"], dtype=bool),
            missing=np.array(obj["missing"], dtype=bool),
            lod=np.array(obj["lod"], dtype=float),
            noise_cv=obj["noise_cv"],
            decision_step=obj["decision_step"],
        )


def measurement_rng(patient_id: str) -> np.random.Generator:
    """Measurement noise generator, seeded from the patient id and independent of engine seeds."""
    digest = hashlib.blake2b(f"measurement/{patient_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def make_observations(
    patient_id: str,
    true_series: dict[str, np.ndarray],
    record_steps: np.ndarray,
    model: MeasurementModel,
) -> Observations:
    """Draw readings from a patient's true series (recorded at `record_steps`).

    Only the builder of virtual patients calls this: it needs the truth. A reading at step s is
    drawn from the truth value labelled s; the readings stop at the decision time.
    """
    if model.lod is None:
        raise ValueError("measurement model has no limits of detection (config/measurement.yaml)")
    steps = model.reading_steps()
    rec_index = {int(s): i for i, s in enumerate(record_steps)}
    rng = measurement_rng(patient_id)
    P = len(model.panel)
    truth = np.array(
        [[true_series[c][rec_index[int(s)]] for c in model.panel] for s in steps], dtype=float
    )
    noise = rng.normal(-0.5 * model.sigma**2, model.sigma, size=truth.shape)  # mean-preserving
    noisy = truth * np.exp(noise)
    lod = np.array([model.lod[c] for c in model.panel], dtype=float)
    # Draw the missingness mask for every reading so the stream does not depend on censoring.
    missing = rng.random(truth.shape) < model.missing_fraction
    censored = (noisy < lod[None, :]) & ~missing
    values = np.where(censored | missing, np.nan, noisy)
    assert values.shape == (len(steps), P)
    return Observations(
        patient_id=patient_id,
        panel=tuple(model.panel),
        steps=steps,
        values=values,
        censored=censored,
        missing=missing,
        lod=lod,
        noise_cv=model.noise_cv,
        decision_step=model.decision_step,
    )


def observations_path(patient_id: str):
    return DATA_DIR / "observations" / f"{patient_id}.json"


def save_observations(obs: Observations) -> None:
    path = observations_path(obs.patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obs.to_json()))


def load_observations(patient_id: str) -> Observations:
    return Observations.from_json(json.loads(observations_path(patient_id).read_text()))
