"""Seed ranges. They never overlap, so no random number stream is shared between roles.

    library      [        1,   500_000)  simulation library: (design point, replicate) pairs
    calibration  [  500_000,  750_000)   fresh simulations inside the GA calibration
    search       [  750_000, 1_000_000)  fresh simulations inside the treatment search
    patient      [1_000_000, 2_000_000)  virtual-patient history seeds
    future       [2_000_000, 3_000_000)  post-decision futures used only to score treatments

CLAUDE.md names three ranges (patients, library and calibration, futures); the first is split into
three here so library runs are never reused by a calibration or a search.
"""

from __future__ import annotations

import hashlib

LIBRARY = (1, 500_000)
CALIBRATION = (500_000, 750_000)
SEARCH = (750_000, 1_000_000)
PATIENT = (1_000_000, 2_000_000)
FUTURE = (2_000_000, 3_000_000)

RANGES = {
    "library": LIBRARY,
    "calibration": CALIBRATION,
    "search": SEARCH,
    "patient": PATIENT,
    "future": FUTURE,
}

MAX_REPLICATES = 16  # library replicates per design point
PILOT_THETA_OFFSET = 30_000  # pilot sweeps use design-point ids from here; the library stays below
MAX_LIBRARY_DESIGN_POINTS = PILOT_THETA_OFFSET


_STRIDE = 7919  # prime, coprime with every range size here, so label + k * stride never repeats


def _base(label: str, rng: tuple[int, int]) -> int:
    """An offset in [0, hi - lo) that depends only on the label."""
    lo, hi = rng
    digest = hashlib.blake2b(label.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (hi - lo)


def _sequence(label: str, k: int, rng: tuple[int, int]) -> int:
    """Seed k of the sequence `label`: distinct for different k, reproducible from the label."""
    lo, hi = rng
    return lo + (_base(label, rng) + k * _STRIDE) % (hi - lo)


def library_seed(theta_id: int, k: int) -> int:
    """Seed of replicate k of design point theta_id. Unique for every (theta_id, k) pair."""
    if not 0 <= k < MAX_REPLICATES:
        raise ValueError(f"replicate index must be in [0, {MAX_REPLICATES})")
    if theta_id < 0:
        raise ValueError("theta_id must be >= 0")
    seed = LIBRARY[0] + theta_id * MAX_REPLICATES + k
    if not LIBRARY[0] <= seed < LIBRARY[1]:
        raise ValueError(f"theta_id {theta_id} is too large for the library seed range")
    return seed


def calibration_seed(patient_id: str, k: int) -> int:
    return _sequence(f"calibration/{patient_id}", k, CALIBRATION)


def search_seed(label: str, k: int) -> int:
    return _sequence(f"search/{label}", k, SEARCH)


def patient_seed(candidate_index: int) -> int:
    """History seed of candidate number candidate_index (distinct for the first 1,000,000)."""
    return PATIENT[0] + (candidate_index * _STRIDE) % (PATIENT[1] - PATIENT[0])


def future_seed(patient_id: str, r: int) -> int:
    """Seed of future r of a patient, used only to score treatments."""
    return _sequence(f"future/{patient_id}", r, FUTURE)


def role_of(seed: int) -> str:
    for name, (lo, hi) in RANGES.items():
        if lo <= seed < hi:
            return name
    raise ValueError(f"seed {seed} is outside every reserved range")
