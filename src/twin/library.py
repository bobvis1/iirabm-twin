"""Simulation library: design points x library seeds, as engine configs and as JSON Lines shards.

The library is N scrambled Sobol points (twin.params.Prior.sobol) x K replicates, each run for the
full horizon at output_stride 51. Config ids are "<theta_id>/<replicate>"; the seed of replicate k
of design point t is twin.seeds.library_seed(t, k).
"""

from __future__ import annotations

import json
from pathlib import Path

from twin import seeds
from twin.config import HORIZON_STEPS, STEPS_PER_READING
from twin.params import Prior, Theta

SCRAMBLE_SEED = 20261019  # fixed: the design is part of the experiment definition


def design(prior: Prior, n: int, start: int = 0) -> list[Theta]:
    """Design points start..start+n-1; theta_id is the index in the Sobol sequence."""
    return prior.sobol(n, SCRAMBLE_SEED, start=start)


def library_configs(
    thetas: list[Theta], first_theta_id: int, k: int, num_steps: int = HORIZON_STEPS
) -> list[dict]:
    configs = []
    for i, th in enumerate(thetas):
        tid = first_theta_id + i
        for rep in range(k):
            configs.append(
                dict(
                    th.to_engine(),
                    id=f"{tid}/{rep}",
                    seed=seeds.library_seed(tid, rep),
                    num_steps=num_steps,
                    output_stride=STEPS_PER_READING,
                )
            )
    return configs


def write_shards(
    configs: list[dict], out_dir: Path, n_shards: int, prefix: str = "shard"
) -> list[Path]:
    """Write configs round-robin into n_shards JSON Lines files, so every shard has a mix of
    cheap (early death) and expensive (28-day) runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{prefix}-{i:03d}.jsonl" for i in range(n_shards)]
    files = [p.open("w") for p in paths]
    try:
        for j, cfg in enumerate(configs):
            files[j % n_shards].write(json.dumps(cfg) + "\n")
    finally:
        for f in files:
            f.close()
    return paths
