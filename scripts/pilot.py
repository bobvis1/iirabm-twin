"""Phase 2.3 pilot sweep: random theta x library seeds over the full 28-day horizon.

Wide starting ranges around the V1 defaults (inj 33, oxyHeal 0.1, spread 5, recur 2, repeat 2).
Writes data/pilot/pilot.npz (git-ignored). Needs IIRABM_BIN (see scripts/iirabm_docker.sh).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from twin import seeds
from twin.config import DATA_DIR, HORIZON_STEPS
from twin.engine import SERIES_NAMES, run_batch
from twin.params import Prior

WIDE = Prior(
    {
        "inj_number": (10, 50),
        "oxyHeal": (0.03, 0.30),
        "infectSpread": (2, 12),
        "numRecurInj": (1, 6),
        "numInfectRepeat": (1, 6),
    }
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-theta", type=int, default=300)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.default_rng(20261019)
    thetas = WIDE.sample(rng, args.n_theta)
    configs = []
    for i, th in enumerate(thetas):
        for k in range(args.n_seeds):
            seed = seeds.library_seed(seeds.PILOT_THETA_OFFSET + i, k)
            configs.append(
                dict(
                    th.to_engine(),
                    id=f"pilot-{i}-{k}",
                    seed=seed,
                    num_steps=HORIZON_STEPS,
                    output_stride=51,
                )
            )
    t0 = time.time()
    results = run_batch(configs, n_workers=args.workers, chunk_size=1)
    n_rec = HORIZON_STEPS // 51
    series = np.full(
        (args.n_theta, args.n_seeds, n_rec, len(SERIES_NAMES)), np.nan, dtype=np.float32
    )
    died = np.zeros((args.n_theta, args.n_seeds), dtype=bool)
    death_step = np.full((args.n_theta, args.n_seeds), -1, dtype=np.int32)
    for idx, res in enumerate(results):
        i, k = divmod(idx, args.n_seeds)
        died[i, k] = res.died
        death_step[i, k] = res.death_step if res.died else -1
        for j, name in enumerate(SERIES_NAMES):
            series[i, k, : len(res.step), j] = res.series[name]
    out = DATA_DIR / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "pilot.npz",
        theta=np.array([t.as_array() for t in thetas]),
        died=died,
        death_step=death_step,
        series=series,
        series_names=np.array(SERIES_NAMES),
    )
    print(f"{len(results)} runs in {time.time() - t0:.0f} s; deaths {died.mean():.2%}")


if __name__ == "__main__":
    main()
