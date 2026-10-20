"""The simulation library, condensed into arrays for calibration and forecasting.

The SageMaker job writes long-format Parquet (about 4 million rows). Calibration needs only the
panel readings up to the decision time, and forecasting needs oxy_deficit and the panel at a few
target steps, so those are pulled out once into data/library/condensed.npz.

Arrays have shape (N design points, K seeds, ...):
    panel_pre   (N, K, 12, P)   panel values at readings 51..612 (NaN if the run had died)
    targets     (N, K, 4, 1+P)  oxy_deficit then the panel at the four target steps (NaN if dead)
    alive_td    (N, K)          alive at the decision time (a record exists at step 612)
    died        (N, K)          dead before the end of the horizon
    death_step  (N, K)          -1 if the run survived
    theta       (N, 5)          the design points

"Day" targets are mapped to the nearest recorded step (records are every 51 steps = 6 h):
day 4 -> 816, day 7 -> 1428, day 14 -> 2856, day 28 -> 5712 (the last record, about day 27.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from twin.config import DATA_DIR, DECISION_STEP, STEPS_PER_READING
from twin.observe import DEFAULT_PANEL
from twin.params import PARAM_NAMES

TARGET_DAYS = (4, 7, 14, 28)
TARGET_STEPS = (816, 1428, 2856, 5712)
TARGET_VARS = ["oxy_deficit"] + DEFAULT_PANEL


@dataclass
class LibraryData:
    theta: np.ndarray
    panel: tuple[str, ...]
    panel_pre: np.ndarray
    targets: np.ndarray
    alive_td: np.ndarray
    died: np.ndarray
    death_step: np.ndarray
    target_steps: tuple[int, ...] = TARGET_STEPS

    @property
    def n(self) -> int:
        return self.theta.shape[0]

    @property
    def k(self) -> int:
        return self.panel_pre.shape[1]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, theta=self.theta, panel=np.array(self.panel), panel_pre=self.panel_pre,
            targets=self.targets, alive_td=self.alive_td, died=self.died,
            death_step=self.death_step,
        )  # fmt: skip

    @classmethod
    def load(cls, path: Path | None = None) -> LibraryData:
        path = path or DATA_DIR / "library" / "condensed.npz"
        with np.load(path) as z:
            return cls(
                theta=z["theta"], panel=tuple(str(x) for x in z["panel"]), panel_pre=z["panel_pre"],
                targets=z["targets"], alive_td=z["alive_td"], died=z["died"],
                death_step=z["death_step"],
            )  # fmt: skip


def condense(results_dir: Path, panel: tuple[str, ...] = tuple(DEFAULT_PANEL)) -> LibraryData:
    """Build LibraryData from the job's runs-*.parquet, series-*.parquet and thetas.parquet."""
    thetas = pq.read_table(results_dir / "thetas.parquet")
    n = thetas.num_rows
    theta = np.stack([np.asarray(thetas.column(c), dtype=float) for c in PARAM_NAMES], axis=1)
    runs = [pq.read_table(f) for f in sorted(results_dir.glob("runs-*.parquet"))]
    k = max(int(np.asarray(r.column("replicate")).max()) for r in runs) + 1
    n_pre = DECISION_STEP // STEPS_PER_READING
    P = len(panel)
    panel_pre = np.full((n, k, n_pre, P), np.nan, dtype=np.float32)
    targets = np.full((n, k, len(TARGET_STEPS), 1 + P), np.nan, dtype=np.float32)
    died = np.zeros((n, k), dtype=bool)
    death_step = np.full((n, k), -1, dtype=np.int32)
    seen = np.zeros((n, k), dtype=bool)
    cols = ["id", "step"] + ["oxy_deficit"] + list(panel)
    for rpath in sorted(results_dir.glob("runs-*.parquet")):
        stem = rpath.name[len("runs-") :]
        runs_t = pq.read_table(rpath)
        tid = np.asarray(runs_t.column("theta_id"))
        rep = np.asarray(runs_t.column("replicate"))
        died[tid, rep] = np.asarray(runs_t.column("died"))
        death_step[tid, rep] = np.asarray(runs_t.column("death_step"))
        seen[tid, rep] = True
        series = pq.read_table(results_dir / f"series-{stem}", columns=cols)
        idx = pc.index_in(series.column("id"), value_set=runs_t.column("id")).to_numpy(
            zero_copy_only=False
        )
        t_i, r_i = tid[idx], rep[idx]
        step = np.asarray(series.column("step"))
        rec = step // STEPS_PER_READING - 1
        vals = {c: np.asarray(series.column(c)) for c in cols[2:]}
        pre = step <= DECISION_STEP
        for j, c in enumerate(panel):
            panel_pre[t_i[pre], r_i[pre], rec[pre], j] = vals[c][pre]
        for ti, ts in enumerate(TARGET_STEPS):
            m = step == ts
            targets[t_i[m], r_i[m], ti, 0] = vals["oxy_deficit"][m]
            for j, c in enumerate(panel):
                targets[t_i[m], r_i[m], ti, 1 + j] = vals[c][m]
    if not seen.all():
        raise ValueError("library is incomplete: some (design point, seed) pairs are missing")
    alive_td = ~died | (death_step > DECISION_STEP)
    return LibraryData(theta, panel, panel_pre, targets, alive_td, died, death_step)


def load_or_build(results_dir: Path | None = None) -> LibraryData:
    cache = DATA_DIR / "library" / "condensed.npz"
    if cache.exists():
        return LibraryData.load(cache)
    results = results_dir or sorted((DATA_DIR / "library" / "results").glob("twin-library-*"))[-1]
    lib = condense(results)
    lib.save(cache)
    return lib


def from_results(
    theta: np.ndarray, results: list[list], panel: tuple[str, ...] = tuple(DEFAULT_PANEL)
) -> LibraryData:
    """LibraryData from in-memory engine results: results[i][k] is replicate k of theta[i].

    Every result must have been run for the full horizon with output_stride 51. Used for the
    full-horizon forecast runs of the GA calibration, which are a small library of their own.
    """
    n, k = len(results), len(results[0])
    n_pre = DECISION_STEP // STEPS_PER_READING
    P = len(panel)
    panel_pre = np.full((n, k, n_pre, P), np.nan, dtype=np.float32)
    targets = np.full((n, k, len(TARGET_STEPS), 1 + P), np.nan, dtype=np.float32)
    died = np.zeros((n, k), dtype=bool)
    death_step = np.full((n, k), -1, dtype=np.int32)
    for i in range(n):
        for j, r in enumerate(results[i]):
            died[i, j] = r.died
            death_step[i, j] = -1 if r.death_step is None else r.death_step
            pre = r.step <= DECISION_STEP
            for c_i, c in enumerate(panel):
                panel_pre[i, j, : pre.sum(), c_i] = r.series[c][pre]
            for ti, ts in enumerate(TARGET_STEPS):
                hit = np.flatnonzero(r.step == ts)
                if len(hit):
                    targets[i, j, ti, 0] = r.series["oxy_deficit"][hit[0]]
                    for c_i, c in enumerate(panel):
                        targets[i, j, ti, 1 + c_i] = r.series[c][hit[0]]
    alive_td = ~died | (death_step > DECISION_STEP)
    return LibraryData(
        np.asarray(theta, dtype=float), panel, panel_pre, targets, alive_td, died, death_step
    )
