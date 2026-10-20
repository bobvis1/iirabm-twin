"""Read the simulation job's Parquet output (runs-*.parquet, series-*.parquet) back into
engine Result objects, keyed by run id."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from twin.engine import SERIES_NAMES, Result


def load_results(results_dir: Path) -> dict[str, Result]:
    out: dict[str, Result] = {}
    runs = [pq.read_table(f).to_pydict() for f in sorted(results_dir.glob("runs-*.parquet"))]
    series = {}
    for f in sorted(results_dir.glob("series-*.parquet")):
        t = pq.read_table(f)
        ids = np.asarray(t.column("id").to_pylist())
        step = np.asarray(t.column("step"))
        cols = {n: np.asarray(t.column(n)) for n in SERIES_NAMES}
        for rid in dict.fromkeys(ids.tolist()):
            m = ids == rid
            series[rid] = (step[m], {n: cols[n][m] for n in SERIES_NAMES})
    for r in runs:
        for i, rid in enumerate(r["id"]):
            died = bool(r["died"][i])
            step, ser = series.get(
                rid, (np.array([], dtype=np.int32), {n: np.array([]) for n in SERIES_NAMES})
            )
            out[rid] = Result(
                id=rid,
                died=died,
                death_step=int(r["death_step"][i]) if died else None,
                steps_run=int(r["steps_run"][i]),
                final_oxy_deficit=float(r["final_oxy_deficit"][i]),
                final_total_infection=float(r["final_total_infection"][i]),
                engine_fixes=bool(r["engine_fixes"][i]),
                step=step.astype(np.int32),
                series={n: v.astype(np.float32) for n, v in ser.items()},
            )
    return out
