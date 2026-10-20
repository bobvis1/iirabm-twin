"""SageMaker Processing entry point: run engine configs and write Parquet.

    python -m twin.jobs.simulate [--input DIR] [--output DIR] [--workers N]

Reads every *.jsonl file in the input directory (SageMaker shards the S3 prefix across instances
with ShardedByS3Key), runs the configs on one engine process per vCPU, and writes
  runs-<shard>.parquet     one row per run: id, theta_id, replicate, seed, died, death_step, ...
  series-<shard>.parquet   long format: id, step, one column per recorded series
  timing.json              total steps, wall time and CPU-time-per-step for sizing later jobs
Config ids are "<theta_id>/<replicate>" (any string is accepted; the parts are split if present).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from twin.engine import SERIES_NAMES, Result, run_batch

IN_DIR = Path("/opt/ml/processing/input")
OUT_DIR = Path("/opt/ml/processing/output")
CHUNK = 256  # configs per run_batch call; bounds memory and gives progress lines

RUN_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("theta_id", pa.int32()),
        ("replicate", pa.int16()),
        ("seed", pa.int64()),
        ("died", pa.bool_()),
        ("death_step", pa.int32()),
        ("steps_run", pa.int32()),
        ("final_oxy_deficit", pa.float32()),
        ("final_total_infection", pa.float32()),
        ("engine_fixes", pa.bool_()),
    ]
)
SERIES_SCHEMA = pa.schema(
    [("id", pa.string()), ("step", pa.int32())] + [(n, pa.float32()) for n in SERIES_NAMES]
)


def split_id(run_id: str) -> tuple[int, int]:
    """'<theta_id>/<replicate>' -> (theta_id, replicate); -1, -1 if the id has another form."""
    try:
        a, b = run_id.split("/")
        return int(a), int(b)
    except ValueError:
        return -1, -1


def runs_table(results: list[Result], seeds: dict[str, int]) -> pa.Table:
    cols = {f.name: [] for f in RUN_SCHEMA}
    for r in results:
        theta_id, rep = split_id(r.id)
        cols["id"].append(r.id)
        cols["theta_id"].append(theta_id)
        cols["replicate"].append(rep)
        cols["seed"].append(seeds[r.id])
        cols["died"].append(r.died)
        cols["death_step"].append(-1 if r.death_step is None else r.death_step)
        cols["steps_run"].append(r.steps_run)
        cols["final_oxy_deficit"].append(r.final_oxy_deficit)
        cols["final_total_infection"].append(r.final_total_infection)
        cols["engine_fixes"].append(r.engine_fixes)
    return pa.table(cols, schema=RUN_SCHEMA)


def series_table(results: list[Result]) -> pa.Table:
    import numpy as np

    ids, steps = [], []
    cols = {n: [] for n in SERIES_NAMES}
    for r in results:
        ids.extend([r.id] * len(r.step))
        steps.append(r.step)
        for n in SERIES_NAMES:
            cols[n].append(r.series[n])
    data = {"id": ids, "step": np.concatenate(steps) if steps else np.array([], dtype=np.int32)}
    for n in SERIES_NAMES:
        data[n] = np.concatenate(cols[n]) if cols[n] else np.array([], dtype=np.float32)
    return pa.table(data, schema=SERIES_SCHEMA)


def process_shard(path: Path, out_dir: Path, workers: int) -> dict:
    configs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for c in configs:
        c.setdefault("output_stride", 51)
    seeds = {c["id"]: c["seed"] for c in configs}
    stem = path.stem
    runs_w = pq.ParquetWriter(out_dir / f"runs-{stem}.parquet", RUN_SCHEMA, compression="zstd")
    series_w = pq.ParquetWriter(
        out_dir / f"series-{stem}.parquet", SERIES_SCHEMA, compression="zstd"
    )
    steps_total, t0 = 0, time.time()
    try:
        for i in range(0, len(configs), CHUNK):
            results = run_batch(configs[i : i + CHUNK], n_workers=workers, chunk_size=1)
            runs_w.write_table(runs_table(results, seeds))
            series_w.write_table(series_table(results))
            steps_total += sum(r.steps_run for r in results)
            done = min(i + CHUNK, len(configs))
            print(
                f"{path.name}: {done}/{len(configs)} configs, {time.time() - t0:.0f} s", flush=True
            )
    finally:
        runs_w.close()
        series_w.close()
    return {
        "shard": path.name,
        "configs": len(configs),
        "steps": steps_total,
        "wall_s": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_DIR)
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.input.rglob("*.jsonl"))
    if not shards:
        raise SystemExit(f"no *.jsonl shards under {args.input}")
    cpu0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    infos = [process_shard(p, args.output, args.workers) for p in shards]
    wall = time.time() - t0
    cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    steps = sum(i["steps"] for i in infos)
    timing = {
        "workers": args.workers,
        "shards": infos,
        "total_steps": steps,
        "wall_s": wall,
        # Wall time x workers / steps: per-step cost with every vCPU busy (an upper bound on the
        # serial cost, since workers share caches and memory bandwidth).
        "ms_per_step_per_worker": 1000 * wall * args.workers / max(steps, 1),
        "cpu_hours": wall * args.workers / 3600,
        # CPU time actually used by the engine processes (user + system): the cost per step that
        # matters for sizing, including any slowdown from all vCPUs being busy at once.
        "engine_cpu_s": cpu_s,
        "ms_per_step_cpu": 1000 * cpu_s / max(steps, 1),
    }
    (args.output / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps({k: v for k, v in timing.items() if k != "shards"}), flush=True)


if __name__ == "__main__":
    main()
