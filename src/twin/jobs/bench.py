"""SageMaker Processing entry point: engine throughput versus number of worker processes.

Runs the same configs with several worker counts and writes bench.json (steps per second, engine
CPU time per step, user and system split) and the CPU model. Used once to choose how many workers
to run per instance (the engine's working set, about 1.5 MB per process, competes for cache).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

from twin.engine import run_batch
from twin.jobs.simulate import IN_DIR, OUT_DIR


def cpu_model() -> dict:
    info = {"cpu_count": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
        for idx in (0, 1, 2, 3):
            d = Path(f"/sys/devices/system/cpu/cpu0/cache/index{idx}")
            if d.exists():
                info[f"cache_index{idx}"] = (
                    f"{(d / 'level').read_text().strip()}/{(d / 'size').read_text().strip()}"
                )
    except OSError:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_DIR)
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", default="36,27,18,12,6")
    ap.add_argument("--n-configs", type=int, default=72)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configs = []
    for p in sorted(args.input.rglob("*.jsonl")):
        configs += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    configs = configs[: args.n_configs]
    out = {"cpu": cpu_model(), "n_configs": len(configs), "trials": []}
    for w in [int(x) for x in args.workers.split(",")]:
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        t0 = time.time()
        results = run_batch(configs, n_workers=w, chunk_size=1)
        wall = time.time() - t0
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        steps = sum(r.steps_run for r in results)
        user, system = r1.ru_utime - r0.ru_utime, r1.ru_stime - r0.ru_stime
        trial = {
            "workers": w,
            "wall_s": wall,
            "steps": steps,
            "steps_per_s": steps / wall,
            "ms_per_step_cpu": 1000 * (user + system) / steps,
            "user_s": user,
            "system_s": system,
        }
        out["trials"].append(trial)
        print(json.dumps(trial), flush=True)
    (args.output / "bench.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
