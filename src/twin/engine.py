"""Run the engine as a separate program (GPL boundary: a subprocess speaking JSON Lines).

The engine is never linked into Python. `run_batch` starts long-lived `iirabm_cli` processes,
feeds them configs in small chunks, and returns results in input order.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
from dataclasses import dataclass, field

import numpy as np

from twin.config import engine_bin

SERIES_NAMES = [
    "oxy_deficit", "total_infection", "TNF", "sTNFr", "IL10", "GCSF", "proTH1", "proTH2", "IFNg",
    "PAF", "IL1", "IL4", "IL8", "IL12", "sIL1r", "IL1ra", "pmn", "mono", "TH1", "TH2",
]  # fmt: skip
MEDIATORS = [
    "PAF",
    "TNF",
    "sTNFr",
    "IL1",
    "sIL1r",
    "IL1ra",
    "IFNg",
    "IL4",
    "IL8",
    "IL10",
    "IL12",
    "GCSF",
]


class EngineError(RuntimeError):
    """The engine rejected or failed a config."""

    def __init__(self, config_id: str, message: str):
        super().__init__(f"engine error for config {config_id!r}: {message}")
        self.config_id = config_id


@dataclass
class Result:
    id: str
    died: bool
    death_step: int | None
    steps_run: int
    final_oxy_deficit: float
    final_total_infection: float
    engine_fixes: bool
    step: np.ndarray  # step labels of the records
    series: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict) -> Result:
        series = {
            k: np.asarray(v, dtype=np.float32) for k, v in obj["series"].items() if k != "step"
        }
        return cls(
            id=obj["id"],
            died=obj["died"],
            death_step=obj["death_step"],
            steps_run=obj["steps_run"],
            final_oxy_deficit=obj["final"]["oxy_deficit"],
            final_total_infection=obj["final"]["total_infection"],
            engine_fixes=obj["engine_fixes"],
            step=np.asarray(obj["series"]["step"], dtype=np.int32),
            series=series,
        )

    def value_at(self, name: str, step: int) -> float:
        """Recorded value of a series at a step label (must be a recorded step)."""
        idx = np.searchsorted(self.step, step)
        if idx >= len(self.step) or self.step[idx] != step:
            raise KeyError(f"step {step} was not recorded")
        return float(self.series[name][idx])


def _worker(cmd: list[str], work: queue.Queue, out: dict, errors: list) -> None:
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        while True:
            item = work.get()
            if item is None:
                break
            start, chunk = item
            proc.stdin.write("".join(json.dumps(c) + "\n" for c in chunk))
            proc.stdin.flush()
            for offset in range(len(chunk)):
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"engine process ended early (configs {start}..): "
                        f"{proc.stderr.read()[-2000:]}"
                    )
                out[start + offset] = json.loads(line)
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        errors.append(exc)
        # Drain the queue so other workers finish instead of blocking.
        while not work.empty():
            try:
                work.get_nowait()
            except queue.Empty:
                break
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait(timeout=60)


def run_batch(
    configs: list[dict],
    n_workers: int | None = None,
    chunk_size: int = 2,
    bin_path: str | None = None,
) -> list[Result]:
    """Run configs on `n_workers` engine processes; results are in input order.

    Raises EngineError, naming the config id, if the engine reports an error for any config.
    """
    if not configs:
        return []
    cmd = shlex.split(bin_path or engine_bin())
    n_workers = max(1, min(n_workers or os.cpu_count() or 1, len(configs)))
    work: queue.Queue = queue.Queue()
    for start in range(0, len(configs), chunk_size):
        work.put((start, configs[start : start + chunk_size]))
    for _ in range(n_workers):
        work.put(None)
    out: dict[int, dict] = {}
    errors: list = []
    threads = [
        threading.Thread(target=_worker, args=(cmd, work, out, errors)) for _ in range(n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    results = []
    for i, cfg in enumerate(configs):
        obj = out[i]
        if "error" in obj:
            raise EngineError(str(obj.get("id", cfg.get("id"))), obj["error"])
        results.append(Result.from_json(obj))
    return results
