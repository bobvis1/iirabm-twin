"""Acceptance figure for Phase 4: for three DEVELOPMENT patients, the readings, the twin's forecast
bands (library posterior, the settings in config/calibration.yaml) and the hidden truth.

    uv run python scripts/plot_forecast_figure.py dev-001 dev-002 dev-003
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from twin.calibrate import CalibrationConfig, library_posterior  # noqa: E402
from twin.config import CONFIG_DIR, DATA_DIR, DECISION_STEP, DOCS_DIR, STEPS_PER_READING  # noqa: E402
from twin.evaluate import assert_development_only  # noqa: E402
from twin.librarydata import load_or_build  # noqa: E402
from twin.observe import load_observations  # noqa: E402
from twin.truth import read_truth  # noqa: E402

VARS = ["oxy_deficit", "TNF", "IL8", "IL10"]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
N_REC = 112  # records at 51 .. 5712


def load_series(results_dir, n: int, k: int) -> np.ndarray:
    """(N*K, N_REC, len(VARS)) library values at every record (NaN after death)."""
    out = np.full((n * k, N_REC, len(VARS)), np.nan, dtype=np.float32)
    for rpath in sorted(results_dir.glob("runs-*.parquet")):
        runs = pq.read_table(rpath, columns=["id", "theta_id", "replicate"])
        flat = np.asarray(runs.column("theta_id")) * k + np.asarray(runs.column("replicate"))
        series = pq.read_table(
            results_dir / f"series-{rpath.name[5:]}", columns=["id", "step", *VARS]
        )
        idx = pc.index_in(series.column("id"), value_set=runs.column("id")).to_numpy(
            zero_copy_only=False
        )
        rec = np.asarray(series.column("step")) // STEPS_PER_READING - 1
        ok = rec < N_REC
        for j, v in enumerate(VARS):
            out[flat[idx][ok], rec[ok], j] = np.asarray(series.column(v))[ok]
    return out


def weighted_bands(values: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    """Weighted quantiles of values (R,) with weights w (R,), ignoring NaN."""
    m = ~np.isnan(values) & (w > 0)
    if not m.any():
        return np.full(len(qs), np.nan)
    v, ww = values[m], w[m]
    o = np.argsort(v)
    c = np.cumsum(ww[o]) / ww.sum()
    return v[o][np.minimum(np.searchsorted(c, qs), len(c) - 1)]


def main(ids: list[str]) -> None:
    assert_development_only(ids)
    lib = load_or_build()
    cfg = CalibrationConfig(
        **yaml.safe_load((CONFIG_DIR / "calibration.yaml").read_text())["library_posterior"]
    )
    results = sorted((DATA_DIR / "library" / "results").glob("twin-library-*"))[-1]
    series = load_series(results, lib.n, lib.k)
    qs = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    fig, axes = plt.subplots(len(ids), len(VARS), figsize=(16, 3.4 * len(ids)), squeeze=False)
    for r, pid in enumerate(ids):
        truth, obs = read_truth(pid), load_observations(pid)
        w = library_posterior(obs, lib, cfg).run_w.reshape(-1)
        bands = np.array(
            [
                [weighted_bands(series[:, t, j], w, qs) for t in range(N_REC)]
                for j in range(len(VARS))
            ]
        )
        hours = np.arange(1, N_REC + 1) * STEPS_PER_READING * 7 / 60
        fut = np.arange(1, N_REC + 1) * STEPS_PER_READING > DECISION_STEP
        for c, v in enumerate(VARS):
            ax = axes[r, c]
            b = bands[c]
            ax.fill_between(
                hours[fut], b[fut, 0], b[fut, 6], color=BLUE, alpha=0.12, lw=0, label="95%"
            )
            ax.fill_between(
                hours[fut], b[fut, 1], b[fut, 5], color=BLUE, alpha=0.20, lw=0, label="80%"
            )
            ax.fill_between(
                hours[fut], b[fut, 2], b[fut, 4], color=BLUE, alpha=0.30, lw=0, label="50%"
            )
            ax.plot(hours[fut], b[fut, 3], color=BLUE, lw=1.4, label="forecast median")
            ax.plot(
                truth.record_steps * 7 / 60,
                truth.series[v],
                color=INK,
                lw=1.0,
                label="truth (hidden)",
            )
            if v in obs.panel:
                j = obs.panel.index(v)
                ok = ~(obs.censored[:, j] | obs.missing[:, j])
                ax.plot(
                    obs.steps[ok] * 7 / 60,
                    obs.values[ok, j],
                    "o",
                    color=ORANGE,
                    ms=4,
                    label="readings",
                )
            ax.axvline(DECISION_STEP * 7 / 60, color=MUTED, ls="--", lw=1)
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{pid}: {v}", fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == len(ids) - 1:
                ax.set_xlabel("hours since injury", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("Library-posterior forecasts for three development patients (in this model); "
                 "bands are conditional on being alive at each time", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "figures" / "forecast_dev_patients.png", dpi=140)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dev-001", "dev-002", "dev-003"])
