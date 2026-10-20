"""Summarise the pilot sweep (data/pilot/pilot.npz): survival at 72 h and 28 days by parameter,
which parameters matter, and the assay detection limits derived from the pilot (not from patients).

Writes docs/results/pilot_summary.json, docs/figures/pilot_outcomes.png and
config/measurement.yaml (limits of detection).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from twin.config import CONFIG_DIR, DATA_DIR, DECISION_STEP, DOCS_DIR  # noqa: E402
from twin.observe import DEFAULT_PANEL  # noqa: E402
from twin.params import DESCRIPTIONS, PARAM_NAMES  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette
INK, MUTED = "#0b0b0b", "#52514e"


def main() -> None:
    z = np.load(DATA_DIR / "pilot" / "pilot.npz")
    theta, died, death_step, series = z["theta"], z["died"], z["death_step"], z["series"]
    names = list(z["series_names"])
    alive72 = (~died) | (death_step > DECISION_STEP)
    alive28 = ~died
    a72, a28 = alive72.mean(1), alive28.mean(1)
    steps_run = np.where(died, death_step, 5760)

    summary = {
        "runs": int(died.size),
        "theta": int(len(theta)),
        "seeds_per_theta": int(died.shape[1]),
        "alive_at_72h": float(alive72.mean()),
        "alive_at_28d": float(alive28.mean()),
        "dead_before_72h": float((died & (death_step <= DECISION_STEP)).mean()),
        "alive_at_28d_given_alive_at_72h": float(alive28[alive72].mean()),
        "theta_with_mixed_28d_outcomes": float(((a28 > 0) & (a28 < 1)).mean()),
        "theta_all_seeds_die": float((a28 == 0).mean()),
        "theta_all_seeds_survive": float((a28 == 1).mean()),
        "mean_steps_per_run": float(steps_run.mean()),
        "spearman": {},
        "binned": {},
    }
    oxy = series[..., names.index("oxy_deficit")]
    with np.errstate(all="ignore"):
        peak72 = np.nanmax(oxy[:, :, :12], axis=2)
    summary["alive72_runs_with_peak_oxy_deficit_below_500"] = float((peak72[alive72] < 500).mean())

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=True)
    for j, (name, ax) in enumerate(zip(PARAM_NAMES, axes, strict=True)):
        rho72 = spearmanr(theta[:, j], a72)[0]
        rho28 = spearmanr(theta[:, j], a28)[0]
        summary["spearman"][name] = {"alive_at_72h": float(rho72), "alive_at_28d": float(rho28)}
        edges = np.quantile(theta[:, j], np.linspace(0, 1, 6))
        bins = np.digitize(theta[:, j], edges[1:-1])
        mid, r72, r28, rng_ = [], [], [], []
        for b in range(5):
            m = bins == b
            mid.append(theta[m, j].mean())
            r72.append(a72[m].mean())
            r28.append(a28[m].mean())
            rng_.append([float(theta[m, j].min()), float(theta[m, j].max())])
        summary["binned"][name] = {"range": rng_, "alive_at_72h": r72, "alive_at_28d": r28}
        ax.plot(mid, r72, color=BLUE, lw=2, marker="o", ms=5, label="alive at 72 h")
        ax.plot(mid, r28, color=ORANGE, lw=2, marker="o", ms=5, label="alive at 28 d")
        ax.set_title(f"{name}\nρ(72h) {rho72:+.2f}   ρ(28d) {rho28:+.2f}", fontsize=9, color=INK)
        ax.set_xlabel(DESCRIPTIONS[name].split(" (")[0], fontsize=8, color=MUTED)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("fraction of runs alive (5 equal-count bins)", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.12))
    fig.suptitle("Pilot sweep: 300 random parameter sets x 4 seeds, in this model", fontsize=11)
    fig.tight_layout()
    (DOCS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_DIR / "figures" / "pilot_outcomes.png", dpi=150)

    # Assay limits of detection: the 5th percentile of true panel values over readings 51..612
    # among runs alive at 72 h. Derived from the pilot only (never from patients).
    lod = {}
    for c in DEFAULT_PANEL:
        v = series[..., names.index(c)][:, :, :12][alive72]
        lod[c] = float(np.nanquantile(v, 0.05))
    summary["lod_5th_percentile_true_values"] = lod
    (DOCS_DIR / "results").mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "results" / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    header = (
        "# Assay model for the virtual patients (arbitrary model units).\n"
        "# lod: lower limit of detection per cytokine = 5th percentile of the true panel values over\n"
        "# readings 51..612 among pilot runs alive at 72 h (scripts/analyze_pilot.py). Readings below\n"
        "# it are reported as censored, never as zero.\n"
    )
    body = yaml.safe_dump(
        {"noise_cv": 0.25, "missing_fraction": 0.0, "panel": DEFAULT_PANEL,
         "lod": {k: round(v, 4) for k, v in lod.items()}},
        sort_keys=False,
    )  # fmt: skip
    (CONFIG_DIR / "measurement.yaml").write_text(header + body)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("binned",)}, indent=1))


if __name__ == "__main__":
    main()
