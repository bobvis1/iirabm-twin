"""docs/figures/example_patient.png: one virtual patient's true trajectories, its noisy readings and
the decision time. Uses the truth, so it is a documentation figure (never calibration code)."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from twin.config import DECISION_STEP, DOCS_DIR  # noqa: E402
from twin.observe import load_observations  # noqa: E402
from twin.truth import read_truth  # noqa: E402

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def main(patient_id: str = "dev-001") -> None:
    truth = read_truth(patient_id)
    obs = load_observations(patient_id)
    steps = truth.record_steps
    hours = steps * 7 / 60  # one step is about 7 minutes
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for ax, c in zip(axes.ravel(), obs.panel, strict=True):
        ax.plot(hours, truth.series[c], color=BLUE, lw=1.8, label="true total (hidden)")
        j = obs.panel.index(c)
        h_obs = obs.steps * 7 / 60
        ok = ~(obs.censored[:, j] | obs.missing[:, j])
        ax.plot(h_obs[ok], obs.values[ok, j], "o", color=ORANGE, ms=5, label="reading")
        if obs.censored[:, j].any():
            ax.plot(h_obs[obs.censored[:, j]], np.full(obs.censored[:, j].sum(), obs.lod[j]), "v",
                    mfc="none", mec=ORANGE, ms=6, label="censored (below detection)")  # fmt: skip
        ax.axvline(DECISION_STEP * 7 / 60, color=MUTED, ls="--", lw=1)
        ax.axhline(obs.lod[j], color=MUTED, lw=0.6, alpha=0.5)
        ax.set_title(c, fontsize=10, color=INK)
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("hours since injury", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    th = truth.theta
    fig.suptitle(
        f"Virtual patient {patient_id} (in this model): hidden truth, noisy 6-hourly readings, "
        f"decision time T_d = 72 h (dashed)\n"
        f"true theta: injury {th.inj_number}, oxyHeal {th.oxyHeal:.3f}, spread {th.infectSpread}, "
        f"recur {th.numRecurInj}, repeat {th.numInfectRepeat}; "
        f"{'dies at step ' + str(truth.death_step) if truth.died else 'survives 28 days'}",
        fontsize=10,
    )
    fig.tight_layout()
    (DOCS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(DOCS_DIR / "figures" / "example_patient.png", dpi=140)


if __name__ == "__main__":
    main(*sys.argv[1:])
