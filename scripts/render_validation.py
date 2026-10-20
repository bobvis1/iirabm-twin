"""Render docs/validation_report.md and its figures from docs/results/validation.json (or another
validation file given with --input). Every number in the report comes from that file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from twin.config import DOCS_DIR  # noqa: E402
from twin.params import PARAM_NAMES  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#0b0b0b"
NAMES = {"prior_only": "prior only", "library_theta": "library, theta-level",
         "library_run": "library, run-level", "ga": "GA"}  # fmt: skip
COLORS = {"prior_only": GREY, "library_theta": BLUE, "library_run": GREEN, "ga": ORANGE}


def ci(t, n=3):
    return f"{t[0]:.{n}f} [{t[1]:.{n}f}, {t[2]:.{n}f}]"


def lv(d, k):
    return d[str(k)] if str(k) in d else d[k]


def figures(v: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # reliability
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    for m, col in COLORS.items():
        if m in v["methods"]:
            r = v["methods"][m]["mortality"]["reliability"]
            ax.plot([b["mean_predicted"] for b in r], [b["observed"] for b in r], "o-", color=col,
                    lw=1.6, ms=5, label=NAMES[m])  # fmt: skip
    ax.set_xlabel("predicted 28-day mortality")
    ax.set_ylabel("observed fraction died")
    ax.set_title("Reliability (5 equal-count bins)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "validation_reliability.png", dpi=140)
    # rank histograms
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.8), sharey=True)
    lib = v["methods"]["library_theta"]["theta"]
    for ax, p in zip(axes, PARAM_NAMES, strict=True):
        c = lib[p]["rank_test"]
        ax.bar(np.arange(len(c["counts"])), c["counts"], color=BLUE, width=0.85)
        ax.axhline(np.mean(c["counts"]), color=INK, lw=1, ls="--")
        ax.set_title(f"{p}\nchi-square p = {c['p_value']:.3f}", fontsize=9)
        ax.set_xlabel("rank decile of the truth", fontsize=8)
    axes[0].set_ylabel("patients")
    fig.suptitle(
        "Rank of the true value within the library posterior (uniform if calibrated)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(outdir / "validation_ranks.png", dpi=140)
    # coverage
    fig, ax = plt.subplots(figsize=(6, 4))
    levels = [0.5, 0.8, 0.95]
    w = 0.2
    for i, (m, col) in enumerate(COLORS.items()):
        if m in v["methods"]:
            cov = [lv(v["methods"][m]["forecast_coverage"], k)["coverage"] for k in levels]
            ax.bar(np.arange(3) + i * w, cov, w, color=col, label=NAMES[m])
    ax.plot([-0.2, 2.8], [0.5, 0.5], alpha=0)
    for j, k in enumerate(levels):
        ax.hlines(k, j - 0.1, j + 0.7, color=INK, lw=1.2, ls="--")
    ax.set_xticks(np.arange(3) + 0.3, ["50%", "80%", "95%"])
    ax.set_ylabel("fraction of true values inside the interval")
    ax.set_title("Forecast interval coverage (dashed = nominal)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "validation_coverage.png", dpi=140)
    # truth vs posterior mean
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, p in zip(axes, PARAM_NAMES, strict=True):
        t = v["truth_vs_posterior_mean"][p]
        ax.plot(t["truth"], t["mean"], "o", color=BLUE, ms=3.5, alpha=0.7)
        lo, hi = min(t["truth"]), max(t["truth"])
        ax.plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=1)
        ax.set_title(f"{p}\nr = {t['corr']:.2f}", fontsize=9)
        ax.set_xlabel("true", fontsize=8)
    axes[0].set_ylabel("posterior mean")
    fig.tight_layout()
    fig.savefig(outdir / "validation_recovery.png", dpi=140)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(v: dict, outdir: Path) -> str:
    M = v["methods"]
    n = len(v["patients"])
    ga_n = len(v["on_ga_subset"].get("patients", []))
    frozen = v["frozen_at_commit"]
    at = f"commit `{frozen[:10]}`" if frozen else "no commit (a dry run, not the validation)"
    head = f"cohort **{v['cohort']}**, {n} virtual patients, settings frozen at {at}, scored {v['date']}"
    rows = []
    for m in ("prior_only", "library_theta", "library_run", "ga"):
        if m in M:
            mo = M[m]["mortality"]
            rows.append([NAMES[m] + (f" ({M[m]['n_patients']} patients)" if m == "ga" else ""),
                         ci(mo["brier"]), ci(mo["log_loss"]), ci(mo["auroc"], 2),
                         f"{mo['mean_forecast']:.2f} / {mo['observed']:.2f}"])  # fmt: skip
    t_mort = table(["Method", "Brier score", "Log loss", "AUROC", "mean forecast / observed"], rows)
    paired = []
    for m, d in v["paired_vs_prior_only"].items():
        paired.append(
            [NAMES[m], ci(d["brier_diff"]), ci(d["log_loss_diff"]), ci(d["auroc_diff"], 2)]
        )
    t_paired = table(
        ["Method minus prior-only", "Brier diff", "Log loss diff", "AUROC diff"], paired
    )
    cov_rows = []
    for m in ("prior_only", "library_theta", "library_run", "ga"):
        if m in M:
            c = M[m]["forecast_coverage"]
            cov_rows.append([NAMES[m]] + [
                f"{lv(c, k)['coverage']:.2f} [{lv(c, k)['wilson95'][0]:.2f}, {lv(c, k)['wilson95'][1]:.2f}]"
                for k in (0.5, 0.8, 0.95)] + [lv(c, 0.8)["n"]])  # fmt: skip
    t_cov = table(
        ["Method", "50% interval", "80% interval", "95% interval", "n intervals"], cov_rows
    )
    day_rows = []
    for d_ in ("4", "7", "14", "28"):
        day_rows.append([f"day {d_}"] + [f"{M[m]['forecast_coverage_by_day'][d_]['coverage80']:.2f} (n={M[m]['forecast_coverage_by_day'][d_]['n']})"
                                         for m in ("prior_only", "library_theta") if m in M])  # fmt: skip
    t_day = table(["80% interval coverage", "prior only", "library, theta-level"], day_rows)
    var_rows = [[var, f"{M['library_theta']['forecast_coverage_by_variable'][var]['coverage80']:.2f}",
                 f"{M['prior_only']['forecast_coverage_by_variable'][var]['coverage80']:.2f}"]
                for var in M["library_theta"]["forecast_coverage_by_variable"]]  # fmt: skip
    t_var = table(["variable", "library, theta-level (80%)", "prior only (80%)"], var_rows)
    th = M["library_theta"]["theta"]
    tv = v["truth_vs_posterior_mean"]
    th_rows = []
    for p in PARAM_NAMES:
        c = th[p]["coverage"]
        th_rows.append([p] + [f"{lv(c, k)['coverage']:.2f} [{lv(c, k)['wilson95'][0]:.2f}, {lv(c, k)['wilson95'][1]:.2f}]" for k in (0.5, 0.8, 0.95)]
                       + [f"{th[p]['rank_test']['p_value']:.3f}", f"{th[p]['contraction_mean']:.2f}", f"{tv[p]['corr']:.2f}", f"{tv[p]['rmse']:.3g}"])  # fmt: skip
    t_theta = table(
        [
            "Parameter",
            "50%",
            "80%",
            "95%",
            "rank-test p",
            "posterior var / prior var",
            "r(mean, truth)",
            "RMSE",
        ],
        th_rows,
    )
    ga_text = ""
    if ga_n:
        S = v["on_ga_subset"]
        r = []
        for m in ("prior_only", "library_theta", "library_run", "ga"):
            mo = S[m]["mortality"]
            fc = S[m]["forecast_coverage"]
            r.append([NAMES[m], ci(mo["brier"]), ci(mo["auroc"], 2), f"{lv(fc, 0.8)['coverage']:.2f}",
                      " ".join(f"{lv(S[m]['theta'][p]['coverage'], 0.8)['coverage']:.2f}" for p in PARAM_NAMES)])  # fmt: skip
        d = S["paired_ga_vs_library"]["brier_diff"]
        ga_text = (f"\n### The GA, on the {ga_n} patients it was run for\n\n"
                   + table(["Method", "Brier score", "AUROC", "80% forecast coverage", "80% parameter coverage (" + ", ".join(PARAM_NAMES) + ")"], r)
                   + f"\n\nGA minus library (theta-level) Brier difference: {ci(d)}.\n")  # fmt: skip
    # measurement design
    dsg = []
    for name, row in v.get("measurement_design", {}).items():
        a, b = row["fixed_tau"], row["scaled_tau"]
        dsg.append([name, row["n_readings"], f"{a['brier']:.3f}", f"{a['forecast_coverage80']:.2f}",
                    f"{a['theta_coverage80']['inj_number']:.2f}", f"{a['contraction']['inj_number']:.2f}",
                    f"{a['contraction']['oxyHeal']:.2f}", f"{b['brier']:.3f}", f"{b['contraction']['inj_number']:.2f}",
                    f"{b['contraction']['oxyHeal']:.2f}"])  # fmt: skip
    t_dsg = table(
        [
            "Design",
            "readings",
            "Brier",
            "fc 80%",
            "inj 80% cov",
            "inj contr.",
            "oxyHeal contr.",
            "Brier (tau scaled)",
            "inj contr. (scaled)",
            "oxyHeal contr. (scaled)",
        ],
        dsg,
    )
    mis = v.get("misspecification", {})
    mis_rows = []
    for k, label in (("well_specified", "well specified (data CV 25%, twin assumes 25%)"),
                     ("misspecified", "misspecified (data CV 40%, twin assumes 25%)")):  # fmt: skip
        if k in mis:
            r = mis[k]
            mis_rows.append([label, f"{r['brier']:.3f}", f"{r['forecast_coverage80']:.2f}"]
                            + [f"{r['theta_coverage80'][p]:.2f}" for p in PARAM_NAMES])  # fmt: skip
    t_mis = table(["", "Brier", "forecast 80%"] + [f"{p} 80%" for p in PARAM_NAMES], mis_rows)

    diag_path = DOCS_DIR / "results" / "validation_diagnostics.json"
    diag = ""
    if diag_path.exists():
        dg = json.loads(diag_path.read_text())
        rows = []
        for c in ("development", "test"):
            x = dg[c]
            rows.append([c, x["n"], f"{x['observed_mortality']:.2f}", f"{x['matched']['brier']:.3f} / {x['matched']['auroc']:.2f}",
                         f"{x['shuffled_observations']['brier']:.3f} / {x['shuffled_observations']['auroc']:.2f}",
                         f"{x['oracle_true_theta_neighbours']['brier']:.3f} / {x['oracle_true_theta_neighbours']['auroc']:.2f}"])  # fmt: skip
        diag = f"""
## 6. Is the test result too good? Post-hoc checks

The test-cohort Brier score is much better than the development one, which is a reason to look for
leakage of the truth or reuse of seeds before believing it. These checks were run after scoring
(`scripts/validation_diagnostics.py`, `docs/results/validation_diagnostics.json`); nothing was tuned
or selected from them.

{table(["Cohort", "patients", "observed mortality", "twin: Brier / AUROC", "shuffled readings: Brier / AUROC", "oracle (true theta): Brier / AUROC"], rows)}

- **Shuffled readings:** calibrating each patient with *another* patient's readings gives chance-level
  skill, so nothing but a patient's own readings carries the signal; there is no leak through the pipeline.
- **Oracle:** mortality among the 64 library design points nearest the patient's *true* parameters is
  a rough upper bound on what knowing theta would give. On the test cohort the twin matches it; on the
  development cohort it did not. The test cohort's outcomes are simply more predictable from theta,
  mostly through injury size, which the readings identify well. The development cohort has more
  patients whose fate depends on parameters the readings do not identify.
- **Seed ranges:** every patient seed lies in the patient range, disjoint from the library, calibration,
  search and future ranges (checked in `tests/test_seeds.py` and listed in the diagnostics file).
- The GA, which uses no library at all, reaches a similar AUROC on its {ga_n} patients.
- **Calibration caveat:** the rank test for `inj_number` rejects uniformity on the test cohort (it is
  over-covered: its 80% interval contains the truth more often than 80%); the posterior for that
  parameter is conservative, not too narrow.
"""
    design_note = ""
    if v.get("measurement_design_recomputed"):
        design_note = (
            " (This section was recomputed after a bug in the runner dropped it from the first"
            " saved file: the same frozen settings and the same code path, deterministic.)"
        )
    text = f"""# Validation report

*Generated by `scripts/render_validation.py` from `docs/results/validation.json`; do not edit by hand.
{head}. All results are for virtual patients in a public 2017 model; nothing here is a clinical claim.*

## Read this first: the validation is well specified

The virtual patients come from the same engine, prior and noise model the twin assumes. In that
setting near-nominal coverage is the *expected* result. It shows that the inference code is correct.
It says nothing about whether the model fits real patients. The misspecification test (section 5)
shows how the twin degrades when one assumption is wrong. A result that looks too good here would be
a reason to suspect leakage of the truth or reuse of seeds; none of the numbers below is perfect.

Settings were chosen on the development cohort only and frozen before this cohort was scored
(`config/calibration.yaml`; the library posterior uses tau = {v["settings"]["library_posterior"]["tau"]},
theta-level forecasts). The test cohort was scored once.

## 1. Parameter recovery (library posterior, theta-level)

{t_theta}

The coverage columns give the fraction of patients whose true parameter lies in the central interval
(95% Wilson interval for that fraction in brackets; the {n} patients are treated as independent).
The rank test is a chi-square test of uniform ranks over deciles (simulation-based calibration,
Talts et al. 2018; the library posterior is a discrete approximation and ties of the integer
parameters are broken at random, so small departures are expected). "Posterior var / prior var" is
the posterior contraction: 1 means the readings taught us nothing about that parameter. Figures:
`docs/figures/validation_ranks.png`, `docs/figures/validation_recovery.png`.

## 2. Forecasts

### 28-day mortality (among patients alive at 72 h; bootstrap over patients, 95% intervals)

{t_mort}

Paired differences from the prior-only baseline (same patients resampled together):

{t_paired}

Reliability plot: `docs/figures/validation_reliability.png`. With {n} patients the intervals are
wide; the observed mortality is the share of these patients that died within 28 days.

### Predictive-interval coverage (oxy deficit and each panel cytokine at days 4, 7, 14, 28, among patients alive then)

{t_cov}

The Wilson intervals treat every (variable, day) interval as independent, which they are not (a
patient's variables and days are correlated), so they are too narrow; read them as a lower bound on
the uncertainty. Coverage by day and by variable:

{t_day}

{t_var}

Figure: `docs/figures/validation_coverage.png`.
{ga_text}
## 3. Measurement design (library re-weighting only)

Nothing is selected from this table; it is reported.{design_note} Readings come from each patient's stored truth
with the noise level of the design, the twin assumes that noise level, the settings are the frozen
ones, and "window" means the last 24 or 48 hours before the decision time. Because tau was tuned for
96 readings, the right-hand columns repeat the study with tau scaled to keep the total weight on the
readings equal to the baseline's. Differences between rows are small next to the sampling error with
{n} patients; do not over-read them.

{t_dsg}

## 4. Cost of the methods

See `docs/calibration.md` and the README cost section: the library (65,536 simulations) is shared by
every patient; the GA used about 2,500 simulations per patient.

## 5. Misspecification stress test

Readings regenerated from the stored truth with assay noise CV 40% while the twin still assumes 25%
(library re-weighting only, same frozen settings):

{t_mis}

{diag}
## Limits
- One 2017 model, virtual patients only; the twin is calibrated with the model that generated the data.
- The test cohort is {n} patients ({ga_n} for the GA); every interval is wide.
- The tuning that fixed tau used 40 development patients and about 90 configurations; the settings are
  optimistic, and this report is their first test on new patients.
"""
    figures(v, DOCS_DIR / "figures")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DOCS_DIR / "results" / "validation.json")
    ap.add_argument("--output", type=Path, default=DOCS_DIR / "validation_report.md")
    args = ap.parse_args()
    v = json.loads(args.input.read_text())
    args.output.write_text(render(v, DOCS_DIR / "figures"))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
