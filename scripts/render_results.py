"""Fill the generated blocks of README.md from the small result files under docs/results/.

Blocks are delimited by <!-- NAME:start --> and <!-- NAME:end -->. Numbers are never hand-copied.
Run:  uv run python scripts/render_results.py
"""

from __future__ import annotations

import json
import re

from twin.config import DOCS_DIR, ROOT

README = ROOT / "README.md"


def block(name: str, body: str, text: str) -> str:
    pat = re.compile(rf"(<!-- {name}:start -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pat.search(text):
        text = text.rstrip() + f"\n\n<!-- {name}:start -->\n<!-- {name}:end -->\n"
    return pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text)


def actual_line() -> str:
    path = DOCS_DIR / "results" / "actual_cost.json"
    if not path.exists():
        return ""
    a = json.loads(path.read_text())
    return (f"\nActual cost from Cost Explorer, whole account since {a['since']}, as of {a['as_of']}: "
            f"**${a['account_total_usd']:.2f}**. Cost Explorer lags about a day, so this is incomplete "
            f"until it is re-run (`scripts/cost_report.py`).\n")  # fmt: skip


def cost_block() -> str:
    t = json.loads((DOCS_DIR / "results" / "sagemaker_timing.json").read_text())
    lib = json.loads((DOCS_DIR / "results" / "library_summary.json").read_text())
    rows = []
    for it, r in t["instances"].items():
        b = r["best"]
        rows.append(
            f"| `{it}` | {b['workers']} of {r['cpu']['cpu_count']} | {b['steps_per_s']:,.0f} | "
            f"${t['price_per_hour_usd'][it]:.3f} | ${r['usd_per_million_steps']:.3f} |"
        )
    job = lib["job"]
    return f"""### Cost

Engine throughput on SageMaker Processing instances (`docs/results/sagemaker_timing.json`):

| Instance | Workers | Steps per second | $ per hour | $ per million steps |
|---|---|---|---|---|
{chr(10).join(rows)}

{t["source_of_prices"]}. The engine is cache-bound: one worker per vCPU is slower overall than one
per physical core.

Simulation library: {lib["n_design_points"]:,} design points x {lib["seeds_per_point"]} seeds =
{lib["runs"]:,} runs ({lib["total_steps"] / 1e6:,.0f} million steps) on {job["instances"]} x
`ml.c8i.8xlarge`, {job["wall_minutes"]:.1f} minutes, about **${job["estimated_cost_usd"]:.2f}**
(estimate from instance-hours; confirm in Cost Explorer, which lags about a day).
{actual_line()}"""


def validation_block() -> str:
    v = json.loads((DOCS_DIR / "results" / "validation.json").read_text())
    m = v["methods"]
    mo = m["library_theta"]["mortality"]
    po = m["prior_only"]["mortality"]
    fc = m["library_theta"]["forecast_coverage"]
    return f"""### Validation (test cohort, {len(v["patients"])} virtual patients, settings frozen before scoring)

In this model, with virtual patients drawn from the same model the twin assumes (a well-specified
setting, so near-nominal coverage is the expected result; see `docs/validation_report.md`):

| 28-day mortality forecast | Brier score | AUROC |
|---|---|---|
| prior only (no patient data) | {po["brier"][0]:.3f} | {po["auroc"][0]:.2f} |
| library twin, theta-level | {mo["brier"][0]:.3f} [{mo["brier"][1]:.3f}, {mo["brier"][2]:.3f}] | {mo["auroc"][0]:.2f} |

Predictive intervals for oxy deficit and the cytokine panel contain the truth {fc["0.5"]["coverage"]:.0%}
(50% interval), {fc["0.8"]["coverage"]:.0%} (80%) and {fc["0.95"]["coverage"]:.0%} (95%) of the time.
"""


def treatment_block() -> str:
    t = json.loads((DOCS_DIR / "results" / "treatment.json").read_text())
    s = t["schedules"]
    p = t["patients"]

    def d(x):
        return f"{x['diff']:+.3f} (95% interval {x['ci95'][0]:+.3f} to {x['ci95'][1]:+.3f})"

    return f"""### Treatment search (test patients, in this model)

**n is small and this is one 2017 model, not a clinical result.** For {len(p)} test virtual patients
chosen by a rule fixed beforehand, a schedule of four mediator interventions (they only *resemble*
approved drugs) was searched on each patient's twin and scored on {t["futures"]} fresh futures of the true
patient (fraction dead by day 28; `docs/treatment_report.md`). Pooled difference in mortality over the
{len(p)} patients: twin schedule minus no treatment {d(s["twin_minus_none"])}; twin schedule minus the best
one-size-fits-all schedule {d(s["twin_minus_one_size"])}; oracle (true parameters) minus no treatment
{d(s["oracle_minus_none"])}. Only two of the five patients had a fate treatment could change; the
twin-personalised schedule did no better than one-size-fits-all.
"""


def load(name: str) -> dict:
    return json.loads((DOCS_DIR / "results" / name).read_text())


def headline_block() -> str:
    eng = ROOT / "engine" / "docs" / "results"  # measured in the engine fork
    fx = json.loads((eng / "engine_fixes.json").read_text())
    d3 = json.loads((eng / "defect3.json").read_text())
    lib, v = load("library_summary.json"), load("validation.json")
    tr, tj = load("treatment.json"), load("treatment_jobs.json")
    gd, gt = load("ga_dev_job.json"), load("ga_test_job.json")
    m = v["methods"]
    mo, po = m["library_theta"]["mortality"], m["prior_only"]["mortality"]
    d = fx["thetas"]["borderline"]
    spend = (lib["job"]["estimated_cost_usd"] + gd["estimated_cost_usd"] + gt["estimated_cost_usd"]
             + sum(x["estimated_cost_usd"] for x in tj.values() if isinstance(x, dict) and "estimated_cost_usd" in x))  # fmt: skip
    return f"""- **Engine.** Five defects in the 2017 code were confirmed with evidence and fixed (or documented).
  One changes results: at the model's default parameters, mortality at 28 days is
  {d["fixes_off"]["mortality"]:.0%} with the original loop and {d["fixes_on"]["mortality"]:.0%} with the fix (200 seeds each). Running
  many simulations in one process (the original GA's pattern) changed the outcome far more:
  {d3["legacy_one_process"]["deaths"]} of {d3["legacy_one_process"]["n"]} died in one process against {d3["fresh_process"]["deaths"]} of {d3["fresh_process"]["n"]} in fresh processes.
- **Twin.** On {len(v["patients"])} held-out virtual patients, the 28-day mortality forecast has Brier score
  {mo["brier"][0]:.3f} against {po["brier"][0]:.3f} with no patient data (AUROC {mo["auroc"][0]:.2f}); 80% predictive intervals
  contain the truth {lv_cov(m["library_theta"], 0.8):.0%} of the time. Only the injury size is clearly identified by
  readings up to 72 h.
- **Treatment.** A schedule searched on each patient's twin lowered mortality for the two (of five) patients
  whose fate was open, but **did no better than a single schedule for everyone**
  (paired difference {tr["schedules"]["twin_minus_one_size"]["diff"]:+.3f}, interval {tr["schedules"]["twin_minus_one_size"]["ci95"][0]:+.3f} to {tr["schedules"]["twin_minus_one_size"]["ci95"][1]:+.3f}).
- **Cost.** About ${spend:.0f} of SageMaker jobs were itemised here (smoke and benchmark jobs, about $2, are
  not), on a $50 alarm.
"""


def lv_cov(method: dict, level: float) -> float:
    c = method["forecast_coverage"]
    return (c[str(level)] if str(level) in c else c[level])["coverage"]


def main() -> None:
    text = README.read_text()
    text = block("headline", headline_block(), text)
    text = block("cost", cost_block(), text)
    if (DOCS_DIR / "results" / "validation.json").exists():
        text = block("validation", validation_block(), text)
    if (DOCS_DIR / "results" / "treatment.json").exists():
        text = block("treatment", treatment_block(), text)
    README.write_text(text)


if __name__ == "__main__":
    main()
