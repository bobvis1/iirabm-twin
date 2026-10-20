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
"""


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


def main() -> None:
    text = README.read_text()
    text = block("cost", cost_block(), text)
    if (DOCS_DIR / "results" / "validation.json").exists():
        text = block("validation", validation_block(), text)
    README.write_text(text)


if __name__ == "__main__":
    main()
