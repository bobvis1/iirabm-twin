"""Choose the calibration settings (tau, run-level tau, variance floor, method) on the
DEVELOPMENT cohort only. Test patients are refused (twin.evaluate.assert_development_only).

Selection rules (the second replaced the first after the first run; both are reported):

* Rule v1 (fixed before the first run, PLAN.md 4.1): the configuration whose pooled 80% intervals
  (parameters and forecasts) cover closest to 80%. It selected tau = 0.3, but pooling hid that
  inj_number, the one parameter the readings identify, was covered only 38% of the time at its 80%
  interval, while the unidentified parameters (posterior close to the prior) were over-covered and
  pulled the pooled number up to 0.79. Pooled coverage is the wrong criterion.
* Rule v2 (used): feasible = no parameter's 80% interval under-covers by more than 0.15 (about two
  standard errors with 40 patients) and the smallest effective sample size is at least 20 (the
  library is not too coarse for any patient). Among feasible configurations, the lowest Brier
  score of the 28-day mortality forecast. Under-coverage is what overconfidence looks like;
  over-coverage of weakly identified parameters is expected and harmless.

tau = 1 is always reported alongside, to show how much tempering mattered. With 40 patients and
about 90 configurations, coverage estimates are noisy (standard error about 0.06) and the chosen
configuration is optimistic; Phase 5 scores it once on the test cohort.

    uv run python scripts/tune_calibration.py             # compute and write the JSON
    uv run python scripts/tune_calibration.py --select    # re-apply the rules to the saved JSON
"""

from __future__ import annotations

import argparse
import itertools
import json

from twin.calibrate import CalibrationConfig, compute_terms, posterior_from_terms, prior_only
from twin.config import DOCS_DIR
from twin.evaluate import assert_development_only, score_patient, summarise
from twin.forecast import ForecastEngine
from twin.librarydata import load_or_build
from twin.observe import load_observations
from twin.truth import list_patients, read_truth

TAUS = [1.0, 0.3, 0.1, 0.05, 0.02, 0.01]
TAU_RUNS = [1.0, 0.3, 0.1, 0.03, 0.01]
VAR_FLOORS = [0.25, 1.0]


def configs() -> list[CalibrationConfig]:
    out = []
    for vf, tau in itertools.product(VAR_FLOORS, TAUS):
        out.append(CalibrationConfig("synthetic", "theta", tau, 1.0, vf))
        for tr in TAU_RUNS:
            out.append(CalibrationConfig("synthetic", "run", tau, tr, vf))
    out += [CalibrationConfig("mixture", "run", tau, tau, 0.25) for tau in TAUS]
    return out


def name(c: CalibrationConfig) -> str:
    return (
        f"{c.theta_method}/{c.forecast_level}/tau={c.tau}/tau_run={c.tau_run}/floor={c.var_floor}"
    )


MAX_UNDER = 0.15
MIN_ESS = 20


def select(rows: list[dict]) -> tuple[dict, dict]:
    v1 = min(rows, key=lambda r: (round(r["coverage80_gap"], 3), r["summary"]["brier"]))
    feasible = []
    for r in rows:
        cov = r["summary"]["theta_coverage_by_param"]
        r["max_under80"] = max(max(0.0, 0.8 - cov[p]["0.8"]) for p in cov)
        if r["max_under80"] <= MAX_UNDER and r["summary"]["ess_theta_min"] >= MIN_ESS:
            feasible.append(r)
    return v1, min(feasible, key=lambda r: r["summary"]["brier"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="re-apply the rules to the saved JSON")
    if ap.parse_args().select:
        path = DOCS_DIR / "results" / "calibration_tuning.json"
        out = json.loads(path.read_text())
        v1, v2 = select(out["all"])
        out["rule_v1_choice"], out["best"] = v1["name"], v2
        path.write_text(json.dumps(out, indent=1, default=str))
        print(
            "v1 chose",
            v1["name"],
            "| v2 chose",
            v2["name"],
            "max under-coverage",
            round(v2["max_under80"], 3),
        )
        return
    ids = list_patients("development")
    assert_development_only(ids)
    lib = load_or_build()
    fe = ForecastEngine(lib)
    truths = {p: read_truth(p) for p in ids}
    obs = {p: load_observations(p) for p in ids}
    cfgs = configs()
    terms = {}  # (patient, var_floor) -> Terms, computed once
    for p in ids:
        for vf in VAR_FLOORS:
            terms[(p, vf)] = compute_terms(obs[p], lib, CalibrationConfig(var_floor=vf))
    rows = []
    for c in cfgs:
        scores = []
        for p in ids:
            post = posterior_from_terms(terms[(p, c.var_floor)], lib, c)
            scores.append(score_patient(truths[p], post, fe.forecast(post), lib))
        s = summarise(scores)
        dev80 = abs(s["theta_coverage"][0.8] - 0.8) + abs(s["forecast_coverage"][0.8] - 0.8)
        rows.append({"name": name(c), "config": c.__dict__, "summary": s, "coverage80_gap": dev80})
        print(f"{name(c):70s} theta80={s['theta_coverage'][0.8]:.2f} fc80={s['forecast_coverage'][0.8]:.2f} "
              f"brier={s['brier']:.3f} ess_min={s['ess_theta_min']:.0f} gap={dev80:.2f}", flush=True)  # fmt: skip
    scores = [
        score_patient(truths[p], prior_only(lib), fe.forecast(prior_only(lib)), lib) for p in ids
    ]
    baseline = summarise(scores)
    v1, best = select(rows)
    out = {"n_patients": len(ids), "prior_only": baseline, "rule_v1_choice": v1["name"],
           "best": best, "all": rows}  # fmt: skip
    (DOCS_DIR / "results" / "calibration_tuning.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    print("\nBEST:", best["name"], f"gap={best['coverage80_gap']:.3f}")
    print(
        "PRIOR ONLY: brier",
        round(baseline["brier"], 3),
        "forecast80",
        round(baseline["forecast_coverage"][0.8], 2),
    )


if __name__ == "__main__":
    main()
