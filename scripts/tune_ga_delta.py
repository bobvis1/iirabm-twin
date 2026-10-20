"""Choose the GA's likelihood window Delta on the DEVELOPMENT patients, and compare the GA with the
library posterior. Test patients are refused.

    uv run python scripts/tune_ga_delta.py data/ga-dev/results/<job>

Delta and kappa rule (the same as scripts/tune_calibration.py rule v2): feasible = no parameter's
80% interval under-covers by more than 0.15; among feasible values, the lowest Brier score of the
28-day mortality forecast; if none is feasible, the pair with the smallest worst under-coverage.
kappa <= 1 tempers the GA's likelihood weights (exp(kappa x log-likelihood difference)); it was
added after the first pass with kappa = 1 under-covered for every Delta.
With 20 patients the coverage estimates are noisy (standard error about 0.09).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from twin.calibrate import CalibrationConfig, compute_terms, ga_posterior, posterior_from_terms
from twin.config import CONFIG_DIR, DOCS_DIR
from twin.evaluate import assert_development_only, prior_variance, score_patient, summarise
from twin.forecast import ForecastEngine, theta_summary
from twin.jobs.calibrate_ga import load_fit
from twin.librarydata import load_or_build
from twin.observe import load_observations
from twin.params import PARAM_NAMES
from twin.truth import read_truth

DELTAS = [3, 5, 8, 12, 20]
KAPPAS = [1.0, 0.5, 0.25, 0.1, 0.05]
MAX_UNDER = 0.15


def main(results: Path) -> None:
    ids = sorted(p.stem for p in results.glob("dev-*.npz"))
    assert_development_only(ids)
    cfg_all = yaml.safe_load((CONFIG_DIR / "calibration.yaml").read_text())
    cal = CalibrationConfig(**cfg_all["library_posterior"])
    lib = load_or_build()
    pv = prior_variance(lib)
    fe_lib = ForecastEngine(lib)
    fits = {p: load_fit(results / f"{p}.npz", p) for p in ids}
    truths = {p: read_truth(p) for p in ids}
    rows = []
    for delta, kappa in itertools.product(DELTAS, KAPPAS):
        scores = []
        for p in ids:
            flib, post = ga_posterior(fits[p], delta, cal, kappa)
            scores.append(
                score_patient(truths[p], post, ForecastEngine(flib).forecast(post), flib, pv)
            )
        s = summarise(scores)
        under = max(max(0.0, 0.8 - c[0.8]) for c in s["theta_coverage_by_param"].values())
        rows.append({"delta": delta, "kappa": kappa, "summary": s, "max_under80": under})
        print(f"delta={delta:>3} kappa={kappa:<5}: theta80 by param "
              + " ".join(f"{s['theta_coverage_by_param'][p][0.8]:.2f}" for p in PARAM_NAMES)
              + f" | fc80 {s['forecast_coverage'][0.8]:.2f} | brier {s['brier']:.3f} | max under {under:.2f}")  # fmt: skip
    feasible = [r for r in rows if r["max_under80"] <= MAX_UNDER]
    best = (
        min(feasible, key=lambda r: r["summary"]["brier"])
        if feasible
        else min(rows, key=lambda r: r["max_under80"])
    )

    # Library posterior (the chosen config) on the same patients, for the comparison.
    lib_scores, agree = [], []
    for p in ids:
        obs = load_observations(p)
        post = posterior_from_terms(compute_terms(obs, lib, cal), lib, cal)
        lib_fc = fe_lib.forecast(post)
        lib_scores.append(score_patient(truths[p], post, lib_fc, lib))
        flib, gpost = ga_posterior(fits[p], best["delta"], cal, best["kappa"])
        g_fc = ForecastEngine(flib).forecast(gpost)
        lt, gt = theta_summary(post, lib), theta_summary(gpost, flib)
        overlap = {}
        for n in PARAM_NAMES:
            (a0, a1), (b0, b1) = lt[n]["intervals"][0.8], gt[n]["intervals"][0.8]
            inter = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            overlap[n] = float(inter / union) if union > 0 else 1.0
        agree.append({"patient": p, "mortality_library": lib_fc.mortality, "mortality_ga": g_fc.mortality,
                      "interval80_overlap": overlap, "n_sims_ga": fits[p].n_sims,
                      "generations": fits[p].generations, "stopped_early": fits[p].stopped_early})  # fmt: skip
    m_lib = np.array([a["mortality_library"] for a in agree])
    m_ga = np.array([a["mortality_ga"] for a in agree])
    out = {
        "patients": ids, "delta_rule": __doc__.split("Delta and kappa rule")[1].split("\n\n")[0].strip(),
        "all_deltas": rows, "best_delta": best["delta"], "best_kappa": best["kappa"], "ga_at_best": best["summary"],
        "library_on_same_patients": summarise(lib_scores),
        "agreement": agree,
        "mortality_forecast_correlation": float(np.corrcoef(m_lib, m_ga)[0, 1]),
        "mean_abs_mortality_difference": float(np.mean(np.abs(m_lib - m_ga))),
        "mean_interval80_overlap": {n: float(np.mean([a["interval80_overlap"][n] for a in agree])) for n in PARAM_NAMES},
        "sims_per_patient_ga_median": float(np.median([a["n_sims_ga"] for a in agree])),
        "sims_library_total": int(lib.n * lib.k),
    }  # fmt: skip
    (DOCS_DIR / "results" / "ga_tuning.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nbest delta", best["delta"], "kappa", best["kappa"], "| GA brier", round(best["summary"]["brier"], 3),
          "| library brier (same patients)", round(out["library_on_same_patients"]["brier"], 3))  # fmt: skip
    print("mortality correlation", round(out["mortality_forecast_correlation"], 2), "mean |diff|", round(out["mean_abs_mortality_difference"], 3))  # fmt: skip
    print(
        "mean overlap of 80% intervals",
        {k: round(v, 2) for k, v in out["mean_interval80_overlap"].items()},
    )
    print("GA sims per patient (median)", out["sims_per_patient_ga_median"], "| library sims (shared by all patients)", out["sims_library_total"])  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]))
