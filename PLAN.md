# PLAN.md: A digital-twin loop on the IIRABM (measure, model, modulate)

## Goal

Show, in simulation where the true answer is known, the loop that the ReSCUED project runs on real
patients: measure cytokines every 6 hours, calibrate a mechanistic immune model to one patient
with honest uncertainty, check the twin's forecasts against the truth, then search for a
treatment schedule on the twin and score it on the true patient. Run the simulations on Amazon
SageMaker in a locked-down AWS account. Budget: about 30 hours of the user's time and $50 of AWS.

**Who does what.** Tasks marked **[you]** need the user: account actions, decisions and reviews.
Everything else is for Claude Code (Sonnet). Claude stops at the end of every phase, reports, and
waits for the user before starting the next one.

## Deliverables

1. **Engine fork** (`github.com/bobvis1/iirabm`, GPL-2.0-or-later, branch `twin`): builds on a
   current toolchain, has a JSON command-line interface, fixes the known defects behind a flag,
   and documents each defect with evidence and its measured effect (`docs/engine_changes.md`).
   It is written so the fixes could be offered upstream.
2. **This repository** (`github.com/bobvis1/iirabm-twin`, MIT): virtual patients, measurement
   model, simulation library, calibration (grid posterior and GA), validation, treatment search,
   Terraform, container, SageMaker job launchers, tests.
3. **A validation report** (`docs/validation_report.md`): parameter recovery, forecast
   calibration and interval coverage on held-out virtual patients, compared with a prior-only
   baseline, plus a measurement-design study (sampling interval, cytokine panel, assay noise).
4. **A treatment comparison** on a few test virtual patients: twin-personalized schedule vs best
   one-size-fits-all schedule vs no treatment, scored on fresh futures of the true patient.
5. `README.md` with a "For reviewers" section, architecture diagram, results filled from result
   files, cost, limitations and licensing; `docs/lab_standards.md`; `scripts/teardown.sh`.

## Key terms

| Term | Meaning here |
|---|---|
| θ (theta) | A patient's five model parameters: `inj_number`, `oxyHeal`, `infectSpread`, `numRecurInj`, `numInfectRepeat` |
| Virtual patient | A hidden θ plus a hidden seed (from the patient seed range). Its full trajectory is the truth |
| Decision time T_d | When the twin must forecast and choose a treatment. Default 72 h = step 612 |
| Readings | Noisy cytokine measurements every 6 h (every 51 steps) up to T_d |
| Library | Simulations over a grid of θ values × library seeds, full horizon, stored once and reused |
| Twin | The calibrated set of plausible θ (with weights) for one virtual patient |
| Horizon | 28 days = 5,760 steps unless a phase says otherwise |

## Repository layout (target)

```
CLAUDE.md  PLAN.md  README.md  LICENSE  Makefile  pyproject.toml  uv.lock
config/settings.example.yaml       # copy to config/settings.yaml (git-ignored)
engine/                            # git submodule: the GPL engine fork, pinned commit
infra/                             # Terraform, adapted from ../sepsisdemo/infra
container/Dockerfile               # builds engine from engine/ + installs src/twin
src/twin/
  config.py      engine.py     seeds.py      params.py     truth.py
  patients.py    observe.py    library.py    likelihood.py ga.py
  calibrate.py   forecast.py   evaluate.py   treat.py      sm_common.py
  jobs/          # entry points that run inside SageMaker jobs
scripts/         # infra.sh, push_image.sh, run_job.py, render_results.py, teardown.sh
tests/
docs/            # parameter_space.md, validation_report.md, lab_standards.md,
                 # future_work.md, results/ (small JSON), figures/
truth/           # git-ignored: hidden virtual-patient parameters, seeds, trajectories
data/            # git-ignored: library and run outputs
```

## Phases

| Phase | Topic | Hours |
|---|---|---|
| 0 | Accounts, forks, scaffold | 0.5 |
| 1 | Engine: legacy baseline, defects, refactor, CLI | 6 |
| 2 | Python wrapper, parameter space, virtual patients, measurements | 3.5 |
| 3 | AWS, container, simulation library on SageMaker | 5 |
| 4 | Calibration: grid posterior and GA | 5 |
| 5 | Validation and measurement design | 4 |
| 6 | Treatment search and scoring | 4 |
| 7 | Results, documentation, teardown | 2.5 |
| | **Total** | **30.5** |

---

### Phase 0: Accounts, forks, scaffold (0.5 h)

**[you]**
1. Confirm the license for this repository (default MIT; `LICENSE` is already in place).
2. Create an empty public GitHub repository `bobvis1/iirabm` for the engine fork. Then, in
   `../iirabm_public`:
   ```bash
   git remote add github https://github.com/bobvis1/iirabm.git
   git push github master            # master stays identical to upstream
   git checkout -b twin && git push -u github twin
   ```
3. Create the GitHub repository `bobvis1/iirabm-twin` (private until Phase 7 if you prefer).
4. **Request SageMaker quotas now** (approval can take a day) in the region from
   `config/settings.yaml`: Processing jobs on `ml.c5.9xlarge` (36 vCPU), at least 4 instances.
   If c5 is unavailable, use the nearest compute-optimized family and record it.
5. Decide whether to reuse the sepsis demo's AWS account and profile (recommended) and its
   budget alarm email.

**Claude**
- Add the submodule: `git submodule add -b twin https://github.com/bobvis1/iirabm.git engine`.
- Scaffold `pyproject.toml` (Python 3.11, `uv`, ruff, pytest; dependencies added as phases need
  them), `.gitignore` (data/, truth/, outputs/, docs_cache/, Terraform state, config/settings.yaml,
  .venv/), `config/settings.example.yaml` (as in the sepsis demo plus `name_prefix: twin`,
  `budget_usd: 50`), a `Makefile` with `help`, `test`, `lint` and `ENGINE_DIR ?= engine`, and
  `docs/future_work.md`.
- **Acceptance:** `git submodule status` shows the pinned engine commit; `make help` lists
  targets; `make test lint` pass on the empty scaffold.

---

### Phase 1: Engine: legacy baseline, defects, refactor, CLI (6 h)

All C++ work happens in `../iirabm_public` on branch `twin`, committed there and pushed to the
fork. This repository then bumps the `engine` submodule. Do the work in order, because each step
is checked against the one before.

**1.1 Legacy build and golden outputs (1.5 h).**
- `tools/build_legacy.sh`: check out the pristine upstream commit (`eaf7e08`) into a temporary
  git worktree, apply the minimal patches in `legacy_patches/`, and build `legacy_iirabm`. The
  patches may only make the code compile and run; they must not change any simulated number:
  - `mpi_shim.h`: a single-process stand-in for the MPI C++ calls the code uses (`Init`,
    `Finalize`, `Wtime`, `COMM_WORLD.Get_rank/Get_size/Scatter/Gather/Bcast/Barrier`).
  - Explicit `#include <random>` (or `-include random`).
  - The `outputSetup()` buffer overflow (defect 1): make `fileStart` large enough.
  - A per-config `Parameters.h` written by the script, so baselines can use different θ, seeds
    and step counts. Use `writeEverything=1`, `GA=0`, `testIntervention=0`.
- Golden outputs, generated **in the reference container** (Linux, GCC, `-O2
  -ffp-contract=off`): 8 configs of 3,000 steps (the default θ with seeds 1-4, plus four other θ
  chosen so at least one dies and one survives) and one full 12,500-step config. Add one
  **multi-run** legacy config (three seeds in one process through `pickParams`), which shows
  defect 3. Store under `tests/golden/` compressed, under 5 MB in total, with a README saying
  how they were made.

**1.2 Confirm the defects with evidence (0.5 h).** Record each in `docs/engine_changes.md`:
what, where, how found (AddressSanitizer output, a minimal reproduction), whether it changes
results, and the fix. Check every agent type (`pmn`, `mono`, `TH0`, `TH1`, `TH2`, the marrow and
germ types) for the self-erase pattern from defect 2, not only `mono`. Confirm defect 3 by showing
that legacy run 2 or 3 in one process differs from the same config run in a fresh process.

**1.3 Refactor without changing results (2 h).**
- A `Config` struct holds what varies per run: θ, `seed`, `num_steps`, `output_stride`,
  `reseed_step`/`reseed_seed` (optional), antibiotic timing and strength (defaults as V1), a list
  of intervention windows `{start_step, end_step, multipliers: {name: value}}`, `engine_fixes`
  (bool), and which outputs to record.
- `reset_state()` clears **all** global state, including `ecIndexes`, every total, every
  multiplier and every agent vector. The CLI always calls it, so defect 3 is fixed for the CLI
  regardless of `engine_fixes`. Defect 3 is a process artifact, not model behavior.
- `run_simulation(const Config&) -> Result` contains the time loop. Sampling point: the same as
  V1's `writeEverything` output (after `updateSystemOxy`, before the agent updates), recorded at
  steps where `istep % output_stride == 0`. With `output_stride = 1` it must equal the legacy
  output exactly. After death, record nothing further: report `died`, `death_step`, and the last
  values.
- Intervention windows must be able to express V1's GA schedule (start 505, spacing 100, length
  99), so the legacy GA is a special case.
- New CLI `iirabm_cli`: reads JSON Lines (one config per line) from `--input` or stdin and writes
  one JSON result per line to `--output` or stdout. Vendor `nlohmann/json` (single header, MIT,
  pinned version, with its license) under `V1/third_party/`. Result schema:
  ```json
  {"id": "...", "died": false, "death_step": null, "steps_run": 5760,
   "final": {"oxy_deficit": 0.0, "total_infection": 0.0},
   "series": {"step": [51, 102], "oxy_deficit": [], "total_infection": [], "TNF": [],
              "sTNFr": [], "IL10": [], "GCSF": [], "proTH1": [], "proTH2": [], "IFNg": [],
              "PAF": [], "IL1": [], "IL4": [], "IL8": [], "IL12": [], "sIL1r": [],
              "IL1ra": [], "pmn": [], "mono": [], "TH1": [], "TH2": []}}
  ```
  Never output IL-6 (defect 4).
- Build with a `Makefile` in the fork: targets `cli`, `legacy`, `asan`, `test`. Flags
  `-std=c++17 -O2 -ffp-contract=off`. No MPI dependency for `cli`.

**1.4 Fix defect 2 behind `engine_fixes` (0.5 h).** When an agent removes itself, it returns
immediately (for example, its update function returns `bool removed`), and the caller does not
advance its index past the agent that moved into the slot. Apply the same fix to every agent type
that has the pattern. With `engine_fixes=false`, keep the old behavior exactly.

**1.5 Tests in the fork (`make test`, run in the reference container) (1 h).**
- **Legacy equivalence:** for every single-run golden config, `iirabm_cli` with
  `engine_fixes=false` and `output_stride=1` matches the golden output exactly.
- **Determinism:** the same config twice gives byte-identical output.
- **Batch equals fresh:** 10 configs in one CLI process give the same results as each config in
  its own process, with `engine_fixes` both false and true.
- **Sanitizers:** with `engine_fixes=true`, an AddressSanitizer and UndefinedBehaviorSanitizer
  build runs 3 configs of 1,000 steps with no reports.
- **Reseed:** two configs with the same seed and different `reseed_seed` agree on every recorded
  value up to `reseed_step` and differ after it.
- **Neutral intervention:** an intervention window with all multipliers at 1.0 gives output
  identical to no intervention.

**1.6 Measure what the fixes change (0.5 h).** For 3 θ (a likely survivor, a borderline case,
a likely death), run 200 seeds each with `engine_fixes` false vs true. Report mortality (with
95% CIs) and the final `oxy_deficit` distributions. For defect 3, compare legacy multi-run
results with fresh-process results. Put the numbers in `docs/engine_changes.md`. Also report
time per step for both modes.

**GPL housekeeping:** add `COPYING` (GPL-2.0 text) to the fork; keep the upstream notice in
`README.md`; add a dated "Modified by Sridhar Viswanathan, 2026: <what>" notice at the top of
every changed file; add a "Changes in this fork" section to the fork's README.

**Acceptance:** all fork tests pass in the reference container; `docs/engine_changes.md` is
complete with numbers; the fork's `twin` branch is pushed; this repository's `engine` submodule
points at that commit. **[you]** Read `docs/engine_changes.md`; you may want to send it upstream
later (Phase 7).

---

### Phase 2: Python wrapper, parameter space, virtual patients, measurements (3.5 h)

**2.1 Engine wrapper (`twin.engine`).** `run_batch(configs, n_workers) -> list[Result]` splits
configs across `n_workers` long-lived `iirabm_cli` processes (JSON Lines over pipes), preserves
input order, and raises on any engine error with the failing config's `id`. The binary path comes
from `IIRABM_BIN`. `Result` is a dataclass with numpy arrays for the series.

**2.2 Seeds (`twin.seeds`).** The three non-overlapping ranges from CLAUDE.md, with helpers that
derive seeds deterministically from names (for example `library_seed(theta_id, k)`), so every
run can be reproduced from its config.

**2.3 Parameter space (`twin.params`) and pilot sweep (1.5 h).**
- Run a local pilot (reference container, all local cores): about 300 random θ × 4 library seeds,
  full 28-day horizon, `engine_fixes=true`. Use wide starting ranges around the V1 defaults
  (`inj_number` 33, `oxyHeal` 0.1, `infectSpread` 5, `numRecurInj` 2, `numInfectRepeat` 2), and
  widen if the edges are still interesting.
- Choose ranges where outcomes vary: avoid large regions where every virtual patient dies before
  72 h or nobody is ever sick. Record survival at 72 h and at 28 days by parameter. Note which
  parameters seem to matter most.
- Define the prior: uniform over a grid on those ranges. The grid's resolution is set by the
  Phase 3 compute budget. `oxyHeal` is continuous in the engine; the grid discretizes it, and the
  GA in Phase 4 may search it continuously.
- Write `docs/parameter_space.md` (ranges, grid, pilot figures). Do not quote published ranges
  unless you have checked the paper. **[you]** Approve the ranges and grid before Phase 3.

**2.4 Virtual patients (`twin.patients`, `twin.truth`).** Sample θ from the prior and a seed from
the patient range; simulate the full horizon; keep only virtual patients **alive at T_d** (like
ICU patients at enrollment). Build two cohorts: **development** (20 patients) and **test**
(40 patients). Write true θ, seeds and full trajectories to `truth/` (git-ignored). Only
`twin.truth` reads them, and only `twin.evaluate` and tests import `twin.truth`.

**2.5 Measurement model (`twin.observe`).**
- Panel (default): TNF, IL1, IL1ra, IL10, IL8, GCSF, IL12, IFNg. These match the cytokines the
  ReSCUED team's ARPA-H teaming profile asked a measurement partner to report every 4-6 hours.
  Cite the source in the docs.
- Readings at every 51 steps (6 h) from step 51 up to T_d: 12 readings by default.
- Noise: multiplicative log-normal with coefficient of variation 25% (configurable); a lower
  limit of detection per cytokine (values below it are reported as censored, not as numbers);
  optional missing readings (default 0%).
- Model cytokine totals are in arbitrary units; downstream code works in `log1p` space.
- The measurement random generator is seeded from the patient id, separately from engine seeds.
- Output: an `Observations` object (times, values, censoring flags, panel, T_d). Nothing else
  about the patient.

**Required tests:** truth isolation (`twin.calibrate`, `twin.treat`, `twin.forecast` and
`twin.likelihood` do not import `twin.truth` or reference `truth/`; check imports with `ast`, not
string search); seed ranges do not overlap; `run_batch` with 1 worker equals 4 workers;
measurements are reproducible from the patient id; censored readings are flagged, never zeros.

**Acceptance:** `make cohort` builds both cohorts locally; `docs/figures/example_patient.png`
shows one virtual patient's true trajectories, its readings and T_d.

---

### Phase 3: AWS, container, simulation library on SageMaker (5 h)

**3.1 Infrastructure (1 h).** Copy `../sepsisdemo/infra` and adapt: prefix `twin`, tag
`project=iirabm-twin`, drop the Model Registry and MLflow, keep VPC (no NAT, S3 gateway endpoint),
KMS, bucket, ECR, execution role, budget alarm. Copy and adapt `scripts/infra.sh`,
`scripts/teardown.sh`, `scripts/push_image.sh` and `src/twin/sm_common.py`. Tell the user the
cost as you apply (KMS about $1 per month; the rest is cents).

**3.2 Container (1 h).** Multi-stage `container/Dockerfile`: stage one builds `iirabm_cli` from
`engine/` (GCC, the reference flags); stage two is Python 3.11 slim with the `twin` package and
pinned dependencies installed, plus the binary. Tag images with this repository's git SHA and
record the engine commit as an image label. Build `linux/amd64` for SageMaker (Docker v2
manifests, as the sepsis demo learned) and `linux/arm64` for fast local runs.

**3.3 Job runner (1 h).** `scripts/run_job.py` launches a **SageMaker Processing job** with the
custom image, network isolation, KMS and the VPC. Input: JSON Lines config shards in S3,
distributed with `ShardedByS3Key` across `instance_count` instances. Inside, `twin.jobs.simulate`
runs one `iirabm_cli` worker per vCPU through `twin.engine.run_batch` and writes Parquet to the
output prefix. Processing jobs write plain files to S3 and are simple; they do not support Spot.
Note the Spot alternative (Training jobs) in `docs/future_work.md`.
- **Smoke job:** 100 configs. Check that results are byte-identical to the same configs run
  locally in the `linux/amd64` container, and compare with `linux/arm64`. Record time per step
  on the SageMaker instance and use it to size 3.4.

**3.4 The library (2 h, mostly waiting).** Grid θ (from Phase 2) × K library seeds (start with
K = 16), full horizon, `output_stride` 51, all outputs. **Size the grid so the whole library
costs at most about 100 CPU-hours** (about $10-15; state the estimate before launching). Store
as Parquet in long format (`theta_id, seed, step, variable values...`) plus a `thetas.parquet`
table. Download a local copy.
- Make the mortality landscape figure (28-day mortality over pairs of parameters), in the spirit
  of Cockrell and An's 2017 landscape analysis. Do not claim it reproduces their figures.

**Acceptance:** every (θ, seed) pair in the library is present exactly once; smoke parity
checked; the cost is recorded in the README cost section; the AWS resources that cost money by
the hour are stopped.

---

### Phase 4: Calibration: grid posterior and GA (5 h)

Work on the **development cohort only** in this phase. Do not load test-cohort observations.

**4.1 Likelihood (`twin.likelihood`) (1.5 h).** A Gaussian **synthetic likelihood** (Wood, 2010)
in `log1p` space. For a candidate θ with K simulated runs:
- **Survival term:** the virtual patient is observed alive at T_d, so add `log(fraction of runs
  alive at T_d)` (with a small floor). Per-reading moments use runs alive at that reading.
- **Reading term:** for each cytokine and reading time, the mean and variance of `log1p(value)`
  across the runs, plus the measurement noise variance; a normal log-density for observed
  readings and a normal log-CDF for censored ones.
- Readings are treated as independent given θ. That is an approximation; say so, and let Phase 5
  coverage show whether it matters.
- Unit tests on a toy simulator with a known answer.

**4.2 Grid posterior (`twin.calibrate`) (1 h).** Posterior ∝ prior × synthetic likelihood over
the library grid. This is cheap (no new simulations) and is the **reference** method. Two
forecast variants (`twin.forecast`):
- **θ-level:** mix the library runs of each θ by the θ's posterior weight.
- **Run-level:** additionally weight each library run by how well its own pre-T_d trajectory
  matches the readings (closer to particle filtering).
Choose between them, or keep both, using the development cohort only.

**4.3 GA calibration (`twin.ga`, `twin.calibrate`) (2 h).** The same objective, but each fitness
evaluation runs **fresh simulations** (calibration seeds, horizon T_d only, so about 0.6 s each),
with no library. This is the method that scales when θ is too large for a grid.
- A generic `twin.ga` module (unit-tested on a toy function): mixed integer and continuous genes,
  tournament selection (size 2), uniform crossover, and mutation that moves one gene a step up or
  down. These mirror the lab's 2018 GA design; cite it.
- Use common random numbers: within a generation, every candidate uses the same calibration
  seeds.
- Start with population 40, up to 25 generations, K = 8 seeds per candidate, and stop early on
  convergence. Tune these on development patients only.
- **Uncertainty:** keep every evaluated θ whose log-likelihood is within Δ of the best, weighted
  by likelihood. Choose Δ on the development cohort. Then simulate the full horizon for those θ
  to forecast.
- Run it as a SageMaker Processing job, several patients per instance, each GA using a share of
  the vCPUs. Estimate about 1-2 CPU-hours per patient and state the job's cost before launching.
- Compare with the grid posterior on development patients: overlap of the high-posterior region,
  agreement of forecasts, and the number of simulations each method used.

**4.4 Prior-only baseline.** The population forecast: all library runs alive at T_d, equally
weighted. Every result in Phase 5 is compared with it.

**Acceptance:** for 3 development patients, a figure shows readings, the twin's forecast bands
and (since these are development patients) the truth. The settings chosen on development
patients are written to `config/calibration.yaml`. Test-cohort observations have not been loaded.

---

### Phase 5: Validation and measurement design (4 h)

Run once on the **test cohort**, with the settings frozen in `config/calibration.yaml`. Only
`twin.evaluate` reads the truth.

**5.1 Parameter recovery (1 h).** For each parameter: posterior mean vs truth; 80% credible
interval coverage; the rank of the true value within the posterior. With a correct model, ranks
are uniform (simulation-based calibration, Talts et al. 2018). Also the posterior contraction:
posterior variance / prior variance, which shows which parameters 6-hourly cytokines can
identify.

**5.2 Forecasts (1.5 h).** For prior-only, grid θ-level, grid run-level and GA:
- 28-day mortality: Brier score, log loss and AUROC, with 95% CIs from a bootstrap over
  patients, plus a reliability plot. With 40 patients the intervals are wide; say so.
- Coverage of the 50%, 80% and 95% predictive intervals for `oxy_deficit` and each panel
  cytokine at days 4, 7, 14 and 28, among virtual patients alive at that day.

**5.3 Measurement design (1 h).** Using the library (re-weighting only, so no new simulations):
sampling every 6 vs 12 vs 24 h; observation window 24 vs 48 vs 72 h; assay noise 10% vs 25% vs
50%; the full panel vs each cytokine left out vs a small panel. Report the effect on Brier score,
coverage and posterior contraction. Nothing is selected from these results; they are reported.

**5.4 Report.** `docs/validation_report.md` and `docs/results/validation.json`, rendered into the
README by `scripts/render_results.py`.

**Required tests:** coverage and rank computations on a synthetic case with a known answer; the
bootstrap resamples patients; evaluation refuses to run on development patients unless asked
explicitly (so test results are never confused with development results).

**Acceptance:** report complete; every interval has its empirical coverage next to it; the
prior-only baseline is in every table.

---

### Phase 6: Treatment search and scoring (4 h)

**6.1 Action space (0.5 h).** From T_d, up to 4 windows of 99 steps, starting every 100 steps.
Four dials, chosen to resemble drugs that are approved for other uses:

| Dial | Levels | Resembles |
|---|---|---|
| IL1ra multiplier | 1, 2, 5, 10 | IL-1 receptor antagonist (anakinra) |
| TNF multiplier | 1, 0.75, 0.5, 0.25 | anti-TNF antibodies |
| GCSF multiplier | 1, 2, 5, 10 | G-CSF (filgrastim) |
| IFNg multiplier | 1, 2, 5, 10 | interferon gamma-1b |

That gives 16 genes with 4 levels each (4^16, about 4 billion schedules), which is why a GA is
used. Remember the engine semantics: values above 1 add (value - 1) per step; values at or below
1 scale.

**6.2 Search on the twin (1.5 h).** Fitness is evaluated on the twin's ensemble only: M = 8 θ
drawn from the posterior × 4 seeds. Twin runs re-simulate from injury, and runs dead before T_d
are replaced. Fitness: 28-day mortality across the ensemble, then mean final `oxy_deficit` +
`total_infection` as a tie-breaker (the V1 fitness). Use common random numbers within a
generation. Start with population 32 and 15 generations; **cap at about 10 CPU-hours per
patient** and state the cost before launching.

**6.3 One-size-fits-all schedule.** The same GA once, with the ensemble drawn from the prior
(library θ alive at T_d) instead of a twin.

**6.4 Scoring on the true patient (1 h).** For each chosen patient: true θ and true history
seed, reseeded at T_d with R = 64 seeds from the **future** range. The history up to T_d must
match the truth (assert it). Score no treatment, the one-size-fits-all schedule and the twin
schedule on the same 64 futures. Report mortality and paired differences with 95% CIs.
Optionally add an **oracle** schedule (GA on the true θ) as an upper bound.

**6.5 Which patients.** Choose 3-5 test patients by a rule fixed before any treatment result is
seen and written down in `docs/`. Suggested rule: twin-forecast 28-day mortality between 30% and
70%, so treatment could plausibly matter. The rule uses only the twin's forecast, never the truth.

**Acceptance:** a per-patient table and figure; the README states plainly that n is small and
that this is one 2017 model, not a clinical result.

---

### Phase 7: Results, documentation, teardown (2.5 h)

- `README.md`:
  - "For reviewers" section at the top.
  - Mermaid architecture diagram.
  - How to reproduce each phase.
  - Results tables filled by `scripts/render_results.py`.
  - Cost, limitations and licensing.
  - How it was built with Claude Code under these rules.
- `docs/lab_standards.md` adapted from the sepsis demo, plus the engine rules (reference
  platform, legacy equivalence, engine fixes measured before being adopted).
- Verify every citation in the README and docs against the source, or remove it.
- Run `scripts/teardown.sh` (ask first); confirm no tagged resources remain; record actual cost
  from Cost Explorer (it lags about a day).
- **[you]** Decide whether to offer the engine fixes upstream (a Bitbucket pull request or a short
  note to Dr. Cockrell with `docs/engine_changes.md`), and when: before or after applying.
- **[you]** Make `bobvis1/iirabm-twin` public; add the link to the resume and cover letter.

---

## Compute and cost estimate

Measured on Apple Silicon: about 1 ms per simulated step on one core. Phase 3's smoke job
replaces this with the SageMaker instance's figure.

| Work | CPU-hours (estimate) |
|---|---|
| Phase 2 pilot (local) | about 2, free |
| Library | at most 100 |
| GA calibration, 60 patients | about 60-120 |
| Treatment search, 5 patients + one-size-fits-all | about 60 |
| **Total on AWS** | **about 220-280** |

On 36-vCPU compute-optimized instances, that is roughly 7-8 instance-hours: an estimate of about
$15-40 including overhead. Check current prices before each launch. The budget alarm is $50.

## Cut order if needed

1. The oracle schedule (Phase 6.4).
2. The GA calibration (Phase 4.3); keep the grid posterior. The GA still appears in Phase 6.
3. Measurement-design studies beyond the sampling interval (Phase 5.3).
4. Smaller cohorts (test 40 → 20).
5. Phase 6 entirely: calibration with validated uncertainty is still a complete project.

Never cut: hidden truth, separate seed ranges, the development/test split, coverage reporting,
the prior-only baseline, the legacy-equivalence test, network isolation.

## Out of scope

Calibrating the engine's internal rules (the lab's later work does this), unknown injury time
(an extra nuisance parameter; note it in future work), real patient data, neural-network
surrogates of the engine, Spot training jobs, SageMaker Pipelines, MLflow, a web showcase (a
possible stretch after Phase 7).

## References (verify before quoting)

- Cockrell C, An G. Sepsis reconsidered: identifying novel metrics for behavioral landscape
  characterization with a high-performance computing implementation of an agent-based model.
  *Journal of Theoretical Biology*, 2017.
- Cockrell RC, An G. Examining the controllability of sepsis using genetic algorithms on an
  agent-based model of systemic inflammation. *PLoS Computational Biology*, 2018.
- Wood SN. Statistical inference for noisy nonlinear ecological dynamic systems. *Nature*, 2010.
- Talts S, et al. Validating Bayesian inference algorithms with simulation-based calibration.
  arXiv:1804.06788, 2018.
- ARPA-H CIRCLE teaming profiles (University of Vermont entry):
  https://arpa-h.gov/explore-funding/programs/circle/teaming
