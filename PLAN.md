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
   model, simulation library, calibration (library posterior and GA), validation, treatment
   search, Terraform, container, SageMaker job launchers, tests.
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
| Library | Simulations at N design points θ (scrambled Sobol over the prior) × K library seeds, full horizon, stored once and reused |
| Step label | The engine's `istep` after its increment at the top of the loop. The record at step s is taken after `updateSystemOxy(s)` and before that iteration's agent pass (V1 convention, pinned by tests in 1.5) |
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
| 4 | Calibration: library posterior and GA | 5 |
| 5 | Validation and measurement design | 4 |
| 6 | Treatment search and scoring | 4 |
| 7 | Results, documentation, teardown | 2.5 |
| | **Total** | **30.5** |

---

### Phase 0: Accounts, forks, scaffold (0.5 h)

**[you]**
1. Confirm the license for this repository (default MIT; `LICENSE` is already in place).
2. Create an empty public GitHub repository `bobvis1/iirabm` for the engine fork. Then, in
   `../iirabm`:
   ```bash
   git remote add github https://github.com/bobvis1/iirabm.git
   git push github main            # main stays identical to upstream
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

All C++ work happens in `../iirabm` on branch `twin`, committed there and pushed to the
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
- Golden outputs, generated **in the reference container** (`linux/amd64`, GCC, `-O2
  -ffp-contract=off`; emulated on the Mac is fine, since these are short runs): 8 configs of
  3,000 steps (the default θ with seeds 1-4, plus four other θ chosen so at least one dies and
  one survives) and one full 12,500-step config. Add one
  **multi-run** legacy config (three seeds in one process through `pickParams`), which shows
  defect 3. Store under `tests/golden/` compressed, under 5 MB in total, with a README saying
  how they were made (including the exact GCC and libstdc++ versions of the container).
- Build the same legacy configs in a native `linux/arm64` container and compare with the
  goldens. If they match, arm64 may be used for fast local runs; if not, record the first
  differing step and use amd64 for everything that is compared or reported. Doing this now,
  not in Phase 3, means the goldens never have to be regenerated.

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
- **Step conventions** (from reading V1's loop; written into `docs/engine_changes.md` and pinned
  by tests in 1.5). Step labels start at 1, so stride 51 records steps 51, 102, ... and the
  reading at T_d = 612 is the 12th. V1 checks `reseedTime` *before* incrementing `istep`, so
  reseeding at R changes the agent pass of the iteration labelled R + 1: records up to and
  including step R + 1 are shared history (if `updateSystemOxy` draws no random numbers; the
  test decides). The CLI keeps V1's comparisons for reseeding and intervention windows rather
  than "tidying" them, and documents exactly which step labels a window `{start_step,
  end_step}` affects.
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
  `engine_fixes=false` and `output_stride=1` matches the golden output exactly. Equivalence is
  defined for the reference build only (same compiler version, flags and standard library).
  The legacy path depends on undefined behavior (defect 2), so a refactor that changes inlining
  or layout could change it. A mismatch is diagnosed before anything is changed: does the
  divergence begin at a step where an agent erased itself? Explain it in
  `docs/engine_changes.md`; do not patch around it.
- **Step conventions:** the first stride-51 record is labelled 51 and equals legacy
  `writeEverything` row 51 (1-based); with `reseed_step` R, records match up to the documented
  boundary and differ after it; an intervention window leaves every record before its
  documented first affected step unchanged and changes a later one.
- **Determinism:** the same config twice gives byte-identical output.
- **Batch equals fresh:** 10 configs in one CLI process give the same results as each config in
  its own process, with `engine_fixes` both false and true.
- **Sanitizers:** with `engine_fixes=true`, an AddressSanitizer and UndefinedBehaviorSanitizer
  build runs 3 configs of 1,000 steps with no reports. Run this one natively (arm64 container or
  macOS); sanitizers are unreliable under emulation and the test is not about numbers.
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

**2.2 Seeds (`twin.seeds`).** The non-overlapping ranges from CLAUDE.md, with helpers that derive
seeds deterministically from names (for example `library_seed(theta_id, k)`), so every run can be
reproduced from its config. *Done:* the shared library-and-calibration range is split into library
`[1, 500_000)`, calibration `[500_000, 750_000)` and search `[750_000, 1_000_000)`, so library runs
are never reused by a calibration or a search; patients stay `[1_000_000, 2_000_000)` and futures
`[2_000_000, 3_000_000)`. Library seeds are `1 + theta_id * 16 + k` (unique); pilot sweeps use
design-point ids from 30,000 up, so the library is limited to fewer than 30,000 design points.
The other helpers return distinct seeds within a label (label hash plus a prime stride).

**2.3 Parameter space (`twin.params`) and pilot sweep (1.5 h).**
- Run a local pilot (reference container, all local cores): about 300 random θ × 4 library seeds,
  full 28-day horizon, `engine_fixes=true`. Use wide starting ranges around the V1 defaults
  (`inj_number` 33, `oxyHeal` 0.1, `infectSpread` 5, `numRecurInj` 2, `numInfectRepeat` 2), and
  widen if the edges are still interesting.
- Choose ranges where outcomes vary: avoid large regions where every virtual patient dies before
  72 h or nobody is ever sick. Record survival at 72 h and at 28 days by parameter. Note which
  parameters seem to matter most.
- Define the prior: uniform on those ranges, continuous for `oxyHeal` and uniform over the
  integers for the others.
- Define the library design: N scrambled Sobol points (fixed scramble seed) mapped onto the
  prior, integers by flooring. N is set by the Phase 3 compute budget. A lattice is not used:
  at the affordable N (a few thousand) it would give about 5 values per axis, and a posterior
  narrower than that spacing would collapse onto one point. Sobol points are a quasi-random
  sample from the prior, so the posterior is simply likelihood weights on the points (no kernel
  needed), every point has a distinct `oxyHeal`, and the design can be extended later with the
  next Sobol points.
- Write `docs/parameter_space.md` (ranges, design, pilot figures). Do not quote published ranges
  unless you have checked the paper. **[you]** Approve the ranges and design before Phase 3.
- *Done (2026-10-20):* pilot of 300 θ × 4 seeds (1,200 runs, 25 min locally); box kept as
  `inj_number` 10-50, `oxyHeal` 0.03-0.30, `infectSpread` 2-12, `numRecurInj` 1-6,
  `numInfectRepeat` 1-6 (`config/prior.yaml`). 77% of runs alive at 72 h; outcomes depend mostly on
  θ. `oxyHeal` does not affect survival to 72 h but strongly affects survival to 28 days, so
  readings up to T_d may say little about it (a Phase 5 question). The proposed library is
  N = 4,096 points (a power of two, so the Sobol points are balanced) × 16 seeds, about 57 CPU-hours
  at 1 ms per step; Phase 3 re-measures the step time and may revise N. Not explored:
  `numRecurInj` = 0, `inj_number` < 10. The assay limits of detection come from the pilot
  (5th percentile of true panel values; `config/measurement.yaml`).
- Local engine runs: `scripts/iirabm_docker.sh` (set as `IIRABM_BIN`) runs the pinned container;
  the Mac host was short of memory in Phase 1 and 2, which slowed Docker.

**2.4 Virtual patients (`twin.patients`, `twin.truth`).** Sample θ from the continuous prior
(independently of the library design, so a patient's θ is never a library point) and a seed from
the patient range; simulate the full horizon; keep only virtual patients **alive at T_d** (like
ICU patients at enrollment). Build two cohorts: **development** (40 patients) and **test**
(100 patients). Truth runs and library-based calibration are cheap, so the cohorts are sized
for tighter coverage estimates (80% coverage on 100 patients has a standard error of about 4
points, against about 6 on 40). The costly GA calibration (Phase 4.3) runs on fixed subsets:
the first 20 development and first 40 test patients by id, decided now. Write true θ, seeds
and full trajectories to `truth/` (git-ignored). Only `twin.truth` reads them, and only
`twin.evaluate` and tests import `twin.truth`.

**2.5 Measurement model (`twin.observe`).**
- Panel (default): TNF, IL1, IL1ra, IL10, IL8, GCSF, IL12, IFNg. These match the cytokines the
  ReSCUED team's ARPA-H teaming profile asked a measurement partner to report every 4-6 hours.
  Verify this against the source before citing it; if it cannot be verified, drop the claim and
  justify the panel on its own terms.
- Readings at every 51 steps (6 h) from step 51 up to T_d: 12 readings by default.
- Noise: multiplicative log-normal with coefficient of variation 25% (configurable); a lower
  limit of detection per cytokine (values below it are reported as censored, not as numbers);
  optional missing readings (default 0%).
- Model cytokine totals are in arbitrary units; downstream code works in `log1p` space.
- The measurement random generator is seeded from the patient id, separately from engine seeds.
- Output: an `Observations` object (times, values, censoring flags, panel, T_d). Nothing else
  about the patient.

**Required tests (all done):** truth isolation (`twin.calibrate`, `twin.treat`, `twin.forecast` and
`twin.likelihood` do not import `twin.truth` or reference `truth/`; check imports with `ast`, not
string search); seed ranges do not overlap; `run_batch` with 1 worker equals 4 workers;
measurements are reproducible from the patient id; censored readings are flagged, never zeros;
observation times, library series and truth series use the same step labels (a reading "at
step 612" is drawn from the truth value labelled 612, and the twin is given nothing labelled
after T_d).

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
manifests, as the sepsis demo learned) and, if Phase 1 showed it matches, `linux/arm64` for fast
local runs. Use the same base image and compiler version as the Phase 1 reference container.

**3.3 Job runner (1 h).** `scripts/run_job.py` launches a **SageMaker Processing job** with the
custom image, network isolation, KMS and the VPC. Input: JSON Lines config shards in S3,
distributed with `ShardedByS3Key` across `instance_count` instances. Inside, `twin.jobs.simulate`
runs one `iirabm_cli` worker per vCPU through `twin.engine.run_batch` and writes Parquet to the
output prefix. Processing jobs write plain files to S3 and are simple; they do not support Spot.
Note the Spot alternative (Training jobs) in `docs/future_work.md`.
- **Smoke job:** 100 configs. Check that results are byte-identical to the same configs run
  locally in the `linux/amd64` container (arm64 parity was settled in Phase 1). Record time per
  step on the SageMaker instance, with one worker per vCPU, and use it to size 3.4 and to
  recheck the Phase 4 and 6 budgets. Expect it to be slower than Apple Silicon.

**3.4 The library (2 h, mostly waiting; done 2026-10-20).** N Sobol design points (from Phase 2) × K library
seeds (K = 16), full horizon, `output_stride` 51, all outputs. **Choose N so the whole library
costs at most about 100 CPU-hours** (about $10-15; state the estimate before launching). Store
as Parquet in long format (`theta_id, seed, step, variable values...`) plus a `thetas.parquet`
table. Download a local copy.
- Make the mortality landscape figure (28-day mortality over pairs of parameters, binned), in
  the spirit of Cockrell and An's 2017 landscape analysis. Do not claim it reproduces their
  figures.
- From the library alone (no patient data), fix the **treatment fitness horizon** H_f for
  Phase 6: the shortest whole number of days by which at least 90% of 28-day deaths, among
  runs alive at T_d, have happened. Record it in `docs/parameter_space.md`.

*Done:* 4,096 x 16 on 4 x ml.c8i.8xlarge (24 workers each), 91.5 minutes, about $11; 65,536 runs,
every (design point, seed) pair present exactly once (checked by `scripts/analyze_library.py`);
H_f = 18 days (`docs/parameter_space.md`). The smoke job was byte-identical to the local amd64
container (100 runs, 6,601 series rows). See the measured cost table under "Compute and cost
estimate" for why the instance is c8i and not the planned c5.

**Acceptance:** every (θ, seed) pair in the library is present exactly once; smoke parity
checked; the cost is recorded in the README cost section; the AWS resources that cost money by
the hour are stopped.

---

### Phase 4: Calibration: library posterior and GA (5 h)

*Done (2026-10-20), see `docs/calibration.md`: settings in `config/calibration.yaml` (tau = 0.1, theta-level; the
selection rule was revised after the first run because pooled coverage hid `inj_number` being too
narrow); the GA's parameter intervals are too narrow on the development patients and are reported as
such; prior-only baseline implemented (`twin.calibrate.prior_only`).*

Work on the **development cohort only** in this phase. Do not load test-cohort observations.

**4.1 Likelihood (`twin.likelihood`) (1.5 h).** A Gaussian **synthetic likelihood** (Wood, 2010)
in `log1p` space. For a candidate θ with K simulated runs:
- **Survival term:** the virtual patient is observed alive at T_d, so add `log(fraction of runs
  alive at T_d)` (with a small floor). Per-reading moments use runs alive at that reading.
- **Reading term:** for each cytokine and reading time, the mean and variance of `log1p(value)`
  across the runs, plus the measurement noise variance; a normal log-density for observed
  readings and a normal log-CDF for censored ones.
- Readings are treated as independent given θ. They are not: given θ, a run's seed makes its
  deviations from the mean persist over time, so about 96 correlated readings (12 times × 8
  cytokines) are counted as independent evidence. **Expect the posterior to be too narrow.**
  Plan the remedies now, not after Phase 5:
  - A tempering exponent τ in (0, 1] on the reading term, chosen on the development cohort so
    that the 80% intervals for θ and forecasts cover close to 80%. Also report τ = 1, to show
    how much tempering mattered.
  - The **run-mixture likelihood**, which respects the correlation within a run:
    p(y | θ) ≈ (1/K) Σ_k Π_t p(y_t | run k at t), using only the measurement noise. It has high
    variance with K = 16 (one run often dominates), but it is the right object. It is the same
    computation as the run-level weights in 4.2.
  - A floor on the across-run variance (K = 16 gives noisy variance estimates), set as a
    fraction of the measurement-noise variance.
- Unit tests on a toy simulator with a known answer, including a correlated toy where the
  independence version is overconfident by a known amount.

**4.2 Library posterior (`twin.calibrate`) (1 h).** Posterior weights ∝ synthetic likelihood at
the library's design points (the points are a quasi-random sample from the prior, so the prior
weights are equal). This is cheap (no new simulations) and is the **reference** method. Report
the effective sample size (ESS) of the weights for every patient. If ESS is small (say under 20)
for many development patients, the library is too coarse for them: say so, and lean on the GA
comparison in 4.3. Two forecast variants (`twin.forecast`):
- **θ-level:** mix the library runs of each θ by the θ's posterior weight.
- **Run-level:** additionally weight each library run by how well its own pre-T_d trajectory
  matches the readings (closer to particle filtering).
Choose between them, or keep both, using the development cohort only.

**4.3 GA calibration (`twin.ga`, `twin.calibrate`) (2 h).** The same objective, but each fitness
evaluation runs **fresh simulations** (calibration seeds, horizon T_d only, so about 0.6 s each),
with no library. This is the method that scales when θ is too large for a library.
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
- Uses the same τ as 4.1 (re-chosen for K = 8 on development patients if needed).
- Compare with the library posterior on development patients: overlap of the high-posterior
  region, agreement of forecasts, and the number of simulations each method used.

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
interval coverage; the rank of the true value within the posterior. With a correct model and
exact inference, ranks are uniform (simulation-based calibration, Talts et al. 2018); the library
posterior is a discrete approximation, so expect small departures, and break ties at random. Also the posterior contraction:
posterior variance / prior variance, which shows which parameters 6-hourly cytokines can
identify.

**5.2 Forecasts (1.5 h).** For prior-only, library θ-level, library run-level and GA:
- 28-day mortality: Brier score, log loss and AUROC, with 95% CIs from a bootstrap over
  patients, plus a reliability plot. With 100 patients (40 for the GA) the intervals are still
  wide; say so.
- Coverage of the 50%, 80% and 95% predictive intervals for `oxy_deficit` and each panel
  cytokine at days 4, 7, 14 and 28, among virtual patients alive at that day.

**5.3 Measurement design (1 h).** Using the library (re-weighting only, so no new simulations):
sampling every 6 vs 12 vs 24 h; observation window 24 vs 48 vs 72 h; assay noise 10% vs 25% vs
50%; the full panel vs each cytokine left out vs a small panel. Report the effect on Brier score,
coverage and posterior contraction. Nothing is selected from these results; they are reported.

**5.3b Misspecification stress test (0.25 h).** Everything above is well specified: the virtual
patients come from the same engine, prior and noise model the twin assumes. There, near-nominal
coverage is the *expected* result. It shows the inference code is correct; it says nothing
about whether the model fits real patients. To show how the twin fails when an assumption is
wrong, regenerate the test readings from the stored truth with assay CV 40% while the twin still
assumes 25%, and report coverage and Brier score next to the well-specified numbers. This only
re-weights the library, so it needs no new simulations.

**5.4 Report.** `docs/validation_report.md` and `docs/results/validation.json`, rendered into the
README by `scripts/render_results.py`. State the well-specified framing of 5.3b near the top.

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
are replaced. Fitness: mortality by the fitness horizon H_f (fixed in 3.4) across the ensemble,
then mean final `oxy_deficit` + `total_infection` as a tie-breaker (the V1 fitness). Use common
random numbers within a generation.
- **The 10 CPU-hour cap per search is binding.** Cost ≈ population × generations × 32 runs ×
  (H_f in steps) × (time per step from 3.3), less early deaths. Defaults: population 24,
  15 generations = 11,520 runs; at H_f = 14 days (2,870 steps) and 1 ms per step that is at
  most about 9 CPU-hours. (The earlier default of population 32 at the full 28 days came to
  13-25 CPU-hours depending on early deaths, over the cap.) If the measured step time or H_f
  makes it larger, cut generations first, then population, and write the arithmetic in the
  launch log. State the cost before launching.
- Scoring (6.4) always uses the full 28 days, whatever H_f is.
- Not done: snapshotting engine state at T_d to skip the shared history. It would save only
  612 of 2,870-5,760 steps per run (11-21%) for a large C++ change (serializing every agent and
  the generator).

**6.3 One-size-fits-all schedule.** The same GA once, with the ensemble drawn from the prior
(library θ alive at T_d) instead of a twin.

**6.4 Scoring on the true patient (1 h).** For each chosen patient: true θ and true history
seed, reseeded at T_d with R = 64 seeds from the **future** range. The history up to T_d must
match the truth (assert it). Score no treatment, the one-size-fits-all schedule and the twin
schedule on the same 64 futures. Report mortality and paired differences with 95% CIs.
Optionally add an **oracle** schedule (GA on the true θ, under the same cap) as an upper bound.

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

**Measured (Phase 3).** The engine runs at about 1 ms per simulated step on one Apple M4 core, but
it is cache-bound, and on SageMaker it is much slower per step and sensitive to the worker count
(`docs/results/sagemaker_timing.json`, `src/twin/jobs/bench.py`). Throughput per instance, best
worker count, and price (AWS public price list, us-east-2; the c8i.8xlarge price is derived from the
c8i.32xlarge):

| Instance | Workers | Steps/s | $/hour | $ per million steps |
|---|---|---|---|---|
| ml.c5.9xlarge (plan's choice) | 18 of 36 vCPU | 4,911 | 1.836 | 0.104 |
| ml.c6i.8xlarge | 16 of 32 | 6,127 | 1.632 | 0.074 |
| ml.c7i.8xlarge | 24 of 32 | 7,461 | 1.714 | 0.064 |
| **ml.c8i.8xlarge (used)** | **24 of 32** | **9,398** (9,737 in the library job) | ~1.80 | **0.053** |

Never run one worker per vCPU: on the c5 it cut throughput by a third. **All SageMaker jobs use
ml.c8i.8xlarge with 24 workers** unless a later benchmark changes this.

| Work | Million steps | Cost on c8i (about $0.052 per million steps) |
|---|---|---|
| Phase 2 pilot (local) | 3.7 | free |
| Smoke, timing and benchmark jobs (c5, c6i, c7i, c8i) | | about $2 |
| **Library, as run (4,096 x 16, 91.5 min on 4 instances)** | **214** | **about $11 (estimate; check the bill)** |
| GA calibration, 60 patients (20 development + 40 test subsets) | about 190 (measured: 20 patients = 27.6 min on 4 instances) | **20 dev patients cost about $3.3; 40 test patients about $6.6** |
| Treatment search, 5 patients + one-size-fits-all (6 x at most 36M steps) | at most 216 | at most about $11 |
| Oracle schedules, optional (5 x at most 36M steps) | at most 180 | at most about $9 |
| **Total on AWS** | | **about $35-45** (library $11, dev GA $3.3, test GA about $6.6, search about $11, oracle about $9, tests $2) |

Earlier (Phase 0) estimate was $15-40 using 1 ms per step on every core; the measured cost is
higher. The budget alarm is $50 and is only an alarm. The plan's per-row "CPU-hour" caps (for
example "at most 10 CPU-hours per search") are now read as 36 million steps, the same work, which
costs about $1.9 per search on c8i. Every row is re-estimated before its job launches.

## Cut order if needed

1. The oracle schedule (Phase 6.4).
2. The GA calibration (Phase 4.3); keep the library posterior. The GA still appears in Phase 6.
3. Measurement-design studies beyond the sampling interval (Phase 5.3).
4. Smaller GA-calibration subsets (test 40 → 20). The library-based cohorts stay at full size.
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
