# CLAUDE.md

Instructions for Claude Code working in this repository. Read PLAN.md before starting any phase,
and read the whole phase section you are about to work on, including its acceptance criteria.

## Project in one paragraph

A time-boxed (about 30 hours of human time) demonstration of a medical digital-twin loop built on
the public Innate Immune Response Agent-Based Model (IIRABM, Cockrell and An, 2017). It runs the
loop "measure, model, modulate" end to end in simulation, where the true answer is known. Virtual
patients with hidden parameters produce noisy cytokine readings every 6 hours. The twin calibrates
to those readings, with uncertainty. The twin's forecasts are checked against the hidden truth.
Then a genetic algorithm searches for a treatment schedule on the twin, and that schedule is
scored on the true patient. Simulations run in parallel on Amazon SageMaker in a locked-down AWS
account. The audience is the An-Cockrell Lab at the University of Vermont (ReSCUED project,
ARPA-H CIRCLE program). Honest verification and validation matter more than impressive numbers.

## Two repositories, two licenses

- **The engine** is the user's fork of IIRABM, licensed **GPL-2.0-or-later**. The local working
  copy is `../iirabm_public` (upstream: `bitbucket.org/cockrell/iirabm_public`). It is included
  here as a git submodule at `engine/`, pinned to a commit. All C++ changes are made in the
  engine fork, on its `twin` branch, never in this repository.
- **This repository** (Python orchestration, analysis, infrastructure, docs) is **MIT**.
- **The boundary is a command-line interface.** Python calls the compiled engine as a separate
  program (subprocess, JSON Lines in and out). **Never link the engine into Python** (no
  pybind11, ctypes, cffi or shared library): that would make this repository a derivative work
  under the GPL. Never copy engine source files into this repository.
- Every modified engine file carries a dated notice of the change (GPL-2.0 section 2(a)), and the
  fork keeps the upstream copyright notice and includes the GPL text (`COPYING`).
- `ENGINE_DIR` (Makefile variable and environment variable) points at the engine source.
  Default: `engine`. During local development before the submodule exists, use
  `ENGINE_DIR=../iirabm_public`.

## The engine at a glance (V1, verified by reading and running it)

- C++, about 3,000 lines in `V1/`. Every setting is a compile-time constant in `Parameters.h`; the
  original `main` drives runs through the MPI C++ bindings (removed from modern MPI releases).
- One time step is about 7 minutes of patient time: 5,760 steps = 28 days, 205 steps is about
  1 day, and **6 hours is 51 steps** (the original 6-hourly output uses this).
- Patient parameters: `inj_number` (injury size, int), `oxyHeal` (host resilience, float),
  `infectSpread` (toxigenesis, int), `numRecurInj` (environmental toxicity, int),
  `numInfectRepeat` (microbial invasiveness, int), and the random `seed`.
- Outcome: the patient dies when `oxyDeficit > 8161`; the simulation then stops early. Otherwise
  `oxyDeficit` and `totalInfection` at the end describe the outcome.
- Built-in schedule: injury at step 0, recurrent injury every 205 steps, antibiotics at step 102
  and then every 205 steps (kills 80% of bacteria).
- Interventions: 12 multipliers (PAF, TNF, sTNFr, IL1, sIL1r, IL1ra, IFNg, IL4, IL8, IL10, IL12,
  GCSF), each from {0, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20}. Values at or below 1 scale a mediator;
  values above 1 add (value - 1) to it each step. Interventions last 99 steps.
- The original GA "reseeds" the random generator at a fixed step so that several futures branch
  from one shared history. The new CLI keeps this as `reseed_step` / `reseed_seed`.
- Speed: about 1 ms per step on one Apple Silicon core at `-O2`. A 72-hour window is about
  0.6 s; a 28-day run is up to about 6 s (deaths stop early).
- **Known defects** (Phase 1 confirms and fixes them):
  1. Stack buffer overflow in `outputSetup()` (`sepsisOutput.cpp`, `strcat` into a 13-byte
     array). Crashes on macOS before the simulation starts. Does not affect results.
  2. In `mono::mono_function` (`cellFunctions.cpp`) a monocyte that erases itself from
     `monoArray` then keeps writing through `this`, which now points at the next agent (or past
     the end). The caller's loop also skips the agent that shifted into the erased slot. This
     changes results. The other agent types should be checked for the same pattern.
  3. `ecIndexes` (`IIR_Sepsis.cpp`) is appended on every `mainSimulation` call and never cleared.
     From the second run in a process onward, some tissue cells are updated twice per step and
     others not at all. This changes results whenever one process runs several simulations,
     which is how the original GA worked.
  4. IL-6 is gathered and written but never recorded (`d_IL6` is never assigned). Do not report
     IL-6.
  5. Some source files rely on `<random>` being included transitively; modern compilers need it
     explicitly.

## Reference platform

`std::uniform_int_distribution` and `std::shuffle` are implementation-defined, so the same seed
gives different trajectories with libc++ (macOS) and libstdc++ (Linux). GCC on arm64 also fuses
multiply-adds by default, which changes float rounding. Therefore:

- The **reference platform** is the Linux container (GCC, libstdc++), built with
  `-O2 -ffp-contract=off` and never `-ffast-math` or `-march=native`. Golden files, legacy
  baselines and every reported number come from it.
- macOS-native builds are for quick development checks only; never compare their output with
  golden files.
- Phase 3 checks that `linux/arm64` (local Docker) and `linux/amd64` (SageMaker) produce
  identical output. If they do not, all reported runs use `linux/amd64`.

## Non-negotiable methodological rules

1. **The truth is hidden from the twin.** A virtual patient's true parameters and seeds are
   written to `truth/` files that only `twin.evaluate` (and tests) may read. Calibration and
   treatment code receive an `Observations` object and nothing else. A test enforces that
   `twin.calibrate` and `twin.treat` never import `twin.truth` or read `truth/` paths.
2. **Seed ranges never overlap.** Virtual-patient seeds come from `[1_000_000, 2_000_000)`,
   library and calibration seeds from `[1, 1_000_000)`, and post-decision "future" seeds used to
   score treatments on the true patient from `[2_000_000, 3_000_000)`. `twin.seeds` owns these
   ranges; a test checks them.
3. **The twin sees only the observation window.** Calibration uses readings up to the decision
   time (default 72 h). Forecasts are scored only on what happens after that time.
4. **Development and test patients are separate.** Virtual patients are split into a development
   cohort (used to choose the noise model, the GA settings, the acceptance rule and anything
   else that is tuned) and a test cohort that is scored once, for the final report. Nothing is
   tuned on test patients.
5. **Uncertainty claims are checked, not asserted.** Every reported interval comes with its
   empirical coverage on test patients. Every calibration method is compared against a
   prior-only baseline (the population forecast with no patient data).
6. **Treatments are scored on fresh futures.** A schedule found on the twin is scored on the
   true patient's parameters and history seed, with post-decision seeds from the reserved range
   that the search never used. It is always compared with no treatment and with the best
   one-size-fits-all schedule.
7. **The engine is deterministic.** The same config gives byte-identical output, and running
   many configs in one process gives the same results as running each in a fresh process. Both
   are tested.
8. **Engine behavior changes are measured.** Bug fixes that change results sit behind an
   `engine_fixes` flag until Phase 1 is accepted. With the flag off, the new CLI reproduces the
   legacy build exactly; with it on, the change in outcome distributions is reported. After
   Phase 1, fixes are on by default and the legacy mode stays available for comparison.

## Words to use and avoid

- This is a **demonstration of method on a public 2017 model**. Never claim that it reproduces
  the lab's current models, the ReSCUED twin, or real patient behavior.
- Virtual patients are "virtual patients", not patients. Multipliers are "mediator
  interventions"; where a dial resembles an approved drug (for example IL-1 receptor antagonist,
  anakinra), say "resembles", not "is".
- Results are "in this model". No clinical claims.

## AWS and cost rules

- **Tell the user about charges; explicit approval is not required.** The overall AWS budget
  (`budget_usd` in `config/settings.yaml`, default $50) is the guardrail. When you run a command
  that creates billable resources (Terraform apply, any SageMaker job, MLflow tracking servers,
  interface VPC endpoints), give a one-line cost estimate as you do it, and say when it is
  stopped or deleted. Still ask before destructive actions (Terraform destroy,
  `scripts/teardown.sh`, deleting data) and before any single job estimated above $10.
- Create resources through IaC (Terraform in `infra/`), not ad-hoc commands. Someone else should
  be able to point this repository at their own AWS account and reproduce it.
- Every resource gets tags: `project=iirabm-twin`, `owner=<from config>`, `env=demo`. Resource
  names start with `twin-`.
- Jobs run in private subnets with no NAT gateway, with network isolation and KMS encryption, as
  in the sepsis demo (`../sepsisdemo/infra`, `../sepsisdemo/docs/lab_standards.md`).
- Region and AWS profile come from config (`config/settings.yaml` or environment variables).
- Never commit credentials, account IDs or ARNs.
- Maintain `scripts/teardown.sh`.

## SageMaker SDK

The SageMaker Python SDK changed substantially across major versions. The sepsis demo pinned
`sagemaker==3.25.0` (v3); reuse that pin unless there is a reason not to. Check the installed
version and current AWS documentation before writing SDK code; do not rely on remembered
signatures. Save documentation you consult repeatedly in `docs_cache/` (git-ignored).

Jobs run with `enable_network_isolation=True`, so nothing can be installed at run time: the
engine binary and every Python dependency are baked into the image.

## Engineering conventions

- Python 3.11, package code under `src/twin/`, managed with `uv` and `pyproject.toml`.
- ruff for lint and format (line length 100); type hints on public functions; pytest under
  `tests/`.
- C++ in the engine fork: C++17, builds with `clang++` and `g++`, no MPI required for the new
  CLI. Keep the original sources compiling as the `legacy` target for comparison.
- **Local first:** every job runs locally on a small sample (for example 2 virtual patients and
  50 simulations) before it runs on SageMaker.
- Simulation outputs, Terraform state and `truth/` files are git-ignored. Small derived results
  that the README quotes (JSON summaries, figures) are committed under `docs/results/`.
- README and model-card numbers are filled from result files by a script, never hand-copied.
- Prefer small, readable modules. Reviewers will read this code.

## Working style

- Work phase by phase from PLAN.md. At the end of each phase, report what was done, how it was
  verified, the hours used, and anything surprising, then **stop and wait for the user** before
  starting the next phase.
- Tasks marked **[you]** in PLAN.md need the user (account actions, decisions, reviews). Ask for
  them; do not work around them.
- If a phase is running long, say so and propose what to cut, using the cut order in PLAN.md.
- When a choice is between something more impressive and something more likely to finish,
  choose the one more likely to finish and note the alternative in `docs/future_work.md`.
- If a result looks too good (for example coverage of exactly 100%, or a treatment that saves
  everyone), suspect leakage of the truth or reuse of seeds before reporting it.
