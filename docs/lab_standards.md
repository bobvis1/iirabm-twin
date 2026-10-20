# Lab standards for simulation work on AWS (one page)

Adapted from the sepsis demo's standards, plus the rules this project added for a simulation engine.
Copy them for the next project.

**Naming.** Everything starts with `twin-` (prefix from `config/settings.yaml`): the IAM role
(`twin-sagemaker-execution`), the bucket (`twin-<account>-<region>`), the ECR repository `twin/main`, the
budget `twin-monthly`. The deployer IAM policy (`infra/iam/deployer-policy.json`) is scoped by this
prefix, so a mistyped name fails closed rather than touching other projects.

**Tagging.** Every resource and every job carries `project=iirabm-twin`, `owner=<from config>`,
`env=demo`: Terraform sets them with provider `default_tags`; SDK jobs set them in
`scripts/run_job.py`. `scripts/teardown.sh` finds what to stop by the `project` tag.

**Encryption.** One customer-managed KMS key encrypts the bucket (SSE-KMS, versioned, public access
blocked, TLS required), job volumes, job outputs and the ECR repository.

**Network isolation.** Jobs run in private subnets with no NAT gateway and no internet route; only an S3
gateway endpoint exists. Every job sets `enable_network_isolation=True`, so a container cannot call out
and **cannot `pip install` at run time: the engine binary and every Python dependency are baked into the
image** (`container/Dockerfile`, pinned `container/requirements.txt`).

**Identity.** Humans and agents use short-lived credentials, never root and never committed keys. The
deployer policy is least-privilege and region-pinned; jobs use a separate execution role limited to the
project bucket, key, logs and VPC network interfaces. No account IDs, ARNs or credentials are
committed: they live in `config/` (git-ignored). The deployer policy must be attached by an admin
before `scripts/infra.sh apply` can create `twin-*` resources.

**Cost.** Say what a billable step costs when you do it. Request quotas early (a new instance family
often starts at zero, and approval can take a day). **Measure before sizing:** the engine was 3 to 10
times slower per step on SageMaker than on a laptop, and one worker per vCPU was the slowest setting;
the benchmark (`twin.jobs.bench`) chose the instance and worker count before the large job ran. Jobs
print a cost upper bound and refuse to start above a limit without an explicit approval flag
(`scripts/run_job.py`). A monthly budget alarm is part of the base stack.

**Reproducibility.** Python, the SageMaker SDK (v3) and the container dependencies are pinned
(`pyproject.toml`, `uv.lock`, `container/requirements.txt`). Images are tagged with the git SHA and
carry the engine commit as a label. Every job runs locally on a small sample before it runs on SageMaker.

## Rules for a simulation engine

**Reference platform.** `std::uniform_int_distribution` and `std::shuffle` differ between libc++ and
libstdc++, and GCC fuses multiply-adds on arm64. The reference platform is therefore Linux, GCC 13.3,
libstdc++, `-O2 -ffp-contract=off` (never `-ffast-math` or `-march=native`), built in the container
pinned by digest in the engine fork's `tools/ref.sh`. Golden files and every reported number come from
it. macOS builds are for quick checks only. `linux/arm64` and `linux/amd64` containers were
checked byte-identical on ten golden runs, and a SageMaker run matched the local amd64 run value for value.

**Legacy equivalence.** The command-line engine with `engine_fixes=false` reproduces the unmodified
legacy code exactly (nine golden runs including a 12,500-step run), in the reference build.

**Engine behaviour changes are measured before they are adopted.** Each defect is documented with
evidence (AddressSanitizer output, a minimal reproduction) and its measured effect on outcome
distributions (`docs/engine_changes.md`). A fix that changes results sat behind a flag until it was
reported and accepted; legacy mode stays available.

**Determinism.** The same config gives byte-identical output, and many configs in one process equal
each config in a fresh process (tests in the engine fork).

## Rules for a digital-twin demonstration

**The truth is hidden.** Calibration, forecasting, likelihood and treatment code never import the truth
module or read `truth/` (an AST test checks the import closure). Only `twin.evaluate` reads it.

**Seed ranges never overlap** (`twin.seeds`, tested): library, calibration, search, patient and future.

**Development and test are separate.** Settings are chosen on the development cohort and frozen (the
config must match a commit) before the test cohort is scored once. The tuning rule that was revised
after the first run is recorded with the reason.

**Uncertainty is checked, not asserted.** Every reported interval has its empirical coverage; every
method is compared with a prior-only baseline; a result that looks too good is checked for leakage
(shuffled readings, an oracle) before it is reported.

**Words.** A demonstration of method on a public 2017 model; virtual patients, not patients;
"mediator interventions" that only resemble drugs; "in this model"; no clinical claims.
