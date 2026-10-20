# A digital-twin loop on the IIRABM

Work in progress. A demonstration of "measure, model, modulate" on the public Innate Immune
Response Agent-Based Model (IIRABM, Cockrell and An, 2017), run in simulation where the true
answer is known, with simulations on Amazon SageMaker. See [PLAN.md](PLAN.md) for the plan and
[CLAUDE.md](CLAUDE.md) for the rules the work follows.

> **Research demonstration only.** Virtual patients from a 2017 model; no clinical claims.

## Licensing

- This repository (orchestration, analysis, infrastructure, docs) is MIT licensed; see
  [LICENSE](LICENSE).
- The simulation engine is a fork of IIRABM by Gary An and Chase Cockrell, licensed
  GPL-2.0-or-later. It is included as a git submodule at `engine/` and is not covered by this
  repository's license. This repository runs the engine as a separate program through its
  command-line interface and does not link to it.

<!-- cost:start -->
### Cost

Engine throughput on SageMaker Processing instances (`docs/results/sagemaker_timing.json`):

| Instance | Workers | Steps per second | $ per hour | $ per million steps |
|---|---|---|---|---|
| `ml.c5.9xlarge` | 18 of 36 | 4,911 | $1.836 | $0.104 |
| `ml.c6i.8xlarge` | 16 of 32 | 6,127 | $1.632 | $0.074 |
| `ml.c7i.8xlarge` | 24 of 32 | 7,461 | $1.714 | $0.064 |
| `ml.c8i.8xlarge` | 24 of 32 | 9,398 | $1.799 | $0.053 |

AWS public price list AmazonSageMaker us-east-2 (2026-10-16); ml.c8i.8xlarge derived as 1/4 of ml.c8i.32xlarge (not listed). The engine is cache-bound: one worker per vCPU is slower overall than one
per physical core.

Simulation library: 4,096 design points x 16 seeds =
65,536 runs (214 million steps) on 4 x
`ml.c8i.8xlarge`, 91.5 minutes, about **$10.98**
(estimate from instance-hours; confirm in Cost Explorer, which lags about a day).
<!-- cost:end -->
