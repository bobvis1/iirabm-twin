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
