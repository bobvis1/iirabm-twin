# Future work

Alternatives noted while choosing the option more likely to finish. Add to this file as phases go.

- **Spot training jobs** instead of Processing jobs for the simulation library (cheaper; needs
  checkpointing and a Training-job wrapper). Processing jobs were chosen because they are simple.
- **Unknown injury time** as an extra nuisance parameter in calibration.
- **Calibrating the engine's internal rules** (not attempted here; the parameters calibrated are the five patient parameters only).
- **Engine snapshot at the decision time**, so treatment search re-simulates only the
  post-decision steps. Skipped because it saves 11-21% of each run for a large C++ change.
- **Adaptive library**: extend the Sobol design with more points where development posteriors
  concentrate (a second library round), or trade seeds per point (K = 8) for more points.
- **Structural misspecification tests** beyond assay noise, for example an injury time that is
  off by a few hours, or a measurement model with a different limit of detection.
- **A sampler instead of an optimiser for calibration without a library** (for example sequential
  Monte Carlo or approximate Bayesian computation): the GA's evaluated points are not a posterior
  sample, and its parameter intervals under-cover (see `docs/calibration.md`).
- **Treatment**: more test patients, chosen so the *true* outcome is uncertain as well as the twin's
  (an oracle-based screen); larger ensembles and a held-out ensemble to measure search overfitting;
  schedules that adapt to the patient's readings after the decision time (closed loop).
- **Upstream the engine fixes** (`docs/engine_changes.md`): decided against for now (no pull request or
  note to the model's authors); the fixes live in the fork. Defect 2 (agents skipped after an erase)
  changes outcomes, so it would need a conversation with the authors before any proposal.
- **A faster engine path**: a snapshot at the decision time (above) and a smaller working set per
  process; the engine is cache-bound and was the dominant cost.
