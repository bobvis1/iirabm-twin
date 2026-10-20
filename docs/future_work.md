# Future work

Alternatives noted while choosing the option more likely to finish. Add to this file as phases go.

- **Spot training jobs** instead of Processing jobs for the simulation library (cheaper; needs
  checkpointing and a Training-job wrapper). Processing jobs were chosen because they are simple.
- **Unknown injury time** as an extra nuisance parameter in calibration.
- **Calibrating the engine's internal rules**, as the lab's later work does.
- **Engine snapshot at the decision time**, so treatment search re-simulates only the
  post-decision steps. Skipped because it saves 11-21% of each run for a large C++ change.
- **Adaptive library**: extend the Sobol design with more points where development posteriors
  concentrate (a second library round), or trade seeds per point (K = 8) for more points.
- **Structural misspecification tests** beyond assay noise, for example an injury time that is
  off by a few hours, or a measurement model with a different limit of detection.
