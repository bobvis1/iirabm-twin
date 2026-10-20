# Treatment search protocol (Phase 6)

*Written before any treatment result existed, so the choices below cannot depend on results.
Everything is "in this model": a public 2017 agent-based model, virtual patients, no clinical claim.
The four dials only resemble approved drugs.*

## Question

For a few test virtual patients, does a treatment schedule found on the patient's **twin** do better
on the **true** patient than (a) no treatment and (b) the best one-size-fits-all schedule?

## Action space

From the decision time T_d (72 h, step 612): 4 windows of 99 steps, starting every 100 steps
(`{start, end} = {612 + 100 w, 612 + 100 w + 99}`; the multipliers act in the agent passes of steps
start+1 .. end). Four dials per window, each with 4 levels (engine semantics: a value above 1 adds
(value - 1) per step; at or below 1 it scales):

| Dial | Levels | Resembles |
|---|---|---|
| IL1ra multiplier | 1, 2, 5, 10 | IL-1 receptor antagonist (anakinra) |
| TNF multiplier | 1, 0.75, 0.5, 0.25 | anti-TNF antibodies |
| GCSF multiplier | 1, 2, 5, 10 | G-CSF (filgrastim) |
| IFNg multiplier | 1, 2, 5, 10 | interferon gamma-1b |

16 genes with 4 levels each (4^16, about 4 billion schedules).

## Search (on the twin only; never sees the truth)

- **Ensemble:** M = 8 parameter sets drawn from the library posterior (frozen settings, theta-level
  weights) x 4 seeds each from the **search** seed range. Every ensemble run re-simulates from injury
  and must be alive at T_d; a run dead before T_d is replaced with another seed of the same theta.
  The ensemble is fixed for the whole search (common random numbers).
- **Fitness (lexicographic, minimised):** mortality by the fitness horizon H_f = 18 days (3,703 steps,
  fixed from the library in Phase 3) across the ensemble, then the mean of final `oxy_deficit` +
  `total_infection` (V1's fitness) as a tie-breaker.
- **GA:** population 24, tournament of 2, uniform crossover, one-gene-one-step mutation, elitism of
  2, up to 12 generations, stop after 4 generations without improvement. The "no treatment"
  schedule is in the initial population, so the result cannot be worse than doing nothing on the
  twin's ensemble. Cost cap about 10 CPU-hours (36 million steps) per search; 24 x 12 x 32 runs x
  3,703 steps is 34 million at most, less with early deaths.
- **One-size-fits-all:** the same GA once, with the ensemble drawn from the prior (library design
  points alive at T_d, equally weighted) instead of a twin.
- **Oracle (optional, an upper bound):** the same GA on the patient's true parameters (32 fresh
  seeds). Reported separately; dropped first if time or money run short, and said so.

## Which patients

Rule, fixed now: among the **test** cohort in id order, the first 5 virtual patients whose twin
forecasts a 28-day mortality between 30% and 70% (library posterior, theta-level, the settings
frozen in `config/calibration.yaml`), so that treatment could plausibly matter. The rule uses only the
twin's forecast, never the truth (`scripts/select_treatment_patients.py`).

## Scoring on the true patient (`twin.evaluate`)

For each chosen patient: the true parameters and true history seed, reseeded at T_d with R = 64 seeds
from the **future** seed range (never used by any search). The history up to T_d must equal the
stored truth exactly (asserted). The schedules compared on the same 64 futures: none, the
one-size-fits-all schedule, the twin schedule (and the oracle if run). Outcome: dead by day 28
(5,760 steps). Paired differences with 95% bootstrap intervals over futures, per patient, and a pooled
summary over patients.

## Honest limits

- n = 5 patients and 64 futures each: wide intervals; a null result is an expected and acceptable
  finding. No schedule is claimed to help real patients; the model, the dials and the search objective
  are all from a 2017 public model.
- The search objective (mortality by 18 days in the twin's ensemble) is a proxy for the scored
  outcome (mortality by 28 days on the true patient).
- A schedule found on 32 simulated runs can overfit them; scoring on fresh futures of the true
  patient is what protects against that.
