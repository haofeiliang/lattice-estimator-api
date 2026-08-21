# Slow-attack calibration

The checked-in plans compare attack-specific preflight estimates with real
lattice-estimator Arora-GB and BKW results. They are research runs, not routine
CI: individual exact attacks can time out and the collector records those
outcomes while continuing. Collection is resumable.

Arora preflight is a structural model, not a regression over these samples.
It reproduces the upstream Gaussian-tail sample count, semi-regular
Hilbert-series solving degree, and dense or sparse guessing composition. The
search is bounded to keep preflight cheap. Calibration is used only to measure
unsafe error and select one safety margin; there are no fitted coefficients or
parameter-specific correction tables.

The Arora plan covers `n` through 1024, `q` through 4096, and Gaussian
`sigma=0.3..4.0`. Values below 0.7 are retained here to characterize the
boundary even if the public backend later rejects them for production inputs.
The `arora-n1024-holdout-v1.json` plan gives the production-valid small-noise
`n=1024` boundary a longer timeout after the broad 600-second sweep.
`arora-v2-validation-v1.json` is an independent grid using `q=4093`, while the
two diversity smoke plans cover finite samples plus uniform and fixed-weight
binary/ternary secrets. Finite samples and a less-tested secret are not reasons
to disable preflight: unsupported or sample-starved instances return a
non-finite preflight and therefore fall through to exact Arora-GB.

Plan format v2 accepts complete error-distribution objects under the `error`
axis. `bounded-errors-v2.json` covers centered binomial eta 1, 2, 4, and 8 and
symmetric uniform integer radii 1, 2, 4, and 8, with `n` through 1024 and `q`
through 4096. `bounded-errors-diversity-v2.json` varies the secret under
unlimited samples. The finite-sample cases are deliberately separated into
`bounded-errors-finite-holdout-v2.json` because that holdout delimits the
reviewed Arora-GB domain.

With a locally running API:

```bash
python3 tools/slow_attack_calibration.py collect \
  --plan calibration/plans/slow-attacks-v1.json \
  --output calibration/arora-bkw-observations.jsonl \
  --workers 16

python3 tools/slow_attack_calibration.py collect \
  --plan calibration/plans/arora-n1024-holdout-v1.json \
  --output calibration/arora-n1024-holdout-observations.jsonl \
  --workers 6

python3 tools/slow_attack_calibration.py summarize \
  --input calibration/arora-bkw-observations.jsonl \
  --output calibration/arora-bkw-summary.json \
  --safety-cushion-bits 8

python3 tools/slow_attack_calibration.py select-plan \
  --plan calibration/plans/bounded-errors-v2.json \
  --input calibration/bounded-errors-v4-observations.jsonl \
  --output calibration/bounded-errors-reviewed-v4-observations.jsonl
```

After changing only the preflight implementation, preserve the expensive exact
observations and replay them:

```bash
python3 tools/slow_attack_calibration.py replay-preflight \
  --input calibration/arora-bkw-observations.jsonl \
  --output calibration/arora-bkw-v2-replay.jsonl \
  --workers 16
```

The unsafe error is `preflight_bits - exact_bits`. The recommended margin is
the largest observed positive error plus the requested cushion, separately for
each attack. Failed, timed-out, and non-finite pairs never justify skipping an
attack.

The current reviewed result is
`calibration/baselines/slow-attacks-v4.json`. The discrete-Gaussian v3 evidence
is retained. Across 127 new comparable unlimited-sample bounded-error Arora-GB
observations, the largest unsafe error was about `1.2e-12` bit. The finite
bounded-error holdout exposed an 88.170-bit underestimate, so finite-sample
bounded Arora-GB remains outside the reviewed domain and always runs exact.
The one Arora-GB production margin remains a 10-bit floor across all reviewed
Gaussian and bounded domains; user configuration may only raise it.

The old BKW table heuristic had a 527.883-bit worst unsafe error. It has been
replaced by a structural log-domain implementation of the pinned coded-BKW
equations. Across 357 comparable observations in the broad, diversity, and
large-dimension grids, its largest unsafe error was 0.395 bit. The common
8-bit cushion gives 9 bits, and production uses one 10-bit BKW floor. The new
bounded grids added 208 comparable unlimited-sample and 8 comparable
finite-sample observations; their worst unsafe errors were 0.369 bit and
0.000018 bit. The reviewed BKW domain therefore includes centered-binomial eta
1 through 8 and symmetric uniform radii 1 through 8 for finite or unlimited
samples. Other bounded errors and every `preflight_unknown` result run exact
BKW. Exact BKW reached 7.863 seconds at n=2048/4096, while the preflight
requests took at most 1.688 seconds in that holdout.
