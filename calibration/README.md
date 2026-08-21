# Slow-attack calibration

The checked-in plans compare attack-specific preflight estimates with real
lattice-estimator Arora-GB and BKW results. They are research runs, not routine
CI: individual exact attacks can time out and the collector records those
outcomes while continuing. Collection is resumable.

Arora preflight v2 is a structural model, not a regression over these samples.
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
`calibration/baselines/slow-attacks-v2.json`; v1 is retained as history of the
replaced approximation. Across 168 comparable Arora-GB observations, including
an independent grid and small finite-sample/secret-diversity checks, the largest
unsafe error was about `2e-12` bit (floating-point noise). An 8-bit cushion
mathematically rounds to 9 bits, and production uses a 10-bit floor. The user
configured margin may only raise that value.

The BKW quick estimate is retained as a diagnostic, but its 527.883-bit worst
unsafe error is too large and dimension-dependent for production skipping.
Exact BKW took 2.279--3.782 seconds in the runtime smoke grid, so the Web
scheduler runs exact BKW and relies on its result cache.
