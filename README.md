# lattice-estimator-api

Stateless FastAPI service around SageMath and
[`malb/lattice-estimator`](https://github.com/malb/lattice-estimator).
Each estimate runs in a dedicated Sage process group so timeout, cancellation,
and descendant-process cleanup remain enforceable.

## Interface

- `GET /healthz`
- `GET /v1/metadata`
- `POST /v1/estimate`
- `POST /v1/preflight` — cheap Arora-GB/BKW estimates used by the Web scheduler

The preflight values are attack-selection estimates, not security reports.
They must be calibrated against `/v1/estimate` before changing the production
stop margin. Calibration plans and tooling live under `calibration/` and
`tools/`; the expensive matrix is intentionally not part of routine CI.

Rule v5 reviews discrete-Gaussian errors plus centered-binomial eta 1 through 8
and symmetric uniform integer `[-r,r]` for `r=1..8`. The Web scheduler applies
one 10-bit margin per attack. Arora-GB and BKW admit finite or unlimited samples
in the reviewed bounded domain. Unknown preflight outcomes always run exact.

The upstream estimator is installed as a Python package from the pinned
`main` commit recorded in `pyproject.toml` and `uv.lock`. Runtime provenance is
read from the installed distribution's `direct_url.json`.

## Development

Fast tests use a mock estimator and do not require Sage:

```bash
uv sync --frozen --all-groups
uv run --frozen pytest tests/unit
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

Real Sage verification runs in the image test stage:

```bash
docker build --target test -t lattice-estimator-api:test .
docker build --target runtime -t lattice-estimator-api:dev .
docker run --rm -p 127.0.0.1:8000:8000 lattice-estimator-api:dev
```

## Releases

Pull requests run the unit checks. Ordinary branch pushes do not trigger CI.
Pushing a semantic-version tag builds and smoke-tests the Sage image before
publishing it to `ghcr.io/<repository-owner>/lattice-estimator-api`:

```bash
git tag -a v0.1.0 -m "lattice-estimator-api v0.1.0"
git push origin v0.1.0
```

Every tag publishes its exact version and `sha-<commit>`. The highest stable
semantic version also publishes `latest` from the same image manifest. A
pre-release tag such as `v0.2.0-rc.1`, or a stable tag older than an existing
release, never updates `latest`.

## Configuration

| Variable | Default |
| --- | --- |
| `LATTICE_ESTIMATOR_API_CONCURRENCY` | `3` |
| `LATTICE_ESTIMATOR_API_WORKER_COMMAND` | `sage -python -m src.worker` |
| `LATTICE_ESTIMATOR_API_CLEANUP_GRACE_SECONDS` | `15` |
