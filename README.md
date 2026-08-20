# lattice-estimator-api

Stateless FastAPI service around SageMath and
[`malb/lattice-estimator`](https://github.com/malb/lattice-estimator).
Each estimate runs in a dedicated Sage process group so timeout, cancellation,
and descendant-process cleanup remain enforceable.

## Interface

- `GET /healthz`
- `GET /v1/metadata`
- `POST /v1/estimate`

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

## Configuration

| Variable | Default |
| --- | --- |
| `LATTICE_ESTIMATOR_API_CONCURRENCY` | `3` |
| `LATTICE_ESTIMATOR_API_WORKER_COMMAND` | `sage -python -m src.worker` |
| `LATTICE_ESTIMATOR_API_CLEANUP_GRACE_SECONDS` | `15` |
