# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 sagemath/sagemath@sha256:2401ffa8e9fc85c7ea17d3649bde5958b4dbf0858b3e504098c4102720151711 AS base

LABEL org.opencontainers.image.title="lattice-estimator-api" \
      org.opencontainers.image.description="Internal SageMath lattice-estimator adapter" \
      org.opencontainers.image.licenses="LGPL-3.0-or-later" \
      org.opencontainers.image.base.name="sagemath/sagemath:10.9" \
      org.opencontainers.image.source="https://github.com/malb/lattice-estimator"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LATTICE_ESTIMATOR_API_CLEANUP_GRACE_SECONDS=15

WORKDIR /app

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
USER sage

COPY --chown=sage:sage requirements.lock /app/requirements.lock
RUN sage -pip install --no-cache-dir --require-hashes -r /app/requirements.lock \
    && sage -pip install --no-cache-dir --no-deps \
      "lattice-estimator @ git+https://github.com/malb/lattice-estimator.git@53da5982597709ba0fdf94ea37a84d822310fd84"

COPY --chown=sage:sage src/ /app/src/
COPY --chown=sage:sage THIRD_PARTY_NOTICES.md /app/THIRD_PARTY_NOTICES.md

FROM base AS test
USER root
RUN sage -pip install --no-cache-dir httpx2==2.9.0 pytest==9.1.1
COPY --chown=sage:sage tests/ /app/tests/
COPY --chown=sage:sage pyproject.toml /app/pyproject.toml
USER sage
RUN sage -python -m pytest -p no:cacheprovider

FROM base AS runtime

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD sage -python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" || exit 1

USER sage
CMD ["sage", "-python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
