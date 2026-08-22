"""Versioned worker constants shared by the API and Sage child."""

from __future__ import annotations

import json
import re
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution

ADAPTER_VERSION = "5"
ADAPTER_SCHEMA_VERSION = 4
SAGE_VERSION = "10.9"
SAGE_IMAGE = (
    "sagemath/sagemath@sha256:2401ffa8e9fc85c7ea17d3649bde5958b4dbf0858b3e504098c4102720151711"
)

REQUEST_BODY_LIMIT_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 3_600
MAX_TIMEOUT_SECONDS = 7_200
DEFAULT_CLEANUP_GRACE_SECONDS = 15
DEFAULT_ESTIMATOR_CONCURRENCY = 3
MAX_ESTIMATOR_CONCURRENCY = 32


@cache
def estimator_commit() -> str:
    """Return the immutable VCS commit recorded for the installed estimator package."""

    try:
        direct_url = distribution("lattice-estimator").read_text("direct_url.json")
    except PackageNotFoundError as error:
        raise RuntimeError("lattice-estimator is not installed") from error
    if not direct_url:
        raise RuntimeError("lattice-estimator has no direct_url.json provenance")
    try:
        commit = json.loads(direct_url)["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("lattice-estimator has invalid VCS provenance") from error
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("lattice-estimator provenance is not a full Git commit")
    return commit
