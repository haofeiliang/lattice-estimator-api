"""Verify that every deployment dependency source names the locked estimator commit."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = re.compile(r"[0-9a-f]{40}")


def estimator_commits(path: str) -> set[str]:
    """Return full Git commits occurring on lines that name lattice-estimator."""
    return {
        commit
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines()
        if "lattice-estimator" in line
        for commit in COMMIT.findall(line)
    }


def main() -> None:
    """Fail when a manifest, lock file, or image build pins a different revision."""
    paths = ("pyproject.toml", "uv.lock", "requirements.lock", "Dockerfile")
    commits = {path: estimator_commits(path) for path in paths}
    if any(len(values) != 1 for values in commits.values()):
        raise SystemExit(f"each dependency source must contain one estimator commit: {commits}")
    unique = set().union(*commits.values())
    if len(unique) != 1:
        raise SystemExit(f"lattice-estimator dependency pins disagree: {commits}")
    print(f"lattice-estimator dependency pin: {unique.pop()}")


if __name__ == "__main__":
    main()
