"""Collect and summarize slow-attack preflight calibration observations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PLAN_FORMAT = "lattice-estimator/slow-attack-calibration-plan"
OBSERVATION_FORMAT = "lattice-estimator/slow-attack-calibration-observation"
V6_TARGETS = (
    96,
    112,
    120,
    128,
    136,
    144,
    160,
    176,
    184,
    192,
    200,
    208,
    224,
    240,
    248,
    256,
    264,
    272,
    288,
)


def _preflight_payload(
    payload: dict[str, Any], required_security_bits: int = 128, requested_margin_bits: int = 0
) -> dict[str, Any]:
    return {
        **payload,
        "operation": "preflight",
        "required_security_bits": str(required_security_bits),
        "requested_arora_gb_coarse_margin_bits": str(requested_margin_bits),
        "requested_arora_gb_refined_margin_bits": str(requested_margin_bits),
    }


def requests_from_plan(plan: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if plan.get("format") != PLAN_FORMAT or plan.get("version") not in {1, 2}:
        raise ValueError("unsupported calibration plan")
    for experiment in plan["experiments"]:
        axes = experiment["axes"]
        if plan["version"] == 1:
            errors = [
                {
                    "kind": "discrete_gaussian",
                    "standard_deviation": sigma,
                }
                for sigma in axes["error_standard_deviation"]
            ]
        else:
            errors = axes["error"]
        for model, n, q, samples, secret, error in itertools.product(
            plan["models"],
            axes["dimension"],
            axes["modulus"],
            axes["samples"],
            axes["secret"],
            errors,
        ):
            yield {
                "schema_version": 5,
                "problem": {
                    "kind": "lwe",
                    "dimension": n,
                    "modulus": q,
                    "samples": samples,
                    "secret": secret,
                    "error": error,
                },
                "cost_model": model["cost_model"],
                "target_attacks": [experiment["attack"]],
                "timeout_seconds": plan["timeout_seconds"],
            }


def _post(
    url: str,
    payload: dict[str, Any],
    client_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        timeout = client_timeout_seconds or payload["timeout_seconds"] + 30
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned {error.code}: {detail}") from error


def _identity(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _result(response: dict[str, Any]) -> dict[str, Any]:
    return response["results"][0]["outcome"]


def _collect_one(payload: dict[str, Any], base_url: str) -> dict[str, Any]:
    identity = _identity(payload)
    preflight_payload = _preflight_payload(payload)
    try:
        preflight = _post(
            f"{base_url.rstrip('/')}/v1/preflight",
            preflight_payload,
            client_timeout_seconds=min(payload["timeout_seconds"] + 30, 120),
        )
        preflight_outcome = _result(preflight)
    except (RuntimeError, OSError, TimeoutError) as error:
        preflight = None
        preflight_outcome = {"kind": "collection_failed", "message": str(error)}
    try:
        exact = _post(f"{base_url.rstrip('/')}/v1/estimate", payload)
        exact_outcome = _result(exact)
    except (RuntimeError, OSError, TimeoutError) as error:
        exact = None
        exact_outcome = {"kind": "collection_failed", "message": str(error)}
    return {
        "format": OBSERVATION_FORMAT,
        "version": 1,
        "identity": identity,
        "request": payload,
        "preflight": preflight_outcome,
        "preflight_duration_ms": None if preflight is None else preflight.get("duration_ms"),
        "exact": exact_outcome,
        "exact_duration_ms": None if exact is None else exact.get("duration_ms"),
        "provenance": (exact or preflight or {}).get("provenance"),
    }


def collect(plan_path: Path, output: Path, base_url: str, workers: int) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    existing: set[str] = set()
    if output.exists():
        existing = {
            row["identity"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip() and not _collection_failed(row := json.loads(line))
        }
    planned_requests = requests_from_plan(plan)
    pending = [payload for payload in planned_requests if _identity(payload) not in existing]
    completed = len(existing)
    total = completed + len(pending)
    with (
        output.open("a", encoding="utf-8") as destination,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        futures = {executor.submit(_collect_one, payload, base_url) for payload in pending}
        for future in as_completed(futures):
            row = future.result()
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            destination.flush()
            completed += 1
            print(f"collected {completed}/{total}", flush=True)


def _replay_one(row: dict[str, Any], base_url: str) -> dict[str, Any]:
    payload = row["request"]
    preflight_payload = _preflight_payload(payload)
    try:
        preflight = _post(
            f"{base_url.rstrip('/')}/v1/preflight",
            preflight_payload,
            client_timeout_seconds=min(payload["timeout_seconds"] + 30, 120),
        )
        preflight_outcome = _result(preflight)
    except (RuntimeError, OSError, TimeoutError) as error:
        preflight = None
        preflight_outcome = {"kind": "collection_failed", "message": str(error)}
    return {
        "format": OBSERVATION_FORMAT,
        "version": 1,
        "identity": row["identity"],
        "request": payload,
        "preflight": preflight_outcome,
        "preflight_duration_ms": None if preflight is None else preflight.get("duration_ms"),
        "exact": row["exact"],
        "exact_duration_ms": row.get("exact_duration_ms"),
        "provenance": row.get("provenance"),
        "preflight_provenance": None if preflight is None else preflight.get("provenance"),
    }


def replay_preflight(source: Path, output: Path, base_url: str, workers: int) -> None:
    """Re-run only preflight while preserving exact outcomes from a pinned dataset."""
    source_lines = source.read_text(encoding="utf-8").splitlines()
    source_rows = [json.loads(line) for line in source_lines if line.strip()]
    existing: set[str] = set()
    if output.exists():
        existing = {
            json.loads(line)["identity"]
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [row for row in source_rows if row["identity"] not in existing]
    completed = len(existing)
    total = completed + len(pending)
    with (
        output.open("a", encoding="utf-8") as destination,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        futures = {executor.submit(_replay_one, row, base_url) for row in pending}
        for future in as_completed(futures):
            row = future.result()
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            destination.flush()
            completed += 1
            print(f"replayed {completed}/{total}", flush=True)


def select_plan(plan_path: Path, source: Path, output: Path) -> None:
    """Materialize the latest observation for every request in a plan."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = {row["identity"]: row for row in _latest_rows([source])}
    selected = []
    missing = []
    for payload in requests_from_plan(plan):
        identity = _identity(payload)
        if identity in rows:
            selected.append(rows[identity])
        else:
            missing.append(identity)
    if missing:
        raise ValueError(f"source is missing {len(missing)} planned observations")
    output.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )


def _computed_bits(outcome: dict[str, Any]) -> float | None:
    if outcome.get("kind") != "computed":
        return None
    value = float(outcome["security_bits"])
    return value if math.isfinite(value) else None


def _collection_failed(row: dict[str, Any]) -> bool:
    return any(
        row.get(key, {}).get("kind") == "collection_failed" for key in ("preflight", "exact")
    )


def _latest_rows(inputs: list[Path]) -> Iterable[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    anonymous = 0
    for path in inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("identity")
            if not isinstance(identity, str):
                identity = f"anonymous:{anonymous}"
                anonymous += 1
            latest[identity] = row
    return latest.values()


def _duration_summary(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.floor(len(ordered) * 0.95))],
        "maximum": ordered[-1],
    }


def summarize(inputs: list[Path], output: Path, cushion: float) -> None:
    deltas: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    preflight_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    exact_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    preflight_durations: dict[str, list[int]] = defaultdict(list)
    exact_durations: dict[str, list[int]] = defaultdict(list)
    worst: dict[str, tuple[float, dict[str, Any] | None]] = {}
    for row in _latest_rows(inputs):
        attack = row["request"]["target_attacks"][0]
        quick = _computed_bits(row["preflight"])
        exact = _computed_bits(row["exact"])
        counts[attack] += 1
        preflight_outcomes[attack][row["preflight"]["kind"]] += 1
        exact_outcomes[attack][row["exact"]["kind"]] += 1
        if isinstance(row.get("preflight_duration_ms"), int):
            preflight_durations[attack].append(row["preflight_duration_ms"])
        if isinstance(row.get("exact_duration_ms"), int):
            exact_durations[attack].append(row["exact_duration_ms"])
        if quick is not None and exact is not None:
            delta = quick - exact
            deltas[attack].append(delta)
            if attack not in worst or delta > worst[attack][0]:
                worst[attack] = (delta, row["request"].get("problem"))
    attacks = {}
    for attack in sorted(counts):
        values = deltas[attack]
        maximum = max([0.0, *values])
        attacks[attack] = {
            "observations": counts[attack],
            "comparable": len(values),
            "maximum_unsafe_error_bits": maximum,
            "recommended_margin_bits": math.ceil(maximum + cushion),
            "preflight_outcomes": dict(sorted(preflight_outcomes[attack].items())),
            "exact_outcomes": dict(sorted(exact_outcomes[attack].items())),
            "preflight_duration_ms": _duration_summary(preflight_durations[attack]),
            "exact_duration_ms": _duration_summary(exact_durations[attack]),
        }
        if attack in worst and worst[attack][1] is not None:
            attacks[attack]["worst_observed_problem"] = worst[attack][1]
    output.write_text(
        json.dumps(
            {
                "format": "lattice-estimator/slow-attack-calibration-summary",
                "version": 1,
                "safety_cushion_bits": cushion,
                "attacks": attacks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_arora_v6_local(
    inputs: list[Path],
    output: Path,
    targets: list[int],
    coarse_margin_floor_bits: int,
    refined_margin_floor_bits: int,
) -> None:
    """Replay the target screen in one Sage process against pinned exact outcomes."""
    from src.adapter import _lwe_parameters
    from src.models import LweProblem
    from src.slow_estimate import arora_gb_threshold_screen

    counts: Counter[str] = Counter()
    tier_counts: dict[str, Counter[str]] = defaultdict(Counter)
    durations: dict[str, list[int]] = defaultdict(list)
    unsafe: list[dict[str, Any]] = []
    source_rows = [
        row for row in _latest_rows(inputs) if row["request"]["target_attacks"] == ["arora_gb"]
    ]
    rows = [row for row in source_rows if _computed_bits(row["exact"]) is not None]
    for row in rows:
        exact_bits = _computed_bits(row["exact"])
        problem = LweProblem.model_validate(row["request"]["problem"])
        params = _lwe_parameters(problem)
        for target in targets:
            started = time.monotonic()
            screen = arora_gb_threshold_screen(
                params,
                target,
                0,
                0,
                coarse_margin_floor_bits=coarse_margin_floor_bits,
                refined_margin_floor_bits=refined_margin_floor_bits,
            )
            duration_ms = max(0, round((time.monotonic() - started) * 1_000))
            counts[screen.decision] += 1
            tier_counts[screen.precision_tier][screen.decision] += 1
            durations[screen.precision_tier].append(duration_ms)
            if (
                screen.decision == "above_threshold"
                and exact_bits is not None
                and exact_bits < target
            ):
                unsafe.append(
                    {
                        "target_bits": target,
                        "exact_security_bits": exact_bits,
                        "tier": screen.precision_tier,
                        "effective_margin_bits": screen.effective_margin_bits,
                        "problem": row["request"]["problem"],
                    }
                )
    output.write_text(
        json.dumps(
            {
                "format": "lattice-estimator/arora-gb-v6-validation",
                "version": 1,
                "targets": targets,
                "coarse_margin_floor_bits": coarse_margin_floor_bits,
                "refined_margin_floor_bits": refined_margin_floor_bits,
                "source_arora_rows": len(source_rows),
                "source_rows": len(rows),
                "excluded_without_finite_exact_bits": len(source_rows) - len(rows),
                "decisions": dict(sorted(counts.items())),
                "tiers": {
                    tier: {
                        "decisions": dict(sorted(decisions.items())),
                        "duration_ms": _duration_summary(durations[tier]),
                    }
                    for tier, decisions in sorted(tier_counts.items())
                },
                "unsafe_skip_count": len(unsafe),
                "unsafe_skips": unsafe,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--plan", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--estimator-url", default="http://127.0.0.1:8000")
    collect_parser.add_argument("--workers", type=int, default=3)
    replay_parser = commands.add_parser("replay-preflight")
    replay_parser.add_argument("--input", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--estimator-url", default="http://127.0.0.1:8000")
    replay_parser.add_argument("--workers", type=int, default=3)
    select_parser = commands.add_parser("select-plan")
    select_parser.add_argument("--plan", type=Path, required=True)
    select_parser.add_argument("--input", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--input", type=Path, action="append", required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.add_argument("--safety-cushion-bits", type=float, default=8.0)
    v6_parser = commands.add_parser("validate-arora-v6-local")
    v6_parser.add_argument("--input", type=Path, action="append", required=True)
    v6_parser.add_argument("--output", type=Path, required=True)
    v6_parser.add_argument("--targets", type=int, nargs="*", default=list(V6_TARGETS))
    v6_parser.add_argument("--coarse-margin-floor-bits", type=int, default=64)
    v6_parser.add_argument("--refined-margin-floor-bits", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.command in {"collect", "replay-preflight"}:
        if arguments.workers < 1:
            raise ValueError("workers must be positive")
    if arguments.command == "collect":
        collect(arguments.plan, arguments.output, arguments.estimator_url, arguments.workers)
    elif arguments.command == "replay-preflight":
        replay_preflight(
            arguments.input,
            arguments.output,
            arguments.estimator_url,
            arguments.workers,
        )
    elif arguments.command == "select-plan":
        select_plan(arguments.plan, arguments.input, arguments.output)
    elif arguments.command == "summarize":
        summarize(arguments.input, arguments.output, arguments.safety_cushion_bits)
    else:
        validate_arora_v6_local(
            arguments.input,
            arguments.output,
            arguments.targets,
            arguments.coarse_margin_floor_bits,
            arguments.refined_margin_floor_bits,
        )


if __name__ == "__main__":
    main()
