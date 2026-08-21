from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def load_tool() -> Any:
    path = Path(__file__).parents[2] / "tools" / "slow_attack_calibration.py"
    spec = importlib.util.spec_from_file_location("slow_attack_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()


def test_checked_in_plan_covers_requested_arora_range() -> None:
    root = Path(__file__).parents[2]
    plan = json.loads(
        (root / "calibration" / "plans" / "slow-attacks-v1.json").read_text(encoding="utf-8")
    )
    requests = list(tool.requests_from_plan(plan))
    assert min(item["problem"]["dimension"] for item in requests) == 64
    assert max(item["problem"]["dimension"] for item in requests) == 1024
    assert max(int(item["problem"]["modulus"]) for item in requests) == 4096
    sigmas = {
        item["problem"]["error"]["standard_deviation"]
        for item in requests
        if item["target_attacks"] == ["arora_gb"]
    }
    assert {"0.3", "0.7", "1", "2", "3.2", "4"} <= sigmas


def test_bkw_plans_cover_large_dimensions_finite_samples_and_secret_diversity() -> None:
    root = Path(__file__).parents[2]
    diversity = json.loads(
        (root / "calibration" / "plans" / "bkw-structural-diversity-v1.json").read_text(
            encoding="utf-8"
        )
    )
    diversity_requests = list(tool.requests_from_plan(diversity))
    assert any(item["problem"]["samples"]["kind"] == "finite" for item in diversity_requests)
    assert {item["problem"]["secret"]["kind"] for item in diversity_requests} == {
        "uniform_ternary",
        "fixed_weight_binary",
        "discrete_gaussian",
    }

    large = json.loads(
        (root / "calibration" / "plans" / "bkw-large-runtime-v1.json").read_text(encoding="utf-8")
    )
    large_requests = list(tool.requests_from_plan(large))
    assert {item["problem"]["dimension"] for item in large_requests} == {2048, 4096}
    assert max(int(item["problem"]["modulus"]) for item in large_requests) == 2**32


def test_v2_bounded_error_plans_use_complete_error_objects() -> None:
    root = Path(__file__).parents[2]
    plan = json.loads(
        (root / "calibration" / "plans" / "bounded-errors-v2.json").read_text(encoding="utf-8")
    )
    requests = list(tool.requests_from_plan(plan))
    assert len(requests) == 258
    assert {item["problem"]["error"]["kind"] for item in requests} == {
        "centered_binomial",
        "uniform_integer",
    }
    assert max(item["problem"]["dimension"] for item in requests) == 1024
    assert max(int(item["problem"]["modulus"]) for item in requests) == 4096
    assert any(
        item["problem"]["error"] == {"kind": "uniform_integer", "lower": "-8", "upper": "8"}
        for item in requests
    )

    finite_plan = json.loads(
        (root / "calibration" / "plans" / "arora-bounded-finite-validation-v2.json").read_text(
            encoding="utf-8"
        )
    )
    finite_requests = list(tool.requests_from_plan(finite_plan))
    assert len(finite_requests) == 80
    assert {item["problem"]["dimension"] for item in finite_requests} == {64, 256, 512}
    assert {item["problem"]["samples"]["count"] for item in finite_requests} == {4096, 65536}
    assert {item["problem"]["secret"]["kind"] for item in finite_requests} == {
        "uniform_binary",
        "fixed_weight_binary",
        "fixed_weight_ternary",
        "discrete_gaussian",
    }


def test_reviewed_v5_baseline_uses_structural_guessing_and_one_margin_per_attack() -> None:
    root = Path(__file__).parents[2]
    baseline = json.loads(
        (root / "calibration" / "baselines" / "slow-attacks-v5.json").read_text(encoding="utf-8")
    )
    assert baseline["provenance"]["preflight_rule_version"] == 5
    assert baseline["method"]["fitted_coefficients"] is False
    assert baseline["attacks"]["arora_gb"]["production_margin_floor_bits"] == 10
    assert baseline["attacks"]["bkw"]["production_margin_floor_bits"] == 10
    assert baseline["attacks"]["bkw"]["maximum_unsafe_error_bits"] < 0.4
    assert baseline["attacks"]["arora_gb"]["fixed_holdout"]["rule_v5_unsafe_error_bits"] < 1e-12


def test_summary_uses_maximum_unsafe_quick_error_plus_cushion(tmp_path: Path) -> None:
    rows = [
        {
            "request": {"target_attacks": ["arora_gb"]},
            "preflight": {"kind": "computed", "security_bits": "120"},
            "exact": {"kind": "computed", "security_bits": "100"},
        },
        {
            "request": {"target_attacks": ["arora_gb"]},
            "preflight": {"kind": "computed", "security_bits": "90"},
            "exact": {"kind": "computed", "security_bits": "100"},
        },
    ]
    source = tmp_path / "observations.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "summary.json"
    tool.summarize([source], output, cushion=8)
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["attacks"]["arora_gb"]["maximum_unsafe_error_bits"] == 20
    assert summary["attacks"]["arora_gb"]["recommended_margin_bits"] == 28


def test_summary_uses_latest_observation_for_retried_identity(tmp_path: Path) -> None:
    rows = [
        {
            "identity": "same",
            "request": {"target_attacks": ["bkw"]},
            "preflight": {"kind": "collection_failed"},
            "exact": {"kind": "collection_failed"},
        },
        {
            "identity": "same",
            "request": {"target_attacks": ["bkw"]},
            "preflight": {"kind": "computed", "security_bits": "100"},
            "exact": {"kind": "computed", "security_bits": "99"},
        },
    ]
    source = tmp_path / "observations.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "summary.json"
    tool.summarize([source], output, cushion=8)
    attack = json.loads(output.read_text(encoding="utf-8"))["attacks"]["bkw"]
    assert attack["observations"] == 1
    assert attack["comparable"] == 1
    assert attack["preflight_outcomes"] == {"computed": 1}


def test_select_plan_materializes_latest_retry_only(tmp_path: Path) -> None:
    payload = {
        "schema_version": 2,
        "problem": {
            "kind": "lwe",
            "dimension": 64,
            "modulus": "17",
            "samples": {"kind": "unlimited"},
            "secret": {"kind": "uniform_binary"},
            "error": {"kind": "centered_binomial", "eta": 1},
        },
        "models": {"cost_model": "BDGL16", "shape_model": "GSA"},
        "target_attacks": ["bkw"],
        "timeout_seconds": 30,
    }
    plan = {
        "format": tool.PLAN_FORMAT,
        "version": 2,
        "timeout_seconds": 30,
        "models": [payload["models"]],
        "experiments": [
            {
                "attack": "bkw",
                "axes": {
                    "dimension": [64],
                    "modulus": ["17"],
                    "samples": [{"kind": "unlimited"}],
                    "secret": [{"kind": "uniform_binary"}],
                    "error": [{"kind": "centered_binomial", "eta": 1}],
                },
            }
        ],
    }
    identity = tool._identity(payload)
    rows = [
        {"identity": identity, "request": payload, "exact": {"kind": "collection_failed"}},
        {"identity": identity, "request": payload, "exact": {"kind": "computed"}},
    ]
    plan_path = tmp_path / "plan.json"
    source = tmp_path / "source.jsonl"
    output = tmp_path / "selected.jsonl"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    tool.select_plan(plan_path, source, output)
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["exact"]["kind"] == "computed"


def test_collector_preserves_preflight_and_exact_durations(monkeypatch: Any) -> None:
    responses = iter(
        [
            {
                "results": [{"outcome": {"kind": "computed", "security_bits": "120"}}],
                "duration_ms": 7,
                "provenance": {"estimator_commit": "test"},
            },
            {
                "results": [{"outcome": {"kind": "computed", "security_bits": "100"}}],
                "duration_ms": 31,
                "provenance": {"estimator_commit": "test"},
            },
        ]
    )
    monkeypatch.setattr(tool, "_post", lambda _url, _payload, **_kwargs: next(responses))
    row = tool._collect_one(
        {
            "schema_version": 2,
            "problem": {},
            "models": {},
            "target_attacks": ["arora_gb"],
            "timeout_seconds": 30,
        },
        "http://127.0.0.1:8000",
    )
    assert row["preflight_duration_ms"] == 7
    assert row["exact_duration_ms"] == 31
