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
