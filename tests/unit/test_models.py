from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.models import Attack, EstimateRequest, attacks_for_problem


def request_data() -> dict[str, object]:
    return {
        "schema_version": 2,
        "problem": {
            "kind": "lwe",
            "dimension": 512,
            "modulus": "65536",
            "samples": {"kind": "unlimited"},
            "secret": {
                "kind": "fixed_weight_ternary",
                "positive_weight": 128,
                "negative_weight": 128,
            },
            "error": {"kind": "discrete_gaussian", "standard_deviation": "16"},
        },
        "models": {
            "cost_model": "BDGL16",
            "shape_model": "GSA",
        },
        "target_attacks": [
            "usvp",
            "bdd",
            "bdd_hybrid",
            "bdd_mitm_hybrid",
            "dual",
            "dual_hybrid",
        ],
        "timeout_seconds": 5,
    }


def request_model(*, timeout_seconds: int = 5) -> EstimateRequest:
    source = request_data()
    source["timeout_seconds"] = timeout_seconds
    return EstimateRequest.model_validate_json(json.dumps(source))


def test_lwe_exposes_fast_and_adaptive_slow_attacks() -> None:
    request = request_model()
    assert attacks_for_problem(request.problem) == (
        Attack.ARORA_GB,
        Attack.BKW,
        Attack.USVP,
        Attack.BDD,
        Attack.BDD_HYBRID,
        Attack.BDD_MITM_HYBRID,
        Attack.DUAL,
        Attack.DUAL_HYBRID,
    )
    assert Attack("arora_gb") is Attack.ARORA_GB
    assert Attack("bkw") is Attack.BKW


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("problem", "modulus"), 65536),
        (("problem", "error", "standard_deviation"), "1e-3"),
        (("problem", "dimension"), "512"),
    ],
)
def test_strict_protocol_rejects_coercion_and_exponents(
    path: tuple[str, ...], value: object
) -> None:
    source = request_data()
    target = source
    for component in path[:-1]:
        target = target[component]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValidationError):
        EstimateRequest.model_validate_json(json.dumps(source))


def test_unknown_fields_are_rejected() -> None:
    source = request_data()
    source["unexpected"] = True
    with pytest.raises(ValidationError):
        EstimateRequest.model_validate_json(json.dumps(source))

    source = request_data()
    source["target_attacks"] = ["bkw", "bkw"]
    with pytest.raises(ValidationError, match="duplicates"):
        EstimateRequest.model_validate_json(json.dumps(source))


def test_fixed_weight_uses_complete_logical_secret_length() -> None:
    source = request_data()
    source["problem"]["secret"] = {  # type: ignore[index]
        "kind": "fixed_weight_ternary",
        "positive_weight": 300,
        "negative_weight": 300,
    }
    with pytest.raises(ValidationError, match="logical secret length"):
        EstimateRequest.model_validate_json(json.dumps(source))


def test_sparse_ternary_has_fixed_coefficient_probabilities() -> None:
    source = request_data()
    source["problem"]["secret"] = {  # type: ignore[index]
        "kind": "sparse_ternary",
    }
    request = EstimateRequest.model_validate_json(json.dumps(source))
    assert request.problem.secret.kind == "sparse_ternary"

    source["problem"]["secret"]["positive_count"] = 128  # type: ignore[index]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EstimateRequest.model_validate_json(json.dumps(source))


def test_ntru_has_fixed_standard_attack_set() -> None:
    source = request_data()
    source["problem"] = {
        "kind": "ntru",
        "dimension": 1024,
        "modulus": "132120577",
        "secret": {"kind": "discrete_gaussian", "standard_deviation": "6.88"},
        "error": {"kind": "discrete_gaussian", "standard_deviation": "6.88"},
        "structure": "matrix",
    }
    source["target_attacks"] = ["usvp", "dsd", "bdd", "bdd_hybrid", "bdd_mitm_hybrid"]
    request = EstimateRequest.model_validate_json(json.dumps(source))
    assert attacks_for_problem(request.problem) == (
        Attack.USVP,
        Attack.DSD,
        Attack.BDD,
        Attack.BDD_HYBRID,
        Attack.BDD_MITM_HYBRID,
    )
