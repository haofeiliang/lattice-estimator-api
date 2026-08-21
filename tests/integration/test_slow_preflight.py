from __future__ import annotations

import pytest

from src.adapter import execute, execute_preflight
from src.models import EstimateRequest, PreflightRequest


@pytest.mark.integration
@pytest.mark.parametrize("sigma", ["0.3", "0.7", "1", "2", "3.2", "4"])
def test_large_dimension_gaussian_preflight_covers_calibration_range(sigma: str) -> None:
    request = PreflightRequest.model_validate(
        {
            "operation": "preflight",
            "schema_version": 2,
            "problem": {
                "kind": "lwe",
                "dimension": 1024,
                "modulus": "4096",
                "samples": {"kind": "unlimited"},
                "secret": {"kind": "uniform_binary"},
                "error": {
                    "kind": "discrete_gaussian",
                    "standard_deviation": sigma,
                },
            },
            "models": {"cost_model": "BDGL16", "shape_model": "GSA"},
            "target_attacks": ["arora_gb", "bkw"],
            "timeout_seconds": 300,
        }
    )
    response = execute_preflight(request)
    assert [result.attack.value for result in response.results] == ["arora_gb", "bkw"]
    assert response.results[1].outcome.kind == "computed"
    assert response.results[1].outcome.metrics["normalized_dimension"].value == "1024"
    assert "normalized_secret" in response.results[1].outcome.metrics
    assert "normalized_error" in response.results[1].outcome.metrics
    if float(sigma) <= 2:
        assert response.results[0].outcome.kind == "computed"
        assert response.results[0].outcome.metrics["preflight_tail"].kind == "integer"
        assert response.results[0].outcome.metrics["preflight_solving_degree"].kind == "integer"
        assert response.results[0].outcome.metrics["preflight_guessed_coordinates"].kind == (
            "integer"
        )
    else:
        # Instances without a solving degree are deliberately unknown; the
        # scheduler must run exact Arora-GB.
        assert response.results[0].outcome.kind == "no_finite_estimate"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("attack", "dimension", "sigma", "minimum_delta", "maximum_delta"),
    [
        ("arora_gb", 256, "0.9", -0.01, 0.01),
        ("bkw", 64, "1", 16, 20),
    ],
)
def test_preflight_unsafe_error_baselines(
    attack: str,
    dimension: int,
    sigma: str,
    minimum_delta: float,
    maximum_delta: float,
) -> None:
    payload = {
        "schema_version": 2,
        "problem": {
            "kind": "lwe",
            "dimension": dimension,
            "modulus": "256",
            "samples": {"kind": "unlimited"},
            "secret": {"kind": "uniform_binary"},
            "error": {"kind": "discrete_gaussian", "standard_deviation": sigma},
        },
        "models": {"cost_model": "BDGL16", "shape_model": "GSA"},
        "target_attacks": [attack],
        "timeout_seconds": 300,
    }
    preflight = execute_preflight(
        PreflightRequest.model_validate({**payload, "operation": "preflight"})
    )
    exact = execute(EstimateRequest.model_validate(payload))
    assert preflight.results[0].outcome.kind == "computed"
    assert exact.results[0].outcome.kind == "computed"
    delta = float(preflight.results[0].outcome.security_bits) - float(
        exact.results[0].outcome.security_bits
    )
    assert minimum_delta < delta < maximum_delta


@pytest.mark.integration
@pytest.mark.parametrize(
    ("dimension", "sigma", "expected_bits", "tail", "degree", "zeta"),
    [
        (1024, "0.7", 290.6484180132149, 10, 21, 1),
        (1024, "1.2", 583.737223642882, 22, 50, 10),
        (1024, "1.5", 804.0655176235326, 28, 78, 0),
        (768, "1.5", 667.4800317893055, 28, 67, 4),
        (1024, "0.3", 40.77307650579265, 2, 5, 983),
    ],
)
def test_arora_v2_matches_long_running_holdouts(
    dimension: int,
    sigma: str,
    expected_bits: float,
    tail: int,
    degree: int,
    zeta: int,
) -> None:
    request = PreflightRequest.model_validate(
        {
            "operation": "preflight",
            "schema_version": 2,
            "problem": {
                "kind": "lwe",
                "dimension": dimension,
                "modulus": "4096",
                "samples": {"kind": "unlimited"},
                "secret": {"kind": "uniform_binary"},
                "error": {
                    "kind": "discrete_gaussian",
                    "standard_deviation": sigma,
                },
            },
            "models": {"cost_model": "BDGL16", "shape_model": "GSA"},
            "target_attacks": ["arora_gb"],
            "timeout_seconds": 300,
        }
    )
    outcome = execute_preflight(request).results[0].outcome
    assert outcome.kind == "computed"
    # The v2 float/log-domain surrogate should track the pinned Sage result to
    # much less than one bit. A one-bit envelope allows harmless FP changes.
    assert abs(float(outcome.security_bits) - expected_bits) < 1
    assert outcome.metrics["preflight_tail"].value == str(tail)
    assert outcome.metrics["preflight_solving_degree"].value == str(degree)
    assert abs(int(outcome.metrics["preflight_guessed_coordinates"].value) - zeta) <= 1


def test_arora_v2_sparse_search_keeps_no_guessing_baseline() -> None:
    request = PreflightRequest.model_validate(
        {
            "operation": "preflight",
            "schema_version": 2,
            "problem": {
                "kind": "lwe",
                "dimension": 512,
                "modulus": "3329",
                "samples": {"kind": "unlimited"},
                "secret": {"kind": "fixed_weight_binary", "hamming_weight": 64},
                "error": {
                    "kind": "discrete_gaussian",
                    "standard_deviation": "0.7",
                },
            },
            "models": {"cost_model": "BDGL16", "shape_model": "GSA"},
            "target_attacks": ["arora_gb"],
            "timeout_seconds": 300,
        }
    )
    outcome = execute_preflight(request).results[0].outcome
    assert outcome.kind == "computed"
    # Pinned upstream exact result: 229.5464979950134 bits at zeta=0.
    assert abs(float(outcome.security_bits) - 229.5464979950134) < 0.01
    assert outcome.metrics["preflight_guessed_coordinates"].value == "0"
