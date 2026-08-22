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
            "required_security_bits": "128",
            "requested_arora_gb_coarse_margin_bits": "64",
            "requested_arora_gb_refined_margin_bits": "10",
            "schema_version": 5,
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
            "cost_model": "BDGL16",
            "target_attacks": ["arora_gb", "bkw"],
            "timeout_seconds": 300,
        }
    )
    response = execute_preflight(request)
    assert [result.attack.value for result in response.results] == ["arora_gb", "bkw"]
    assert all(result.duration_ms >= 0 for result in response.results)
    assert all(result.duration_scope == "attack" for result in response.results)
    assert all(result.shared_attacks == [] for result in response.results)
    assert response.results[1].outcome.kind == "computed"
    assert response.results[1].outcome.metrics["preflight_model"].value == ("coded_bkw_structural")
    assert response.results[1].outcome.metrics["preflight_b"].kind == "integer"
    assert response.results[1].outcome.metrics["preflight_t1"].kind == "integer"
    assert response.results[1].outcome.metrics["preflight_t2"].kind == "integer"
    assert response.results[1].outcome.metrics["preflight_rule_version"].value == "5"
    assert response.results[1].outcome.metrics["normalized_dimension"].value == "1024"
    assert "normalized_secret" in response.results[1].outcome.metrics
    assert "normalized_error" in response.results[1].outcome.metrics
    arora = response.results[0].outcome
    assert arora.kind == "threshold_screen"
    assert arora.decision in {"above_threshold", "needs_exact"}
    assert arora.precision_tier in {"coarse", "refined"}
    assert arora.metrics["preflight_rule_version"].value == "6"
    assert "preflight_max_degree_checked" in arora.metrics
    if sigma == "3.2":
        assert abs(float(response.results[1].outcome.security_bits) - 244.116929779912) < 0.01


@pytest.mark.integration
def test_bkw_finite_sample_amplification_matches_exact_structure() -> None:
    request = PreflightRequest.model_validate(
        {
            "operation": "preflight",
            "required_security_bits": "128",
            "requested_arora_gb_coarse_margin_bits": "64",
            "requested_arora_gb_refined_margin_bits": "10",
            "schema_version": 5,
            "problem": {
                "kind": "lwe",
                "dimension": 128,
                "modulus": "3329",
                "samples": {"kind": "finite", "count": 4096},
                "secret": {"kind": "uniform_ternary"},
                "error": {"kind": "discrete_gaussian", "standard_deviation": "1.5"},
            },
            "cost_model": "BDGL16",
            "target_attacks": ["bkw"],
            "timeout_seconds": 300,
        }
    )
    execution = execute_preflight(request).results[0]
    outcome = execution.outcome
    assert execution.duration_ms >= 0
    assert execution.duration_scope == "attack"
    assert outcome.kind == "computed"
    assert abs(float(outcome.security_bits) - 58.38753990517848) < 0.01
    assert outcome.metrics["preflight_sample_additions"].value == "5"
    assert abs(float(outcome.metrics["preflight_effective_error_stddev"].value) - 3.3541) < 0.001


@pytest.mark.integration
@pytest.mark.parametrize(
    "error",
    [
        {"kind": "centered_binomial", "eta": 1},
        {"kind": "uniform_integer", "lower": "-1", "upper": "1"},
    ],
)
@pytest.mark.parametrize("attack", ["arora_gb", "bkw"])
def test_reviewed_unlimited_bounded_preflight_tracks_exact(
    attack: str, error: dict[str, object]
) -> None:
    payload = {
        "schema_version": 5,
        "problem": {
            "kind": "lwe",
            "dimension": 64,
            "modulus": "256",
            "samples": {"kind": "unlimited"},
            "secret": {"kind": "uniform_binary"},
            "error": error,
        },
        "cost_model": "BDGL16",
        "target_attacks": [attack],
        "timeout_seconds": 300,
    }
    preflight = (
        execute_preflight(
            PreflightRequest.model_validate(
                {
                    **payload,
                    "operation": "preflight",
                    "required_security_bits": "128",
                    "requested_arora_gb_coarse_margin_bits": "64",
                    "requested_arora_gb_refined_margin_bits": "10",
                }
            )
        )
        .results[0]
        .outcome
    )
    exact = execute(EstimateRequest.model_validate(payload)).results[0].outcome
    assert exact.kind == "computed"
    if attack == "arora_gb":
        assert preflight.kind == "threshold_screen"
        assert preflight.metrics["preflight_rule_version"].value == "6"
        if preflight.decision == "above_threshold":
            assert float(exact.security_bits) >= 128
    else:
        assert preflight.kind == "computed"
        assert preflight.metrics["preflight_rule_version"].value == "5"
        assert abs(float(preflight.security_bits) - float(exact.security_bits)) < 1


@pytest.mark.integration
@pytest.mark.parametrize(
    ("secret", "error", "expected_bits", "zeta", "degree"),
    [
        (
            {"kind": "fixed_weight_binary", "hamming_weight": 64},
            {"kind": "centered_binomial", "eta": 8},
            236.8962420950805,
            65,
            33,
        ),
        (
            {"kind": "uniform_ternary"},
            {"kind": "centered_binomial", "eta": 1},
            101.72389331798854,
            3,
            10,
        ),
        (
            {"kind": "uniform_ternary"},
            {"kind": "uniform_integer", "lower": "-1", "upper": "1"},
            98.99302894442379,
            1,
            10,
        ),
    ],
)
def test_bounded_arora_guess_composition_matches_finite_sample_holdouts(
    secret: dict[str, object],
    error: dict[str, object],
    expected_bits: float,
    zeta: int,
    degree: int,
) -> None:
    request = PreflightRequest.model_validate(
        {
            "operation": "preflight",
            "required_security_bits": "128",
            "requested_arora_gb_coarse_margin_bits": "64",
            "requested_arora_gb_refined_margin_bits": "10",
            "schema_version": 5,
            "problem": {
                "kind": "lwe",
                "dimension": 128,
                "modulus": "4093",
                "samples": {"kind": "finite", "count": 4096},
                "secret": secret,
                "error": error,
            },
            "cost_model": "BDGL16",
            "target_attacks": ["arora_gb"],
            "timeout_seconds": 300,
        }
    )
    outcome = execute_preflight(request).results[0].outcome
    assert outcome.kind == "threshold_screen"
    assert outcome.metrics["preflight_rule_version"].value == "6"
    assert outcome.decision == ("above_threshold" if expected_bits >= 128 else "needs_exact")
    assert int(outcome.metrics["preflight_max_degree_checked"].value) < 64


@pytest.mark.integration
@pytest.mark.parametrize(
    ("attack", "dimension", "sigma", "minimum_delta", "maximum_delta"),
    [
        ("arora_gb", 256, "0.9", -0.01, 0.01),
        ("bkw", 64, "1", -0.01, 0.01),
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
        "schema_version": 5,
        "problem": {
            "kind": "lwe",
            "dimension": dimension,
            "modulus": "256",
            "samples": {"kind": "unlimited"},
            "secret": {"kind": "uniform_binary"},
            "error": {"kind": "discrete_gaussian", "standard_deviation": sigma},
        },
        "cost_model": "BDGL16",
        "target_attacks": [attack],
        "timeout_seconds": 300,
    }
    preflight = execute_preflight(
        PreflightRequest.model_validate(
            {
                **payload,
                "operation": "preflight",
                "required_security_bits": "128",
                "requested_arora_gb_coarse_margin_bits": "64",
                "requested_arora_gb_refined_margin_bits": "10",
            }
        )
    )
    exact = execute(EstimateRequest.model_validate(payload))
    assert exact.results[0].outcome.kind == "computed"
    if attack == "arora_gb":
        assert preflight.results[0].outcome.kind == "threshold_screen"
        if preflight.results[0].outcome.decision == "above_threshold":
            assert float(exact.results[0].outcome.security_bits) >= 128
    else:
        assert preflight.results[0].outcome.kind == "computed"
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
            "required_security_bits": "128",
            "requested_arora_gb_coarse_margin_bits": "64",
            "requested_arora_gb_refined_margin_bits": "10",
            "schema_version": 5,
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
            "cost_model": "BDGL16",
            "target_attacks": ["arora_gb"],
            "timeout_seconds": 300,
        }
    )
    outcome = execute_preflight(request).results[0].outcome
    assert outcome.kind == "threshold_screen"
    assert outcome.decision == ("above_threshold" if expected_bits >= 128 else "needs_exact")
    assert int(outcome.metrics["preflight_max_degree_checked"].value) < 64


def test_arora_v2_sparse_search_keeps_no_guessing_baseline() -> None:
    request = PreflightRequest.model_validate(
        {
            "operation": "preflight",
            "required_security_bits": "128",
            "requested_arora_gb_coarse_margin_bits": "64",
            "requested_arora_gb_refined_margin_bits": "10",
            "schema_version": 5,
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
            "cost_model": "BDGL16",
            "target_attacks": ["arora_gb"],
            "timeout_seconds": 300,
        }
    )
    outcome = execute_preflight(request).results[0].outcome
    assert outcome.kind == "threshold_screen"
    assert outcome.decision == "above_threshold"
    assert float(outcome.decision_threshold_bits) == 192


@pytest.mark.integration
def test_first_bsk_v6_stops_well_before_the_exact_solving_degree() -> None:
    request = PreflightRequest.model_validate(
        {
            "operation": "preflight",
            "required_security_bits": "128",
            "requested_arora_gb_coarse_margin_bits": "64",
            "requested_arora_gb_refined_margin_bits": "10",
            "schema_version": 5,
            "problem": {
                "kind": "lwe",
                "dimension": 1024,
                "modulus": "67104769",
                "samples": {"kind": "unlimited"},
                "secret": {"kind": "sparse_ternary"},
                "error": {
                    "kind": "discrete_gaussian",
                    "standard_deviation": "3.19",
                },
            },
            "cost_model": "BDGL16",
            "target_attacks": ["arora_gb"],
            "timeout_seconds": 300,
        }
    )
    execution = execute_preflight(request).results[0]
    outcome = execution.outcome
    assert outcome.kind == "threshold_screen"
    assert outcome.decision == "above_threshold"
    assert outcome.precision_tier == "coarse"
    assert outcome.decision_threshold_bits == "192"
    assert int(outcome.metrics["preflight_max_degree_checked"].value) <= 12
    # The worker enforces a four-second wall-clock budget. Allow a small amount
    # of scheduler and signal-delivery jitter on slower CI hosts.
    assert execution.duration_ms < 4_500
