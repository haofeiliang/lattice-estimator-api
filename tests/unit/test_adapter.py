from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest

from src.adapter import (
    _canonical_decimal,
    _distribution,
    _log2_cost,
    _normalize_attack,
    _run_estimator,
    execute,
)
from src.models import (
    Attack,
    CenteredBinomial,
    DiscreteGaussian,
    FixedWeightBinary,
    FixedWeightTernary,
    SparseTernary,
    UniformBinary,
    UniformInteger,
    UniformTernary,
)
from tests.unit.test_models import request_model


class FakeDistributionFactory:
    @staticmethod
    def Uniform(lower: int, upper: int, *, n: int | None) -> tuple[object, ...]:
        return ("uniform", lower, upper, n)

    @staticmethod
    def SparseBinary(weight: int, *, n: int | None) -> tuple[object, ...]:
        return ("sparse_binary", weight, n)

    @staticmethod
    def SparseTernary(positive: int, negative: int, *, n: int | None) -> tuple[object, ...]:
        return ("sparse_ternary", positive, negative, n)

    @staticmethod
    def DiscreteGaussian(standard_deviation: str, *, n: int | None) -> tuple[object, ...]:
        return ("discrete_gaussian", standard_deviation, n)

    @staticmethod
    def CenteredBinomial(eta: int, *, n: int | None) -> tuple[object, ...]:
        return ("centered_binomial", eta, n)


class FakeLwe:
    deny_list: tuple[str, ...] | None = None
    raw_results: ClassVar[dict[str, object]] = {}

    class Parameters:
        def __init__(self, **values: object) -> None:
            self.values = values

    @classmethod
    def estimate(cls, _parameters: object, **options: object) -> dict[str, object]:
        cls.deny_list = options["deny_list"]  # type: ignore[assignment]
        return cls.raw_results


class FakeReal:
    def __init__(self, value: object) -> None:
        self.value = value

    def log(self, base: int) -> float:
        import math

        assert base == 2
        return math.log2(self.value)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return repr(float(self.value))


class FakeRealField:
    def __init__(self, _precision: int) -> None:
        pass

    def __call__(self, value: object) -> FakeReal:
        return FakeReal(value)


@pytest.fixture(autouse=True)
def fake_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeLwe.raw_results = {}
    monkeypatch.setitem(
        sys.modules,
        "estimator",
        SimpleNamespace(
            ND=FakeDistributionFactory,
            LWE=FakeLwe,
            NTRU=object(),
            SIS=object(),
            RC=SimpleNamespace(BDGL16=object()),
            Simulator=SimpleNamespace(GSA=object()),
        ),
    )
    monkeypatch.setitem(sys.modules, "sage", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "sage.all",
        SimpleNamespace(oo=object(), RealField=FakeRealField),
    )


@pytest.mark.parametrize(
    ("model", "length", "expected"),
    [
        (UniformBinary(kind="uniform_binary"), 8, ("uniform", 0, 1, 8)),
        (UniformTernary(kind="uniform_ternary"), 9, ("uniform", -1, 1, 9)),
        (
            SparseTernary(kind="sparse_ternary"),
            10,
            ("sparse_ternary", 3, 3, 10),
        ),
        (
            FixedWeightBinary(kind="fixed_weight_binary", hamming_weight=3),
            10,
            ("sparse_binary", 3, 10),
        ),
        (
            FixedWeightTernary(kind="fixed_weight_ternary", positive_weight=2, negative_weight=3),
            12,
            ("sparse_ternary", 2, 3, 12),
        ),
        (
            DiscreteGaussian(kind="discrete_gaussian", standard_deviation="3.25"),
            None,
            ("discrete_gaussian", "3.25", None),
        ),
        (
            CenteredBinomial(kind="centered_binomial", eta=4),
            None,
            ("centered_binomial", 4, None),
        ),
        (
            UniformInteger(kind="uniform_integer", lower="-2", upper="5"),
            None,
            ("uniform", -2, 5, None),
        ),
    ],
)
def test_distribution_mapping_preserves_parameters(
    model: object, length: int | None, expected: tuple[object, ...]
) -> None:
    assert _distribution(model, length) == expected


def test_canonical_decimal_does_not_round_through_binary_float() -> None:
    assert _canonical_decimal("12345678901234567890.123456789") == (
        "12345678901234567890.123456789"
    )
    assert _canonical_decimal("1.2300") == "1.23"
    assert _canonical_decimal("-0.0") == "0"


def test_canonical_decimal_evaluates_symbolic_sage_metric() -> None:
    class SymbolicMetric:
        def __str__(self) -> str:
            return "6.2175161e33*e^(-10)"

        def __float__(self) -> float:
            import math

            return 6.2175161e33 * math.exp(-10)

    assert _canonical_decimal(SymbolicMetric()) == "282274794237118760000000000000"


def test_finite_exact_rop_is_converted_to_numeric_security_bits() -> None:
    assert _log2_cost(2**40) == "40"
    assert _log2_cost(2**4096) == "4096"

    execution = _normalize_attack(
        Attack.BKW,
        {"bkw": {"rop": 2**40, "tag": "coded-bkw"}},
        {},
    )
    assert execution.outcome.kind == "computed"
    assert execution.outcome.security_bits == "40"


def test_fast_lwe_plan_denies_slow_upstream_attacks() -> None:
    request = request_model()
    _run_estimator(request, request.target_attacks)
    assert FakeLwe.deny_list == ("arora-gb", "bkw")


def test_multi_attack_execution_reports_shared_request_timing() -> None:
    request = request_model()
    FakeLwe.raw_results = {
        {Attack.ARORA_GB: "arora-gb"}.get(attack, attack.value): {"rop": 2**40}
        for attack in request.target_attacks
    }
    response = execute(request)

    assert response.duration_ms >= 0
    assert all(result.outcome.kind == "computed" for result in response.results)
    assert all(result.duration_ms == response.duration_ms for result in response.results)
    assert all(result.duration_scope == "request_group" for result in response.results)
    assert all(
        result.shared_attacks == request_model().target_attacks for result in response.results
    )


def test_single_attack_no_finite_result_has_independent_timing() -> None:
    request = request_model().model_copy(update={"target_attacks": [Attack.BKW]})
    FakeLwe.raw_results = {"bkw": {"rop": float("inf")}}

    response = execute(request)

    assert response.results[0].outcome.kind == "no_finite_estimate"
    assert response.results[0].duration_ms == response.duration_ms
    assert response.results[0].duration_scope == "attack"
    assert response.results[0].shared_attacks == []


def test_estimator_failure_has_independent_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_class: type[FakeLwe], _parameters: object, **_options: object) -> dict[str, object]:
        raise RuntimeError("mock estimator failure")

    monkeypatch.setattr(FakeLwe, "estimate", classmethod(fail))
    request = request_model().model_copy(update={"target_attacks": [Attack.BKW]})

    response = execute(request)

    assert response.results[0].outcome.kind == "failed"
    assert response.results[0].duration_ms == response.duration_ms
    assert response.results[0].duration_scope == "attack"


def test_adaptive_slow_plan_remains_callable() -> None:
    request = request_model().model_copy(update={"target_attacks": [Attack.ARORA_GB, Attack.BKW]})
    _run_estimator(request, request.target_attacks)
    assert FakeLwe.deny_list == (
        "bdd",
        "bdd_hybrid",
        "bdd_mitm_hybrid",
        "dual",
        "dual_hybrid",
        "usvp",
    )


def test_infinite_rop_is_a_terminal_auditable_estimate() -> None:
    execution = _normalize_attack(
        Attack.DUAL_HYBRID,
        {"dual_hybrid": {"rop": float("inf"), "tag": "test"}},
        {"stderr": "estimator diagnostic"},
    )

    assert execution.outcome.kind == "no_finite_estimate"
    assert execution.outcome.code == "no_finite_rop"
    assert execution.outcome.raw_result == {
        "result": "{'rop': inf, 'tag': 'test'}",
        "stderr": "estimator diagnostic",
    }
