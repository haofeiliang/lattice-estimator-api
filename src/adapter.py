"""Thin adapter over the pinned lattice-estimator API.

This module is imported only inside the killable Sage child process.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import (
    LWE_ATTACKS,
    NTRU_ATTACKS,
    Attack,
    AttackExecution,
    BooleanMetric,
    CenteredBinomial,
    ComputedOutcome,
    DecimalMetric,
    DiscreteGaussian,
    EstimateRequest,
    FailedOutcome,
    FixedWeightBinary,
    FixedWeightTernary,
    IntegerMetric,
    LweProblem,
    NoFiniteEstimateOutcome,
    NormalizedMetric,
    NtruProblem,
    SisNorm,
    SisProblem,
    SparseTernary,
    TextMetric,
    UniformBinary,
    UniformInteger,
    UniformTernary,
    WorkerResponse,
)

PUBLIC_TO_UPSTREAM = {
    Attack.ARORA_GB: "arora-gb",
    Attack.BKW: "bkw",
    Attack.USVP: "usvp",
    Attack.BDD: "bdd",
    Attack.BDD_HYBRID: "bdd_hybrid",
    Attack.BDD_MITM_HYBRID: "bdd_mitm_hybrid",
    Attack.DUAL: "dual",
    Attack.DUAL_HYBRID: "dual_hybrid",
    Attack.DSD: "dsd",
    Attack.LATTICE: "lattice",
}

# About 77 decimal digits. This guards numeric serialization at the unstable
# Sage boundary; it does not reduce lattice-estimator's model or cache identity.
NORMALIZATION_REAL_PRECISION_BITS = 256


def execute(request: EstimateRequest) -> WorkerResponse:
    """Run exactly the attacks requested by the Rust scheduler."""

    started = time.monotonic()
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(captured_stdout),
            contextlib.redirect_stderr(captured_stderr),
        ):
            raw_results = _run_estimator(request, request.target_attacks)
    except Exception as error:  # noqa: BLE001 - normalize the unstable upstream boundary
        audit = _audit_capture(captured_stdout, captured_stderr)
        audit["exception_type"] = type(error).__name__
        results = [
            AttackExecution(
                attack=attack,
                outcome=FailedOutcome(
                    kind="failed",
                    code="estimator_exception",
                    message=str(error) or type(error).__name__,
                    retryable=False,
                    raw_result=audit,
                ),
            )
            for attack in request.target_attacks
        ]
    else:
        audit = _audit_capture(captured_stdout, captured_stderr)
        results = [
            _normalize_attack(attack, raw_results, audit) for attack in request.target_attacks
        ]

    return WorkerResponse(
        results=results,
        duration_ms=max(0, round((time.monotonic() - started) * 1_000)),
    )


def _run_estimator(request: EstimateRequest, attacks: list[Attack]) -> dict[str, Any]:
    from estimator import LWE, NTRU, RC, SIS, Simulator  # type: ignore[import-not-found]

    cost_model = getattr(RC, request.models.cost_model.value)
    shape_model = getattr(Simulator, request.models.shape_model.value)
    requested = {PUBLIC_TO_UPSTREAM[attack] for attack in attacks}

    if isinstance(request.problem, LweProblem):
        from sage.all import oo  # type: ignore[import-not-found]

        problem = request.problem
        samples = oo if problem.samples.kind == "unlimited" else problem.samples.count
        params = LWE.Parameters(
            n=problem.dimension,
            q=int(problem.modulus),
            Xs=_distribution(problem.secret, problem.dimension),
            Xe=_distribution(problem.error, None),
            m=samples,
        )
        all_attacks = {PUBLIC_TO_UPSTREAM[item] for item in LWE_ATTACKS}
        return LWE.estimate(
            params,
            red_cost_model=cost_model,
            red_shape_model=shape_model,
            deny_list=tuple(sorted(all_attacks - requested)),
            jobs=1,
            catch_exceptions=True,
            quiet=True,
        )

    if isinstance(request.problem, NtruProblem):
        problem = request.problem
        params = NTRU.Parameters(
            n=problem.dimension,
            q=int(problem.modulus),
            Xs=_distribution(problem.secret, problem.dimension),
            Xe=_distribution(problem.error, None),
            m=problem.dimension,
            ntru_type=problem.structure.value,
        )
        all_attacks = {PUBLIC_TO_UPSTREAM[item] for item in NTRU_ATTACKS}
        return NTRU.estimate(
            params,
            red_cost_model=cost_model,
            red_shape_model=shape_model,
            deny_list=tuple(sorted(all_attacks - requested)),
            jobs=1,
            catch_exceptions=True,
            quiet=True,
        )

    if isinstance(request.problem, SisProblem):
        from sage.all import oo  # type: ignore[import-not-found]

        params = SIS.Parameters(
            n=request.problem.dimension,
            q=int(request.problem.modulus),
            length_bound=request.problem.length_bound,
            m=request.problem.columns,
            norm=2 if request.problem.norm is SisNorm.L2 else oo,
        )
        return SIS.estimate(
            params,
            red_cost_model=cost_model,
            red_shape_model=shape_model,
            deny_list=(),
            jobs=1,
            catch_exceptions=True,
            quiet=True,
        )

    raise AssertionError("strict request model admitted an unknown problem")


def _distribution(distribution: Any, logical_length: int | None) -> Any:
    from estimator import ND  # type: ignore[import-not-found]

    if isinstance(distribution, UniformBinary):
        return ND.Uniform(0, 1, n=logical_length)
    if isinstance(distribution, UniformTernary):
        return ND.Uniform(-1, 1, n=logical_length)
    if isinstance(distribution, SparseTernary):
        if logical_length is None:
            raise AssertionError("sparse ternary is only valid for a secret with a known length")
        # Primus defines sparse_ternary coefficient-wise with probabilities
        # 1/4, 1/2, 1/4. lattice-estimator only exposes a fixed-composition
        # SparseTernary, so use its balanced modal composition as the explicit
        # estimator model. The public distribution remains probabilistic.
        typical_sign_weight = (logical_length + 2) // 4
        return ND.SparseTernary(typical_sign_weight, typical_sign_weight, n=logical_length)
    if isinstance(distribution, FixedWeightBinary):
        return ND.SparseBinary(distribution.hamming_weight, n=logical_length)
    if isinstance(distribution, FixedWeightTernary):
        return ND.SparseTernary(
            distribution.positive_weight,
            distribution.negative_weight,
            n=logical_length,
        )
    if isinstance(distribution, DiscreteGaussian):
        return ND.DiscreteGaussian(distribution.standard_deviation, n=logical_length)
    if isinstance(distribution, CenteredBinomial):
        return ND.CenteredBinomial(distribution.eta, n=logical_length)
    if isinstance(distribution, UniformInteger):
        return ND.Uniform(int(distribution.lower), int(distribution.upper), n=logical_length)
    raise AssertionError(f"strict model admitted unsupported distribution {distribution.kind}")


def _normalize_attack(
    attack: Attack,
    raw_results: dict[str, Any],
    audit: dict[str, Any],
) -> AttackExecution:
    upstream_name = PUBLIC_TO_UPSTREAM[attack]
    raw = raw_results.get(upstream_name)
    if raw is None:
        return AttackExecution(
            attack=attack,
            outcome=FailedOutcome(
                kind="failed",
                code="estimator_no_result",
                message=f"estimator returned no result for {attack.value}",
                retryable=False,
                raw_result=audit,
            ),
        )

    rop = raw.get("rop") if hasattr(raw, "get") else None
    security_bits = _log2_cost(rop)
    if security_bits is None:
        return AttackExecution(
            attack=attack,
            outcome=NoFiniteEstimateOutcome(
                kind="no_finite_estimate",
                code="no_finite_rop",
                reason=f"{attack.value} returned no finite positive rop",
                raw_result={"result": _safe_text(raw), **audit},
            ),
        )

    metrics: dict[str, NormalizedMetric] = {}
    if hasattr(raw, "items"):
        for key, value in raw.items():
            if str(key) == "rop":
                continue
            metric = _normalize_metric(value)
            if metric is not None:
                metrics[str(key)] = metric
    return AttackExecution(
        attack=attack,
        outcome=ComputedOutcome(kind="computed", security_bits=security_bits, metrics=metrics),
    )


def _log2_cost(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value <= 0:
            return None
        from sage.all import RealField  # type: ignore[import-not-found]

        # ``value`` is commonly an exact Sage Integer. ``log(value, 2)`` then
        # remains a symbolic expression, whose string cannot be parsed as a
        # canonical decimal. Coerce the cost to a high-precision real first so
        # the logarithm is numeric. Checking the logarithm instead of
        # ``float(value)`` also preserves finite costs larger than the IEEE-754
        # exponent range.
        security_bits = RealField(NORMALIZATION_REAL_PRECISION_BITS)(value).log(2)
        if not math.isfinite(float(security_bits)):
            return None
        return _canonical_decimal(security_bits)
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None


def _normalize_metric(value: Any) -> NormalizedMetric | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return BooleanMetric(kind="boolean", value=value)
    if isinstance(value, int):
        return IntegerMetric(kind="integer", value=str(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return TextMetric(kind="text", value=_safe_text(value))
    if math.isfinite(numeric):
        return DecimalMetric(kind="decimal", value=_canonical_decimal(value))
    return TextMetric(kind="text", value=_safe_text(value))


def _canonical_decimal(value: Any) -> str:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        decimal = _evaluate_numeric_expression(value)
    if not decimal.is_finite():
        raise ValueError("decimal is not finite")
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        return "0"
    return text


def _evaluate_numeric_expression(value: Any) -> Decimal:
    """Evaluate finite Sage expressions before decimal serialization.

    Some estimator metrics are symbolic-looking but numerically evaluable, for
    example ``6.2175161e33*e^(-10)``.  Decimal intentionally does not parse such
    expressions, so ask Sage for a high-precision real value first.  The float
    fallback keeps unit tests and non-Sage tooling usable; security-bit values
    already arrive here as high-precision Sage reals.
    """

    try:
        from sage.all import RealField  # type: ignore[import-not-found]

        decimal = Decimal(str(RealField(NORMALIZATION_REAL_PRECISION_BITS)(value)))
    except (ImportError, InvalidOperation, TypeError, ValueError, OverflowError):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"cannot normalize finite decimal {value!r}") from error
        if not math.isfinite(numeric):
            raise ValueError("decimal is not finite") from None
        decimal = Decimal(repr(numeric))
    if not decimal.is_finite():
        raise ValueError("decimal is not finite")
    return decimal


def _audit_capture(stdout: io.StringIO, stderr: io.StringIO) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if stdout.getvalue():
        result["stdout"] = stdout.getvalue()[-16_384:]
    if stderr.getvalue():
        result["stderr"] = stderr.getvalue()[-16_384:]
    return result


def _safe_text(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - audit normalization must not mask the primary error
        return f"<{type(value).__name__}>"
