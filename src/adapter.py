"""Boundary between the service protocol and estimation implementations.

This module is imported only inside the killable Sage child process.  There are
two public execution paths:

* :func:`execute` calls the installed ``lattice-estimator`` package for exact
  attacks, then converts its unstable/raw output to the public response model.
* :func:`execute_preflight` runs the service's fast Arora-GB and BKW screening
  algorithms; it does not execute the corresponding upstream attacks.

The direct upstream calls are centralized in :func:`_run_estimator`, while
:func:`_lwe_parameters` and :func:`_distribution` translate public parameters to
upstream Sage objects.
"""

from __future__ import annotations

import contextlib
import io
import math
import signal
import threading
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
    PreflightRequest,
    PreflightUnknownOutcome,
    SisNorm,
    SisProblem,
    SparseTernary,
    TextMetric,
    ThresholdScreenOutcome,
    UniformBinary,
    UniformInteger,
    UniformTernary,
    WorkerResponse,
)
from .slow_estimate import (
    ARORA_GB_PREFLIGHT_RULE_VERSION,
    ARORA_TOTAL_BUDGET_SECONDS,
    BKW_PREFLIGHT_RULE_VERSION,
    AroraScreenDeadline,
    arora_gb_threshold_screen,
    bkw_estimate,
)

# Keep upstream spellings at this boundary.  Public names remain stable even if
# lattice-estimator changes display names or uses hyphens internally.
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
    """Run exact upstream attacks requested by the caller and normalize results."""

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

    duration_ms = max(0, round((time.monotonic() - started) * 1_000))
    duration_scope = "attack" if len(request.target_attacks) == 1 else "request_group"
    shared_attacks = [] if duration_scope == "attack" else request.target_attacks
    return WorkerResponse(
        results=[
            result.model_copy(
                update={
                    "duration_ms": duration_ms,
                    "duration_scope": duration_scope,
                    "shared_attacks": shared_attacks,
                }
            )
            for result in results
        ],
        duration_ms=duration_ms,
    )


def execute_preflight(request: PreflightRequest) -> WorkerResponse:
    """Run local fast screens that decide whether slow exact attacks are needed."""
    started = time.monotonic()
    params = _lwe_parameters(request.problem)
    normalized = params.normalize()
    normalized_audit = {
        "dimension": str(normalized.n),
        "modulus": str(normalized.q),
        "samples": str(normalized.m),
        "secret": str(normalized.Xs),
        "error": str(normalized.Xe),
    }
    normalized_metrics: dict[str, NormalizedMetric] = {
        "normalized_dimension": IntegerMetric(kind="integer", value=str(normalized.n)),
        "normalized_modulus": IntegerMetric(kind="integer", value=str(normalized.q)),
        "normalized_samples": TextMetric(kind="text", value=str(normalized.m)),
        "normalized_secret": TextMetric(kind="text", value=str(normalized.Xs)),
        "normalized_error": TextMetric(kind="text", value=str(normalized.Xe)),
    }
    results: list[AttackExecution] = []
    for attack in request.target_attacks:
        attack_started = time.monotonic()
        if attack is Attack.ARORA_GB:
            metrics = {
                **normalized_metrics,
                "preflight_rule_version": IntegerMetric(
                    kind="integer", value=str(ARORA_GB_PREFLIGHT_RULE_VERSION)
                ),
            }
            try:
                with _arora_alarm(ARORA_TOTAL_BUDGET_SECONDS):
                    screen = arora_gb_threshold_screen(
                        params,
                        float(request.required_security_bits),
                        float(request.requested_arora_gb_coarse_margin_bits),
                        float(request.requested_arora_gb_refined_margin_bits),
                        started=attack_started,
                    )
            except AroraScreenDeadline:
                screen = None
            if screen is None:
                outcome = ThresholdScreenOutcome(
                    kind="threshold_screen",
                    decision="needs_exact",
                    precision_tier="refined",
                    required_security_bits=request.required_security_bits,
                    requested_margin_bits=request.requested_arora_gb_refined_margin_bits,
                    calibrated_margin_floor_bits="10",
                    effective_margin_bits=_canonical_decimal(
                        max(float(request.requested_arora_gb_refined_margin_bits), 10.0)
                    ),
                    decision_threshold_bits=_canonical_decimal(
                        float(request.required_security_bits)
                        + max(float(request.requested_arora_gb_refined_margin_bits), 10.0)
                    ),
                    reason="time_budget_exhausted",
                    metrics=metrics,
                )
            else:
                outcome = ThresholdScreenOutcome(
                    kind="threshold_screen",
                    decision=screen.decision,
                    precision_tier=screen.precision_tier,
                    required_security_bits=request.required_security_bits,
                    requested_margin_bits=(
                        request.requested_arora_gb_coarse_margin_bits
                        if screen.precision_tier == "coarse"
                        else request.requested_arora_gb_refined_margin_bits
                    ),
                    calibrated_margin_floor_bits=_canonical_decimal(
                        screen.calibrated_margin_floor_bits
                    ),
                    effective_margin_bits=_canonical_decimal(screen.effective_margin_bits),
                    decision_threshold_bits=_canonical_decimal(screen.decision_threshold_bits),
                    reason=screen.reason,
                    metrics={
                        **metrics,
                        **_preflight_diagnostics(screen.diagnostics),
                    },
                )
            results.append(
                AttackExecution(
                    attack=attack,
                    outcome=outcome,
                    duration_ms=max(0, round((time.monotonic() - attack_started) * 1_000)),
                )
            )
            continue

        estimate = bkw_estimate(params)
        if math.isfinite(estimate.log2_cost):
            metrics = {
                **normalized_metrics,
                "preflight_rule_version": IntegerMetric(
                    kind="integer", value=str(BKW_PREFLIGHT_RULE_VERSION)
                ),
                **_preflight_diagnostics(estimate.diagnostics),
            }
            results.append(
                AttackExecution(
                    attack=attack,
                    outcome=ComputedOutcome(
                        kind="computed",
                        security_bits=_canonical_decimal(estimate.log2_cost),
                        metrics=metrics,
                    ),
                    duration_ms=max(0, round((time.monotonic() - attack_started) * 1_000)),
                )
            )
        else:
            results.append(
                AttackExecution(
                    attack=attack,
                    outcome=PreflightUnknownOutcome(
                        kind="preflight_unknown",
                        code="bounded_search_no_finite_candidate",
                        reason=(
                            f"{attack.value} bounded preflight search found no finite "
                            "candidate; exact estimation is required"
                        ),
                        raw_result=normalized_audit,
                    ),
                    duration_ms=max(0, round((time.monotonic() - attack_started) * 1_000)),
                )
            )
    return WorkerResponse(
        results=results,
        duration_ms=max(0, round((time.monotonic() - started) * 1_000)),
    )


@contextlib.contextmanager
def _arora_alarm(seconds: float):
    """Interrupt one Arora screen without aborting later attacks in the worker."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    def deadline_exceeded(_signum: int, _frame: object) -> None:
        """Turn SIGALRM into a scoped exception handled by the preflight path."""
        raise AroraScreenDeadline

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, deadline_exceeded)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _preflight_diagnostics(
    diagnostics: dict[str, int | float | str],
) -> dict[str, NormalizedMetric]:
    """Prefix and normalize algorithm diagnostics for the public metrics map."""
    metrics: dict[str, NormalizedMetric] = {}
    for name, value in diagnostics.items():
        key = f"preflight_{name}"
        if isinstance(value, int):
            metrics[key] = IntegerMetric(kind="integer", value=str(value))
        elif isinstance(value, float):
            metrics[key] = DecimalMetric(kind="decimal", value=_canonical_decimal(value))
        else:
            metrics[key] = TextMetric(kind="text", value=value)
    return metrics


def _run_estimator(request: EstimateRequest, attacks: list[Attack]) -> dict[str, Any]:
    """Call the installed ``lattice-estimator`` package for an exact request.

    This is the main upstream integration point.  Attack names, cost models,
    problem parameters, and deny lists are translated here before dispatching to
    ``LWE.estimate``, ``NTRU.estimate``, or ``SIS.estimate``.
    """
    # Import only inside the Sage worker: the HTTP process does not need Sage or
    # lattice-estimator loaded and remains independently killable/responsive.
    from estimator import LWE, NTRU, RC, SIS, Simulator  # type: ignore[import-not-found]

    cost_model = getattr(RC, request.cost_model.value)
    requested = {PUBLIC_TO_UPSTREAM[attack] for attack in attacks}

    if isinstance(request.problem, LweProblem):
        from sage.all import oo  # type: ignore[import-not-found]

        params = _lwe_parameters(request.problem)
        all_attacks = {PUBLIC_TO_UPSTREAM[item] for item in LWE_ATTACKS}
        return LWE.estimate(
            params,
            red_cost_model=cost_model,
            red_shape_model=Simulator.GSA,
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
            red_shape_model=Simulator.GSA,
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
            red_shape_model=Simulator.GSA,
            deny_list=(),
            jobs=1,
            catch_exceptions=True,
            quiet=True,
        )

    raise AssertionError("strict request model admitted an unknown problem")


def _lwe_parameters(problem: LweProblem) -> Any:
    """Translate the public LWE problem model to upstream ``LWE.Parameters``."""
    from estimator import LWE  # type: ignore[import-not-found]
    from sage.all import oo  # type: ignore[import-not-found]

    samples = oo if problem.samples.kind == "unlimited" else problem.samples.count
    return LWE.Parameters(
        n=problem.dimension,
        q=int(problem.modulus),
        Xs=_distribution(problem.secret, problem.dimension),
        Xe=_distribution(problem.error, None),
        m=samples,
    )


def _distribution(distribution: Any, logical_length: int | None) -> Any:
    """Translate a public distribution variant to an upstream ``estimator.ND`` object."""
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
    """Convert one upstream attack dictionary into a stable outcome variant."""
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
    """Return canonical ``log2(rop)`` for a finite positive upstream cost."""
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
    """Convert a supported scalar diagnostic to its tagged wire representation."""
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
    """Serialize a finite numeric value without exponent or insignificant zeros."""
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
    """Collect bounded non-empty upstream output for failure diagnostics."""
    result: dict[str, Any] = {}
    if stdout.getvalue():
        result["stdout"] = stdout.getvalue()[-16_384:]
    if stderr.getvalue():
        result["stderr"] = stderr.getvalue()[-16_384:]
    return result


def _safe_text(value: Any) -> str:
    """Render an upstream object while preventing broken ``str`` implementations."""
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - audit normalization must not mask the primary error
        return f"<{type(value).__name__}>"
