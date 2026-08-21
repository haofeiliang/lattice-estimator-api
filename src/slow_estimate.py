"""Cheap, attack-specific LWE cost estimates used before expensive Sage runs.

These estimates are deliberately optimistic screens.  They are not security
results: calibration against the corresponding lattice-estimator attacks is
required before their output is used with a stop margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SLOW_ESTIMATE_RULE_VERSION = 2
ARORA_MAX_SOLVING_DEGREE = 256
ARORA_GAUSSIAN_MAX_TAIL_SEARCH = 64
ARORA_GAUSSIAN_REAL_PRECISION_BITS = 256


@dataclass(frozen=True)
class SlowEstimate:
    """A cheap log-cost estimate with calibration diagnostics."""

    log2_cost: float
    diagnostics: dict[str, int | float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class _AroraCoreEstimate:
    log2_cost: float
    degree: int
    tail: int
    samples: int


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _is_infinite(value: Any) -> bool:
    try:
        return math.isinf(float(value))
    except (TypeError, ValueError, OverflowError):
        return str(value) in {"+Infinity", "Infinity"}


def _log2_int_like(value: Any) -> float:
    if _is_infinite(value):
        return math.inf
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if integer <= 0:
        return math.inf
    return math.log2(integer)


def _bounds_width(distribution: Any) -> int | None:
    if not bool(getattr(distribution, "is_bounded", False)):
        return None
    bounds = getattr(distribution, "bounds", None)
    if bounds is None:
        return None
    try:
        low, high = (int(item) for item in bounds)
    except (TypeError, ValueError, OverflowError):
        return None
    width = high - low + 1
    return width if width > 0 else None


def _log2_binomial(n: int, k: int) -> float:
    if k < 0 or k > n:
        return math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def _log2_monomials(num_vars: int, degree: int) -> float:
    return _log2_binomial(num_vars + degree, degree)


def _semi_regular_dreg(
    num_vars: int,
    equations: tuple[tuple[int, int], ...],
    *,
    max_degree: int = ARORA_MAX_SOLVING_DEGREE,
) -> int | None:
    """Estimate solving degree from the first negative Hilbert coefficient."""
    numerator = {0: 1}
    for equation_degree, count in equations:
        if equation_degree <= 0 or count <= 0:
            continue
        following: dict[int, int] = {}
        for shift, base in numerator.items():
            for index in range(min(count, (max_degree - shift) // equation_degree) + 1):
                coefficient = math.comb(count, index)
                if index % 2:
                    coefficient = -coefficient
                target = shift + index * equation_degree
                following[target] = following.get(target, 0) + base * coefficient
        numerator = {
            degree: coefficient for degree, coefficient in following.items() if coefficient
        }

    for degree in range(max_degree + 1):
        coefficient = sum(
            numerator_coefficient * math.comb(num_vars + degree - shift - 1, degree - shift)
            for shift, numerator_coefficient in numerator.items()
            if degree >= shift
        )
        if coefficient < 0:
            return degree
    return None


def _secret_equations(params: Any, dimension: int | None = None) -> tuple[tuple[int, int], ...]:
    dimension = int(params.n) if dimension is None else dimension
    try:
        if params.Xs > params.Xe:
            return ()
    except TypeError:
        return ()
    width = _bounds_width(params.Xs)
    if width is not None:
        return ((width, dimension),)
    if bool(getattr(params.Xs, "is_Gaussian_like", False)):
        sigma = _finite_float(getattr(params.Xs, "stddev", None))
        if sigma is not None and sigma > 0:
            return ((2 * math.ceil(3 * sigma) + 1, dimension),)
    return ()


def _sample_aware_degree(n: int, error_degree: int, samples: Any) -> int:
    if _is_infinite(samples):
        return error_degree
    try:
        if int(samples) <= n * n:
            return max(error_degree + 2, 2 * error_degree - 2)
    except (TypeError, ValueError, OverflowError):
        return max(error_degree + 2, 2 * error_degree - 2)
    exponent = _log2_int_like(samples) / math.log2(n)
    return error_degree + max(0, math.ceil(error_degree - exponent))


def _arora_bounded(params: Any, omega: float) -> float:
    n = int(params.n)
    width = _bounds_width(params.Xe)
    if width is None or width > 128:
        return math.inf
    maximum_samples = n**width
    samples = maximum_samples if _is_infinite(params.m) else min(int(params.m), maximum_samples)
    equations = ((width, samples), *_secret_equations(params))
    degree = _semi_regular_dreg(n, equations)
    if degree is None:
        degree = _sample_aware_degree(n, width, samples)
    return omega * _log2_monomials(n, degree)


def _gaussian_tail_sample_count(sigma_value: Any, tail: int) -> int:
    """Match estimator's 256-bit ``floor(log(0.99, p_single))`` cheaply."""
    try:
        from sage.all import RealField  # type: ignore[import-not-found]
    except ImportError:
        RealField = None
    if RealField is not None:
        real = RealField(ARORA_GAUSSIAN_REAL_PRECISION_BITS)
        ratio = real(tail) / real(str(sigma_value))
        epsilon = real(2) / (ratio * (real(2) * real.pi()).sqrt())
        epsilon *= (-(ratio**2) / 2).exp()
        single_probability = real(1) - epsilon
        if single_probability == 1:
            return 2**31
        return max(
            0,
            int((real("0.99").log() / single_probability.log()).floor()),
        )

    sigma = _finite_float(sigma_value)
    if sigma is None or sigma <= 0:
        return 0
    ratio = tail / sigma
    log_epsilon = math.log(2) - math.log(ratio * math.sqrt(2 * math.pi)) - ratio * ratio / 2
    # RealField(256) rounds 1-epsilon to one at roughly this boundary.  The
    # upstream estimator then uses the same arbitrary 2^31 cap.
    if log_epsilon <= -ARORA_GAUSSIAN_REAL_PRECISION_BITS * math.log(2):
        return 2**31
    epsilon = math.exp(log_epsilon)
    if not 0 < epsilon < 1:
        return 0
    log_single_probability = math.log1p(-epsilon)
    if not math.isfinite(log_single_probability) or log_single_probability >= 0:
        return 0
    return max(0, math.floor(math.log(0.99) / log_single_probability))


def _arora_gaussian_core(params: Any, omega: float, n: int) -> _AroraCoreEstimate | None:
    sigma_value = getattr(params.Xe, "stddev", None)
    sigma = _finite_float(sigma_value)
    if sigma is None or sigma <= 0:
        return None
    max_tail = min(n - 1, ARORA_GAUSSIAN_MAX_TAIL_SEARCH)
    if max_tail < math.ceil(sigma):
        return None
    finite_sample_log2 = _log2_int_like(params.m)
    secret_equations = _secret_equations(params, n)
    best: _AroraCoreEstimate | None = None
    stuck = 0
    for tail in range(max(1, math.ceil(sigma)), max_tail + 1):
        samples = _gaussian_tail_sample_count(sigma_value, tail)
        if samples <= 0:
            continue
        if not _is_infinite(params.m) and math.log2(samples) > finite_sample_log2:
            break
        degree = _semi_regular_dreg(n, ((2 * tail + 1, samples), *secret_equations))
        if degree is None:
            continue
        current = _AroraCoreEstimate(
            log2_cost=omega * _log2_monomials(n, degree),
            degree=degree,
            tail=tail,
            samples=samples,
        )
        if best is None or current.log2_cost < best.log2_cost:
            best = current
            stuck = 0
        else:
            stuck += 1
            if stuck >= 5:
                break
    return best


def _dense_guess_base(params: Any) -> int | None:
    if bool(getattr(params.Xs, "is_sparse", False)):
        return None
    return _bounds_width(params.Xs)


def _log_combination(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_add(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _sparse_repetition_log2(n: int, h: int, zeta: int, base: int) -> float:
    """Log2 of upstream's sparse guessing repetition/search-space factor."""
    if h == 0 or zeta == 0:
        return 0.0
    log_probability = -math.inf
    search_space = 0
    best = math.inf
    for gamma in range(min(h, zeta)):
        log_term = (
            _log_combination(n - h, zeta - gamma)
            + _log_combination(h, gamma)
            - _log_combination(n, zeta)
        )
        log_probability = _log_add(log_probability, log_term)
        search_space += math.comb(zeta, gamma) * base**gamma
        if log_probability < -36:
            log_trials = math.log(-math.log(0.01)) - log_probability
            current = log_trials / math.log(2) + math.log2(search_space)
        else:
            probability = min(1.0, math.exp(log_probability))
            trials = (
                1 if probability >= 0.99 else math.ceil(math.log(0.01) / math.log1p(-probability))
            )
            current = math.log2(trials) + math.log2(search_space)
        if current >= best:
            break
        best = current
    return best


def _composition_cost(
    params: Any,
    core: _AroraCoreEstimate,
    omega: float,
    zeta: int,
    base: int,
) -> _AroraCoreEstimate | None:
    n = int(params.n) - zeta
    if n <= 0:
        return None
    equations = (
        (2 * core.tail + 1, core.samples),
        *_secret_equations(params, n),
    )
    degree = _semi_regular_dreg(n, equations)
    if degree is None:
        return None
    return _AroraCoreEstimate(
        log2_cost=omega * _log2_monomials(n, degree) + zeta * math.log2(base),
        degree=degree,
        tail=core.tail,
        samples=core.samples,
    )


def _arora_dense_guessing(
    params: Any, core: _AroraCoreEstimate, omega: float
) -> tuple[_AroraCoreEstimate, int]:
    """Approximate the upstream dense-secret local minimum over guessed coordinates."""
    base = _dense_guess_base(params)
    if base is None or base <= 1:
        return core, 0
    max_zeta = min(int(params.n) - 1, math.floor(core.log2_cost / math.log2(base)))
    if max_zeta <= 0:
        return core, 0

    cache: dict[int, _AroraCoreEstimate | None] = {0: core}

    def evaluate(zeta: int) -> _AroraCoreEstimate | None:
        zeta = max(0, min(max_zeta, zeta))
        if zeta not in cache:
            cache[zeta] = _composition_cost(params, core, omega, zeta, base)
        return cache[zeta]

    # The upstream cost curve is searched as a local minimum.  Integer ternary
    # search keeps this preflight logarithmic in n while retaining the same
    # composition term base^zeta.
    low, high = 0, max_zeta
    while high - low > 8:
        left = low + (high - low) // 3
        right = high - (high - low) // 3
        left_cost = evaluate(left)
        right_cost = evaluate(right)
        left_value = left_cost.log2_cost if left_cost is not None else math.inf
        right_value = right_cost.log2_cost if right_cost is not None else math.inf
        if left_value <= right_value:
            high = right - 1
        else:
            low = left + 1
    for zeta in range(low, high + 1):
        evaluate(zeta)
    finite = [(estimate.log2_cost, zeta, estimate) for zeta, estimate in cache.items() if estimate]
    if not finite:
        return core, 0
    _, zeta, _ = min(finite)
    # Solving-degree jumps make the integer curve only approximately unimodal.
    # Polish the best coarse point so adjacent zeta values cannot be skipped.
    for neighbor in range(max(0, zeta - 4), min(max_zeta, zeta + 4) + 1):
        evaluate(neighbor)
    finite = [(estimate.log2_cost, zeta, estimate) for zeta, estimate in cache.items() if estimate]
    _, zeta, estimate = min(finite)
    return estimate, zeta


def _arora_sparse_guessing(params: Any, omega: float) -> tuple[_AroraCoreEstimate, int] | None:
    """Approximate upstream sparse-secret guessing in the log domain."""
    n = int(params.n)
    h = int(getattr(params.Xs, "hamming_weight", 0))
    width = _bounds_width(params.Xs)
    if h > 0 and (width is None or width <= 1):
        return None
    base = 1 if h == 0 else width - 1
    max_zeta = max(0, n - 40)
    # Guessing is optional, so zeta=0 is a mandatory baseline.  Sparse cost
    # curves are not globally unimodal; seeding this endpoint prevents the
    # logarithmic search from returning a result worse than not guessing.
    baseline = _arora_gaussian_core(params, omega, n)
    cache: dict[int, _AroraCoreEstimate | None] = {0: baseline}

    def evaluate(zeta: int) -> _AroraCoreEstimate | None:
        zeta = max(0, min(max_zeta, zeta))
        if zeta not in cache:
            core = _arora_gaussian_core(params, omega, n - zeta)
            if core is None:
                cache[zeta] = None
            else:
                repeat = _sparse_repetition_log2(n, h, zeta, base)
                cache[zeta] = _AroraCoreEstimate(
                    log2_cost=core.log2_cost + repeat,
                    degree=core.degree,
                    tail=core.tail,
                    samples=core.samples,
                )
        return cache[zeta]

    low, high = 0, max_zeta
    while high - low > 8:
        left = low + (high - low) // 3
        right = high - (high - low) // 3
        left_cost = evaluate(left)
        right_cost = evaluate(right)
        left_value = left_cost.log2_cost if left_cost is not None else math.inf
        right_value = right_cost.log2_cost if right_cost is not None else math.inf
        if left_value <= right_value:
            high = right - 1
        else:
            low = left + 1
    for zeta in range(low, high + 1):
        evaluate(zeta)
    finite = [(estimate.log2_cost, zeta, estimate) for zeta, estimate in cache.items() if estimate]
    if not finite:
        return None
    _, zeta, _ = min(finite)
    for neighbor in range(max(0, zeta - 4), min(max_zeta, zeta + 4) + 1):
        evaluate(neighbor)
    finite = [(estimate.log2_cost, zeta, estimate) for zeta, estimate in cache.items() if estimate]
    _, zeta, estimate = min(finite)
    return estimate, zeta


def arora_gb_estimate(params: Any, omega: float = 2.0) -> SlowEstimate:
    """Return the v2 Arora-GB estimate and its selected algebraic parameters."""
    normalized = params.normalize()
    if bool(getattr(normalized.Xe, "is_bounded", False)):
        cost = _arora_bounded(normalized, omega)
        return SlowEstimate(cost, {"model": "bounded"})
    if not bool(getattr(normalized.Xe, "is_Gaussian_like", False)):
        return SlowEstimate(math.inf)
    if bool(getattr(normalized.Xs, "is_sparse", False)):
        sparse = _arora_sparse_guessing(normalized, omega)
        if sparse is None:
            return SlowEstimate(math.inf)
        selected, zeta = sparse
        composition = "sparse_guessing"
    else:
        core = _arora_gaussian_core(normalized, omega, int(normalized.n))
        if core is None:
            return SlowEstimate(math.inf)
        selected, zeta = _arora_dense_guessing(normalized, core, omega)
        composition = "dense_guessing"
    return SlowEstimate(
        selected.log2_cost,
        {
            "model": "gaussian",
            "composition": composition,
            "tail": selected.tail,
            "solving_degree": selected.degree,
            "guessed_coordinates": zeta,
            "reduced_dimension": int(normalized.n) - zeta,
            "log2_samples": math.log2(selected.samples),
        },
    )


def arora_gb_log2_cost(params: Any, omega: float = 2.0) -> float:
    """Return only the v2 Arora-GB log-cost for compatibility."""
    return arora_gb_estimate(params, omega).log2_cost


def bkw_log2_cost(params: Any) -> float:
    """Return the restored optimistic coded-BKW table-cost estimate."""
    normalized = params.normalize()
    sigma = _finite_float(getattr(normalized.Xe, "stddev", None))
    if sigma is None or sigma <= 0:
        return math.inf
    logq = math.log2(int(normalized.q))
    available_log2_samples = _log2_int_like(normalized.m)
    best = math.inf
    for block_size in range(1, min(int(normalized.n), 64) + 1):
        blocks = math.ceil(int(normalized.n) / block_size)
        table_log2 = block_size * logq
        required_log2_samples = math.log2(max(1, blocks)) + table_log2 + 8
        if not _is_infinite(normalized.m) and available_log2_samples < required_log2_samples:
            continue
        effective_sigma_log2 = math.log2(sigma) + 0.5 * blocks
        if effective_sigma_log2 - logq > math.log2(0.25):
            continue
        best = min(best, math.log2(max(1, blocks)) + table_log2)
    return best
