"""Cheap, attack-specific LWE cost estimates used before expensive Sage runs.

These estimates are deliberately optimistic screens.  They are not security
results: calibration against the corresponding lattice-estimator attacks is
required before their output is used with a stop margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

SLOW_ESTIMATE_RULE_VERSION = 4
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


@dataclass(frozen=True)
class _BkwEstimate:
    log2_cost: float
    log2_samples: float
    b: int
    t1: int
    t2: int
    ncod: int
    ntop: int
    ntest: int
    sample_additions: int = 0


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
    """Return the reviewed Arora-GB estimate and its selected algebraic parameters."""
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


def _log2_add(*values: float) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    if maximum == math.inf:
        return math.inf
    return maximum + math.log2(sum(2 ** (value - maximum) for value in finite))


def _log2_positive_difference(log2_left: float, right: int) -> float:
    if right <= 0:
        return log2_left
    log2_right = math.log2(right)
    if log2_left <= log2_right:
        return -math.inf
    delta = log2_right - log2_left
    if delta < -54:
        return log2_left
    return log2_left + math.log2(1 - 2**delta)


def _bkw_q_minus_one_log2(b: int, log2_q: float) -> float:
    exponent = b * log2_q
    if exponent > 54:
        return exponent
    return exponent + math.log2(1 - 2 ** (-exponent))


def _bkw_n(i: int, ell: int, ntest: int, b: int, log2_q: float) -> int:
    if ntest <= 0:
        return 0
    denominator = ell / ntest + i / (2 * log2_q)
    return math.floor(b / denominator) if denominator > 0 else 0


def _bkw_ntest(n: int, ell: int, t1: int, t2: int, b: int, log2_q: float) -> int:
    if t1 * b >= n:
        return 0
    upper = n - t1 * b

    def residual(ntest: int) -> int:
        ncod = sum(_bkw_n(i, ell, ntest, b, log2_q) for i in range(1, t2 + 1))
        return n - ncod - ntest - t1 * b

    low, high = 1, upper
    while low < high:
        middle = (low + high) // 2
        if residual(middle) > 0:
            low = middle + 1
        else:
            high = middle
    candidates = range(max(1, low - 4), min(upper, low + 4) + 1)
    return min(candidates, key=lambda value: (abs(residual(value)), value))


def _bkw_t1(n: int, ell: int, total_t2: int, b: int, log2_q: float) -> int:
    ntest = _bkw_ntest(n, ell, 0, total_t2, b, log2_q)
    result = sum(_bkw_n(i, ell, ntest, b, log2_q) <= b for i in range(1, total_t2 + 1))
    return min(result, n // b)


def _bkw_amplification_log2(log2_variance: float, log2_q: float) -> float:
    log2_sigma_over_q = 0.5 * log2_variance + 0.5 * math.log2(2 * math.pi) - log2_q
    if log2_sigma_over_q > 4:
        return math.inf
    log_advantage = -math.pi * 2 ** (2 * log2_sigma_over_q)
    if log_advantage >= math.log(0.99):
        return 0.0
    twice_log_advantage = 2 * log_advantage
    numerator = -2 * math.log(0.02)
    if twice_log_advantage < -36:
        return (math.log(numerator) - twice_log_advantage) / math.log(2)
    denominator = -math.log1p(-math.exp(twice_log_advantage))
    if denominator <= 0:
        return math.inf
    return math.log2(math.ceil(numerator / denominator))


def _bkw_local_minimum(
    start: int,
    stop: int,
    evaluate: Any,
    better: Any,
) -> _BkwEstimate | None:
    """Mirror estimator.util.local_minimum for integer parameters."""
    if stop <= start:
        return None
    low = start
    high = stop - 1
    initial_low, initial_high = low, high
    direction = -1
    next_value: int | None = high
    last_value: int | None = None
    seen: set[int] = set()
    best: _BkwEstimate | None = None
    while (
        next_value is not None
        and next_value not in seen
        and initial_low <= next_value <= initial_high
    ):
        last_value = next_value
        next_value = None
        current = evaluate(last_value)
        seen.add(last_value)
        if best is None and current is not None:
            best = current
        if current is not None and best is not None and better(current, best):
            best = current
            if abs(direction) != 1:
                direction = -1
                next_value = last_value - 1
            elif direction == -1:
                direction = -2
                high = last_value
                next_value = math.ceil((low + high) / 2)
            else:
                direction = 2
                low = last_value
                next_value = math.floor((low + high) / 2)
        elif direction == -1:
            direction = 1
            next_value = last_value + 2
        elif direction == 1:
            next_value = None
        elif direction == -2:
            low = last_value
            next_value = math.ceil((low + high) / 2)
        else:
            high = last_value
            next_value = math.floor((low + high) / 2)
        if next_value == last_value:
            next_value = None
    return best


def _bkw_cost(
    *,
    n: int,
    q: int,
    secret_stddev: float,
    error_stddev: float,
    secret_width: float,
    secret_larger_than_error: bool,
    total_t2: int,
    b: int,
) -> _BkwEstimate | None:
    log2_q = math.log2(q)
    ell = b - 1
    t1 = _bkw_t1(n, ell, total_t2, b, log2_q)
    t2 = total_t2 - t1
    ntest = _bkw_ntest(n, ell, t1, t2, b, log2_q)
    if ntest:
        log2_sigma_set = (1 - ell / ntest) * log2_q - 0.5 * math.log2(12)
        ni = [_bkw_n(i, ell, ntest, b, log2_q) for i in range(1, t2 + 1)]
    else:
        log2_sigma_set = -math.inf
        ni = [0] * t2
    ncod = sum(ni)
    ntot = ncod + ntest
    ntop = max(n - ncod - ntest - t1 * b, 0)
    invalid = _BkwEstimate(math.inf, math.inf, b, t1, t2, ncod, ntop, ntest)
    steps = t1 + t2
    error_variance_log2 = steps + 2 * math.log2(error_stddev)
    coding_variance_log2 = (
        2 * math.log2(secret_stddev) + 2 * log2_sigma_set + math.log2(ntot)
        if ntot > 0 and secret_stddev > 0
        else -math.inf
    )
    final_variance_log2 = _log2_add(error_variance_log2, coding_variance_log2)
    log2_m_amplification = _bkw_amplification_log2(final_variance_log2, log2_q)
    if not math.isfinite(log2_m_amplification):
        return invalid

    log2_qb_minus_one = _bkw_q_minus_one_log2(b, log2_q)

    def log2_m_plus_tables(table_count: int) -> float:
        table_term = (
            math.log2(table_count) + log2_qb_minus_one - 1 if table_count > 0 else -math.inf
        )
        return _log2_add(log2_m_amplification, table_term)

    log2_samples = log2_m_plus_tables(steps)
    costs: list[float] = []
    if secret_larger_than_error:
        remaining = n - t1 * b
        log2_m_minus_n = _log2_positive_difference(log2_samples, remaining)
        if log2_m_minus_n == -math.inf:
            return invalid
        costs.append(log2_m_minus_n + math.log2(n + 1) + math.log2(math.ceil(remaining / (b - 1))))

    c1_terms = []
    for i in range(1, t1 + 1):
        c1_terms.append(math.log2(n + 1 - i * b) + log2_m_plus_tables(steps - i))
    costs.append(_log2_add(*c1_terms))

    c2_terms: list[float] = []
    prefix_n = 0
    for i, n_i in enumerate(ni, start=1):
        if n_i > 0:
            c2_terms.append(2 + log2_m_plus_tables(i) + math.log2(n_i))
        prefix_n += n_i
        dimension_factor = ntop + ntest + prefix_n
        if dimension_factor > 0:
            c2_terms.append(math.log2(dimension_factor) + log2_m_plus_tables(i - 1))
    costs.append(_log2_add(*c2_terms))

    guessing_base = 2 * secret_width + 1
    if ntop > 0:
        costs.append(log2_m_amplification + math.log2(ntop) + ntop * math.log2(guessing_base))
    c4_test = 2 + log2_m_amplification + math.log2(ntest) if ntest > 0 else -math.inf
    c4_fft = ntop * math.log2(guessing_base) + b * log2_q + math.log2(b * log2_q + 1)
    costs.append(_log2_add(c4_test, c4_fft))
    log2_cost = _log2_add(*costs)
    success = math.erf(secret_width / math.sqrt(2 * error_stddev))
    if success <= 0:
        return invalid
    if ntop:
        log2_cost -= ntop * math.log2(success)
    return _BkwEstimate(log2_cost, log2_samples, b, t1, t2, ncod, ntop, ntest)


def bkw_estimate(params: Any) -> SlowEstimate:
    """Approximate pinned coded-BKW equations using ordinary Python and log space."""
    normalized = params.normalize()
    n = int(normalized.n)
    q = int(normalized.q)
    secret_stddev = _finite_float(getattr(normalized.Xs, "stddev", None))
    original_error_stddev = _finite_float(getattr(normalized.Xe, "stddev", None))
    bounds = getattr(normalized.Xs, "bounds", None)
    if (
        n <= 0
        or q <= 1
        or secret_stddev is None
        or secret_stddev <= 0
        or original_error_stddev is None
        or original_error_stddev <= 0
        or bounds is None
    ):
        return SlowEstimate(math.inf, {"model": "coded_bkw_structural"})
    try:
        lower, upper = (float(value) for value in bounds)
    except (TypeError, ValueError, OverflowError):
        return SlowEstimate(math.inf, {"model": "coded_bkw_structural"})
    if (
        bool(getattr(normalized.Xs, "is_Gaussian_like", False))
        and float(getattr(normalized.Xs, "mean", 0)) == 0
    ):
        lower = max(lower, -3 * secret_stddev)
        upper = min(upper, 3 * secret_stddev)
    secret_width = upper - lower + 1
    if not math.isfinite(secret_width) or secret_width <= 0:
        return SlowEstimate(math.inf, {"model": "coded_bkw_structural"})

    def search(error_stddev: float, available_log2_samples: float) -> _BkwEstimate | None:
        secret_larger = secret_stddev > error_stddev

        def better(current: _BkwEstimate, incumbent: _BkwEstimate) -> bool:
            sample_regression = (
                incumbent.log2_samples <= available_log2_samples < current.log2_samples
            )
            return current.log2_cost <= incumbent.log2_cost and not sample_regression

        def evaluate_b(b: int) -> _BkwEstimate | None:
            return _bkw_local_minimum(
                2,
                max(3, n // b),
                lambda total_t2: _bkw_cost(
                    n=n,
                    q=q,
                    secret_stddev=secret_stddev,
                    error_stddev=error_stddev,
                    secret_width=secret_width,
                    secret_larger_than_error=secret_larger,
                    total_t2=total_t2,
                    b=b,
                ),
                better,
            )

        return _bkw_local_minimum(
            2,
            3 * math.ceil(math.log2(q)),
            evaluate_b,
            better,
        )

    if _is_infinite(normalized.m):
        best = search(original_error_stddev, math.inf)
    else:
        sample_count = int(normalized.m)
        candidates: list[_BkwEstimate] = []
        original = search(original_error_stddev, math.log2(sample_count))
        if original is not None and original.log2_samples <= math.log2(sample_count):
            candidates.append(original)
        stale = 0
        for additions in range(1, min(sample_count, 64) + 1):
            capacity = _log2_binomial(sample_count, additions) + additions
            current = search(original_error_stddev * math.sqrt(additions), capacity)
            if current is not None and current.log2_samples <= capacity:
                current = replace(current, sample_additions=additions)
                if not candidates or current.log2_cost < min(
                    candidate.log2_cost for candidate in candidates
                ):
                    stale = 0
                else:
                    stale += 1
                candidates.append(current)
            elif candidates:
                stale += 1
            if stale >= 8:
                break
        best = min(candidates, key=lambda candidate: candidate.log2_cost) if candidates else None
    if best is None or not math.isfinite(best.log2_cost):
        return SlowEstimate(math.inf, {"model": "coded_bkw_structural"})

    error_stddev = original_error_stddev * math.sqrt(max(1, best.sample_additions))

    return SlowEstimate(
        best.log2_cost,
        {
            "model": "coded_bkw_structural",
            "b": best.b,
            "t1": best.t1,
            "t2": best.t2,
            "coded_coordinates": best.ncod,
            "guessed_coordinates": best.ntop,
            "tested_coordinates": best.ntest,
            "log2_samples": best.log2_samples,
            "effective_error_stddev": error_stddev,
            "sample_additions": best.sample_additions,
        },
    )
