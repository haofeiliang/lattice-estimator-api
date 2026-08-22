from __future__ import annotations

import time

from src.slow_estimate import (
    ARORA_COARSE_MARGIN_FLOOR_BITS,
    ARORA_REFINED_MARGIN_FLOOR_BITS,
    _threshold_degree_limit,
    arora_gb_threshold_screen,
)


class BoundedDistribution:
    is_bounded = True
    is_Gaussian_like = False
    is_sparse = False

    def __init__(self, bounds: tuple[int, int]) -> None:
        self.bounds = bounds


class BoundedParameters:
    n = 64
    q = 256
    m = float("inf")
    Xs = BoundedDistribution((0, 1))
    Xe = BoundedDistribution((-1, 1))

    def normalize(self) -> BoundedParameters:
        return self


def test_target_degree_limit_grows_with_budget_but_stays_bounded() -> None:
    low = _threshold_degree_limit(512, 128, 2)
    high = _threshold_degree_limit(512, 256, 2)
    assert 0 <= low < high < 256


def test_arora_v6_uses_tier_floors_instead_of_output_dependent_margin() -> None:
    result = arora_gb_threshold_screen(BoundedParameters(), 128, 16, 16)
    assert result.precision_tier in {"coarse", "refined"}
    expected_floor = (
        ARORA_COARSE_MARGIN_FLOOR_BITS
        if result.precision_tier == "coarse"
        else ARORA_REFINED_MARGIN_FLOOR_BITS
    )
    assert result.calibrated_margin_floor_bits == expected_floor
    assert result.effective_margin_bits == max(16, expected_floor)
    assert result.decision_threshold_bits == 128 + result.effective_margin_bits
    assert result.diagnostics["max_degree_checked"] < 256


def test_expired_budget_never_returns_above_threshold() -> None:
    result = arora_gb_threshold_screen(
        BoundedParameters(),
        128,
        16,
        16,
        started=time.monotonic() - 5,
    )
    assert result.decision == "needs_exact"
    assert result.reason == "time_budget_exhausted"


def test_user_margins_are_independent_between_tiers() -> None:
    result = arora_gb_threshold_screen(BoundedParameters(), 128, 80, 24)
    expected_margin = 80 if result.precision_tier == "coarse" else 24
    assert result.effective_margin_bits == expected_margin
    assert result.decision_threshold_bits == 128 + expected_margin
