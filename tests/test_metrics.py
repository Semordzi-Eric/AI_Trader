"""Tests for metrics. These verify the math against hand-calculable cases.

Bad metrics → bad decisions → real money lost. Worth being paranoid about.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ai_trader.utils.metrics import (
    calmar_ratio,
    drawdown_series,
    equity_to_returns,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


class TestEquityToReturns:
    def test_simple_returns_known_values(self):
        eq = np.array([100.0, 110.0, 99.0, 99.0])
        rets = equity_to_returns(eq)
        np.testing.assert_allclose(rets, [0.1, -0.1, 0.0], atol=1e-12)

    def test_empty_input_returns_empty(self):
        assert equity_to_returns([]).size == 0

    def test_single_value_returns_empty(self):
        assert equity_to_returns([100.0]).size == 0


class TestSharpe:
    def test_zero_returns_gives_zero(self):
        assert sharpe_ratio([0.0, 0.0, 0.0, 0.0]) == 0.0

    def test_constant_positive_returns_with_zero_std_gives_zero(self):
        # std == 0 means no risk, so undefined; we return 0 (don't propagate inf).
        assert sharpe_ratio([0.001, 0.001, 0.001, 0.001]) == 0.0

    def test_known_value_against_formula(self):
        rets = np.array([0.01, -0.005, 0.012, -0.003, 0.008])
        ppy = 252
        expected = math.sqrt(ppy) * rets.mean() / rets.std(ddof=1)
        assert sharpe_ratio(rets, periods_per_year=ppy) == pytest.approx(expected)


class TestSortino:
    def test_no_downside_with_positive_mean_is_inf(self):
        assert sortino_ratio([0.01, 0.02, 0.005]) == float("inf")

    def test_no_downside_with_zero_mean_is_zero(self):
        assert sortino_ratio([0.0, 0.0]) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown_when_monotonic_increasing(self):
        assert max_drawdown([100, 110, 120, 130]) == 0.0

    def test_known_drawdown(self):
        # Peak at 200, low at 100 → -50% DD.
        eq = np.array([100, 200, 100, 150, 90, 110])
        # Peak at 200, low at 90 → -55%.
        assert max_drawdown(eq) == pytest.approx(-0.55)

    def test_drawdown_series_starts_at_zero(self):
        dd = drawdown_series([100, 90])
        assert dd[0] == 0.0
        assert dd[1] == pytest.approx(-0.10)


class TestCalmar:
    def test_negative_total_return_negative_calmar(self):
        # Equity halves: total return = -0.5; max DD = -0.5 → calmar < 0.
        eq = np.array([100.0] * 10 + [50.0] * 10)
        c = calmar_ratio(eq, periods_per_year=20)
        assert c < 0


class TestTradeMetrics:
    def test_profit_factor_known(self):
        # wins = 30, losses = 10 → PF = 3
        assert profit_factor([10, -5, 20, -5]) == pytest.approx(3.0)

    def test_profit_factor_no_losses_gives_inf(self):
        assert profit_factor([10, 5, 1]) == float("inf")

    def test_profit_factor_no_trades_gives_zero(self):
        assert profit_factor([]) == 0.0

    def test_win_rate(self):
        assert win_rate([1, -1, 1, -1, 1]) == pytest.approx(3 / 5)

    def test_expectancy(self):
        # Mean of trade pnls.
        assert expectancy([10, -5, 20, -5]) == pytest.approx((10 - 5 + 20 - 5) / 4)
