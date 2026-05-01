"""Performance analytics. All functions accept either returns or an equity curve.

Conventions
-----------
* `returns` are bar-level simple returns (not log returns) unless stated otherwise.
* `periods_per_year` defaults to 252 * 96 for M15 forex (96 bars per 24h day, 252 sessions).
  Pass the right value for your timeframe — get it wrong and Sharpe is meaningless.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_array(x: Iterable[float]) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x, dtype=float)


def equity_to_returns(equity: Iterable[float]) -> np.ndarray:
    """Convert an equity curve to bar-level simple returns."""
    eq = _to_array(equity)
    if eq.size < 2:
        return np.zeros(0)
    return np.diff(eq) / eq[:-1]


# -----------------------------------------------------------------------------
# Risk-adjusted return
# -----------------------------------------------------------------------------
def sharpe_ratio(returns: Iterable[float], periods_per_year: int = 252 * 96) -> float:
    r = _to_array(returns)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / std)


def sortino_ratio(returns: Iterable[float], periods_per_year: int = 252 * 96) -> float:
    r = _to_array(returns)
    if r.size < 2:
        return 0.0
    downside = r[r < 0]
    if downside.size == 0:
        return float("inf") if r.mean() > 0 else 0.0
    dd_std = downside.std(ddof=1)
    if dd_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / dd_std)


def calmar_ratio(equity: Iterable[float], periods_per_year: int = 252 * 96) -> float:
    eq = _to_array(equity)
    if eq.size < 2:
        return 0.0
    total_return = eq[-1] / eq[0] - 1.0
    years = eq.size / periods_per_year
    cagr = (1 + total_return) ** (1 / max(years, 1e-9)) - 1.0 if total_return > -1 else -1.0
    mdd = max_drawdown(eq)
    if mdd == 0:
        return 0.0
    return float(cagr / abs(mdd))


# -----------------------------------------------------------------------------
# Drawdown
# -----------------------------------------------------------------------------
def max_drawdown(equity: Iterable[float]) -> float:
    """Return the most negative drawdown as a fraction (e.g. -0.23)."""
    eq = _to_array(equity)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def drawdown_series(equity: Iterable[float]) -> np.ndarray:
    eq = _to_array(equity)
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / peak


# -----------------------------------------------------------------------------
# Trade-level
# -----------------------------------------------------------------------------
def profit_factor(trade_pnls: Iterable[float]) -> float:
    p = _to_array(trade_pnls)
    wins = p[p > 0].sum()
    losses = -p[p < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(trade_pnls: Iterable[float]) -> float:
    p = _to_array(trade_pnls)
    if p.size == 0:
        return 0.0
    return float((p > 0).mean())


def expectancy(trade_pnls: Iterable[float]) -> float:
    """Average $ per trade. Simple but tells you whether the edge is real."""
    p = _to_array(trade_pnls)
    if p.size == 0:
        return 0.0
    return float(p.mean())


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def summary_stats(
    equity: Iterable[float],
    trade_pnls: Iterable[float] | None = None,
    periods_per_year: int = 252 * 96,
) -> dict:
    """Return a dict with everything we care about. Used by reports & dashboard."""
    eq = _to_array(equity)
    rets = equity_to_returns(eq)
    out = {
        "final_equity": float(eq[-1]) if eq.size else 0.0,
        "total_return_pct": float((eq[-1] / eq[0] - 1.0) * 100) if eq.size >= 2 else 0.0,
        "sharpe": sharpe_ratio(rets, periods_per_year),
        "sortino": sortino_ratio(rets, periods_per_year),
        "calmar": calmar_ratio(eq, periods_per_year),
        "max_drawdown_pct": max_drawdown(eq) * 100,
        "volatility_ann_pct": float(rets.std(ddof=1) * np.sqrt(periods_per_year) * 100)
        if rets.size > 1
        else 0.0,
        "n_bars": int(eq.size),
    }
    if trade_pnls is not None:
        tp = _to_array(trade_pnls)
        out.update(
            {
                "n_trades": int(tp.size),
                "win_rate_pct": win_rate(tp) * 100,
                "profit_factor": profit_factor(tp),
                "expectancy": expectancy(tp),
                "avg_win": float(tp[tp > 0].mean()) if (tp > 0).any() else 0.0,
                "avg_loss": float(tp[tp < 0].mean()) if (tp < 0).any() else 0.0,
            }
        )
    return out
