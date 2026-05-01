"""Event-driven backtester.

Why duplicate logic with the env? The training env makes choices to keep training
fast (vectorized obs, immediate reward). The backtester is the *honest* test: it
walks the data bar-by-bar, executes orders at next bar's open with explicit
slippage and spread, integrates with the live risk module, and returns a complete
trade ledger and equity curve.

Lookahead-leak is the most common silent bug in trading research; comparing
training-env metrics against backtester metrics is one of the best smoke tests
for it. They should agree to within a few percent. If they don't, somebody is
peeking at the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..risk.manager import RiskManager
from ..utils.logging_setup import get_logger
from ..utils.metrics import summary_stats
from .policies import Policy

logger = get_logger(__name__)


@dataclass
class Trade:
    open_time: pd.Timestamp
    close_time: Optional[pd.Timestamp]
    open_price: float
    close_price: Optional[float]
    direction: int                          # +1 long, -1 short
    units: float
    pnl: float = 0.0
    cost: float = 0.0


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: List[Trade] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def trade_pnls(self) -> np.ndarray:
        return np.array([t.pnl for t in self.trades if t.close_time is not None])

    def summary(self, periods_per_year: int = 252 * 96) -> dict:
        return summary_stats(self.equity, self.trade_pnls, periods_per_year)


@dataclass
class EventDrivenBacktester:
    """Walks bars sequentially and applies a Policy with realistic frictions."""

    initial_equity: float = 100_000.0
    contract_size: float = 100_000.0
    spread_pips: float = 0.8
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.3
    point_value: float = 1e-4
    latency_bars: int = 1               # signal at t executed at t+latency
    max_position: float = 1.0
    risk: Optional[RiskManager] = None  # optional risk overlay

    def run(
        self,
        prices: pd.DataFrame,
        observations: np.ndarray,
        policy: Policy,
        window_size: int = 64,
    ) -> BacktestResult:
        """Run a backtest. `observations[i]` is the obs constructed for bar
        `i + window_size` (i.e. it ends with bar i+window_size-1, so it's safe
        to act on bar i+window_size). The position taken applies for the bar
        from i+window_size+latency to i+window_size+latency+1.
        """
        if observations.shape[0] != len(prices) - window_size:
            raise ValueError(
                f"obs len {observations.shape[0]} ≠ prices len {len(prices)} - window {window_size}"
            )

        equity = self.initial_equity
        peak_equity = equity
        position = 0.0
        equity_curve: List[float] = []
        positions_log: List[float] = []
        timestamps: List[pd.Timestamp] = []
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None

        n = observations.shape[0]
        # Pending orders queue (bar_to_execute, target_position)
        pending: list[tuple[int, float]] = []

        for i in range(n - 1):
            ts = prices.index[i + window_size]
            price_now = float(prices.iloc[i + window_size]["close"])

            # ---- 1. Risk gate (kill-switch, daily loss, etc.) ----
            blocked = False
            if self.risk is not None:
                blocked = self.risk.should_halt(equity=equity, peak_equity=peak_equity, ts=ts)

            # ---- 2. Apply any pending orders due now ----
            target_pos = position
            for execute_at, tgt in list(pending):
                if execute_at == i:
                    target_pos = tgt
                    pending.remove((execute_at, tgt))

            # ---- 3. Execute change: charge costs at this bar's open ----
            if abs(target_pos - position) > 1e-9 and not blocked:
                exec_price = float(prices.iloc[i + window_size]["open"])
                # Slippage: positive when increasing long / decreasing short, negative the
                # other way; here we model it symmetrically as a worst-case cost.
                delta = target_pos - position
                slip = self.slippage_pips * self.point_value * (1 if delta > 0 else -1)
                fill_price = exec_price + slip
                spread_cost = abs(delta) * self.spread_pips * self.point_value * self.contract_size
                comm_cost = abs(delta) * self.commission_per_lot
                total_cost = spread_cost + comm_cost

                # Trade book-keeping
                if open_trade is None and target_pos != 0:
                    open_trade = Trade(
                        open_time=ts,
                        close_time=None,
                        open_price=fill_price,
                        close_price=None,
                        direction=int(np.sign(target_pos)),
                        units=abs(target_pos),
                        cost=total_cost,
                    )
                elif open_trade is not None and (
                    np.sign(target_pos) != open_trade.direction or target_pos == 0
                ):
                    # Close existing trade
                    open_trade.close_time = ts
                    open_trade.close_price = fill_price
                    open_trade.pnl = (
                        open_trade.direction
                        * (fill_price - open_trade.open_price)
                        * open_trade.units
                        * self.contract_size
                    ) - open_trade.cost
                    trades.append(open_trade)
                    open_trade = None
                    if target_pos != 0:
                        # Immediately open the opposite-side trade
                        open_trade = Trade(
                            open_time=ts,
                            close_time=None,
                            open_price=fill_price,
                            close_price=None,
                            direction=int(np.sign(target_pos)),
                            units=abs(target_pos),
                            cost=total_cost,
                        )

                equity -= total_cost
                position = target_pos

            # ---- 4. Mark to market on next close ----
            next_close = float(prices.iloc[i + window_size + 1]["close"])
            mtm_pnl = position * (next_close - price_now) * self.contract_size
            equity += mtm_pnl
            peak_equity = max(peak_equity, equity)

            equity_curve.append(equity)
            positions_log.append(position)
            timestamps.append(ts)

            # ---- 5. Decide for next bar (signal computed from current obs) ----
            if not blocked:
                target_frac = policy.act(observations[i])
                target_frac = float(np.clip(target_frac, -1.0, 1.0))
                tgt = target_frac * self.max_position
                # Risk overlay can clamp size
                if self.risk is not None:
                    tgt = self.risk.clamp_position(tgt, equity=equity)
                # Schedule for execution at i+latency
                if abs(tgt - position) > 1e-9:
                    pending.append((i + self.latency_bars, tgt))
            else:
                # If risk gate blocks us, flatten on next bar.
                if abs(position) > 1e-9:
                    pending.append((i + self.latency_bars, 0.0))

        # Close any dangling open trade at the final price
        if open_trade is not None:
            final_price = float(prices.iloc[-1]["close"])
            open_trade.close_time = prices.index[-1]
            open_trade.close_price = final_price
            open_trade.pnl = (
                open_trade.direction
                * (final_price - open_trade.open_price)
                * open_trade.units
                * self.contract_size
            ) - open_trade.cost
            trades.append(open_trade)

        eq_series = pd.Series(equity_curve, index=pd.DatetimeIndex(timestamps), name="equity")
        pos_series = pd.Series(positions_log, index=pd.DatetimeIndex(timestamps), name="position")
        return BacktestResult(
            equity=eq_series,
            positions=pos_series,
            trades=trades,
            metadata={
                "initial_equity": self.initial_equity,
                "n_bars": n,
                "n_trades": len(trades),
            },
        )
