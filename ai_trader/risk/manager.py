"""Risk overlay used by the backtester and the live runtime alike.

Halts and clamps are deliberately conservative — they shut things down on
ambiguous signals. A trading system that loses 10% in an afternoon is a system
that gets switched off; one that misses some trades because a circuit broke
is a system that lives to trade tomorrow. The trade is asymmetric.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class RiskConfig:
    max_position_units: float = 1.0
    max_open_positions: int = 1
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 15.0
    max_consecutive_losses: int = 5
    circuit_breaker_vol_mult: float = 5.0
    kill_switch_path: str = "artifacts/KILL"


class RiskManager:
    """Stateful risk overlay. Created with a config; ticked from the loop."""

    def __init__(self, config: RiskConfig) -> None:
        self.cfg = config
        self.consecutive_losses = 0
        self.day_start_equity: Optional[float] = None
        self.current_day: Optional[pd.Timestamp] = None
        self.baseline_vol: Optional[float] = None
        self.recent_returns: list[float] = []
        self._halted_reason: Optional[str] = None

    # --------------------------- decisions ---------------------------------
    def should_halt(self, equity: float, peak_equity: float, ts: pd.Timestamp) -> bool:
        """Return True if trading should be halted right now. Sets ._halted_reason."""

        # 1. Manual kill-switch — file presence on disk.
        if Path(self.cfg.kill_switch_path).exists():
            self._set_halt("manual_kill_switch")
            return True

        # 2. Max drawdown cutoff.
        dd_pct = 0.0 if peak_equity == 0 else (equity - peak_equity) / peak_equity * 100
        if dd_pct < -self.cfg.max_drawdown_pct:
            self._set_halt(f"max_drawdown_breach ({dd_pct:.2f}%)")
            return True

        # 3. Daily loss cutoff. Reset day_start_equity at first bar of each UTC day.
        day = ts.normalize() if hasattr(ts, "normalize") else pd.Timestamp(ts).normalize()
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = equity
            # Per-day reset of consecutive losses is intentional — a new session
            # gets a clean slate, but the kill-switch is sticky.
            self.consecutive_losses = 0

        if self.day_start_equity is not None:
            day_loss_pct = (equity - self.day_start_equity) / self.day_start_equity * 100
            if day_loss_pct < -self.cfg.max_daily_loss_pct:
                self._set_halt(f"daily_loss_breach ({day_loss_pct:.2f}%)")
                return True

        # 4. Consecutive losses.
        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self._set_halt(f"consecutive_losses ({self.consecutive_losses})")
            return True

        # 5. Volatility circuit breaker.
        if self.baseline_vol is not None and len(self.recent_returns) > 5:
            recent_std = float(np.std(self.recent_returns))
            if recent_std > self.cfg.circuit_breaker_vol_mult * self.baseline_vol:
                self._set_halt(
                    f"circuit_breaker_vol ({recent_std:.5f} vs baseline {self.baseline_vol:.5f})"
                )
                return True

        return False

    def clamp_position(self, target: float, equity: float) -> float:  # noqa: ARG002
        """Apply per-bar position cap. Equity-based scaling left as an extension hook."""
        cap = self.cfg.max_position_units
        return float(np.clip(target, -cap, cap))

    # --------------------------- updates -----------------------------------
    def record_trade_result(self, pnl: float) -> None:
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def update_vol_baseline(self, returns: np.ndarray | pd.Series) -> None:
        """Set the baseline volatility against which the circuit breaker compares."""
        r = np.asarray(returns)
        r = r[~np.isnan(r)]
        if r.size > 30:
            self.baseline_vol = float(np.std(r))

    def push_return(self, ret: float, max_buf: int = 30) -> None:
        self.recent_returns.append(ret)
        if len(self.recent_returns) > max_buf:
            self.recent_returns.pop(0)

    def reset_kill(self) -> None:
        path = Path(self.cfg.kill_switch_path)
        if path.exists():
            path.unlink()
            logger.warning("Kill-switch file removed: %s", path)
        self._halted_reason = None

    # --------------------------- internals ---------------------------------
    def _set_halt(self, reason: str) -> None:
        if self._halted_reason != reason:
            logger.error("RISK HALT: %s", reason)
        self._halted_reason = reason

    @property
    def halted_reason(self) -> Optional[str]:
        return self._halted_reason
