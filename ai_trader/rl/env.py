"""Custom Gymnasium trading environment.

Action space (continuous): a single scalar in [-1, 1] representing target position
as a fraction of `max_position`. Negative is short, zero is flat. The agent learns
position sizing implicitly — there is no separate 'size' action, which keeps the
problem tractable.

State / observation space: (window_size * n_features) flattened, plus a few portfolio
state variables (current position, unrealized PnL %, drawdown). Concatenating portfolio
state to features lets a vanilla MLP policy learn risk-aware behavior without
needing a recurrent net.

Reward (per bar):
    delta_equity / equity_prev               # PnL component
    - drawdown_penalty * max(0, dd - threshold)
    - vol_penalty * recent_vol
    - turnover_penalty * |Δposition|
This is shaped, not pure PnL — pure PnL rewards reckless leverage. The penalties
are configured in YAML and easy to tune.

Costs: spread (round-trip), commission (per side), slippage. Position changes
incur cost; holding does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class EnvConfig:
    """All knobs for the environment. Read from YAML in production."""
    window_size: int = 64
    initial_equity: float = 100_000.0
    contract_size: float = 100_000.0           # 1 standard FX lot
    max_position: float = 1.0                  # in lots
    spread_pips: float = 0.8
    commission_per_lot: float = 7.0            # round-turn USD
    slippage_pips: float = 0.3
    point_value: float = 1e-4                  # FX 4-digit
    # Reward shaping
    pnl_weight: float = 1.0
    drawdown_penalty: float = 0.5
    drawdown_threshold: float = 0.05           # start penalizing past 5% DD
    vol_penalty: float = 0.1
    turnover_penalty: float = 5e-4
    holding_bonus: float = 0.0
    # Risk
    max_drawdown_stop: float = 0.30            # episode terminates here
    # Rendering
    verbose: bool = False


class TradingEnv(gym.Env):
    """Single-asset, single-position trading environment."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: np.ndarray,
        config: EnvConfig,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        if features.shape[0] != len(prices):
            raise ValueError("features and prices must align in length")
        if features.shape[0] < config.window_size + 2:
            raise ValueError("not enough data for one full window")

        self.prices = prices.reset_index(drop=False)
        self.features = features.astype(np.float32)
        self.cfg = config
        self.feature_names = feature_names or [f"f{i}" for i in range(features.shape[1])]
        self.n_features = features.shape[1]

        # Observation: flat window + portfolio state (3 scalars)
        obs_dim = self.cfg.window_size * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: target position in [-1, 1] times max_position
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._step_idx: int = 0
        self._reset_state()

    # ----- gym API ----------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), self._info()

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # action: array of shape (1,)
        target_pos_frac = float(np.clip(action[0], -1.0, 1.0))
        target_pos = target_pos_frac * self.cfg.max_position

        prev_pos = self.position
        prev_equity = self.equity
        bar_idx = self._step_idx + self.cfg.window_size

        # Execute at next bar's open (avoid lookahead); penalize via slippage.
        # Here we use current bar's close as fill reference for simplicity, plus slippage.
        price = float(self.prices.loc[bar_idx, "close"])

        # ---- Apply cost on position change ----
        delta = target_pos - prev_pos
        cost = 0.0
        if abs(delta) > 1e-9:
            # Spread cost: paid on the change in absolute exposure (round-turn share).
            spread_cost = abs(delta) * self.cfg.spread_pips * self.cfg.point_value * self.cfg.contract_size
            slip_cost = abs(delta) * self.cfg.slippage_pips * self.cfg.point_value * self.cfg.contract_size
            comm_cost = abs(delta) * self.cfg.commission_per_lot
            cost = spread_cost + slip_cost + comm_cost

        self.position = target_pos

        # ---- Mark to next bar ----
        self._step_idx += 1
        next_bar_idx = self._step_idx + self.cfg.window_size
        terminated = False
        truncated = next_bar_idx >= len(self.prices) - 1

        if not truncated:
            next_price = float(self.prices.loc[next_bar_idx, "close"])
            pnl = self.position * (next_price - price) * self.cfg.contract_size
        else:
            pnl = 0.0

        self.equity = self.equity + pnl - cost
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = 0.0 if self.peak_equity == 0 else (self.equity - self.peak_equity) / self.peak_equity

        # ---- Reward shaping ----
        ret_step = (self.equity - prev_equity) / max(prev_equity, 1e-9)
        # Recent vol: rolling std of step returns over last 30 steps.
        self._return_buf.append(ret_step)
        if len(self._return_buf) > 30:
            self._return_buf.pop(0)
        recent_vol = float(np.std(self._return_buf)) if len(self._return_buf) > 1 else 0.0

        reward = self.cfg.pnl_weight * ret_step
        if drawdown < -self.cfg.drawdown_threshold:
            reward -= self.cfg.drawdown_penalty * (-drawdown - self.cfg.drawdown_threshold)
        reward -= self.cfg.vol_penalty * recent_vol
        reward -= self.cfg.turnover_penalty * abs(delta)
        if abs(self.position) > 1e-9:
            reward += self.cfg.holding_bonus

        # ---- Trade book-keeping ----
        # A 'trade' closes when position changes sign or returns to zero.
        if (prev_pos != 0) and (np.sign(self.position) != np.sign(prev_pos) or self.position == 0):
            trade_pnl = self.equity - self._trade_open_equity - cost
            self._trade_pnls.append(trade_pnl)
            if self.position != 0:
                self._trade_open_equity = self.equity
        elif prev_pos == 0 and self.position != 0:
            self._trade_open_equity = self.equity

        # ---- Termination on max DD ----
        if drawdown < -self.cfg.max_drawdown_stop:
            terminated = True
            if self.cfg.verbose:
                logger.info("env: terminated on max drawdown %.2f%%", drawdown * 100)

        info = self._info()
        info["bar_pnl"] = pnl - cost
        info["cost"] = cost
        info["drawdown"] = drawdown

        return self._obs(), float(reward), terminated, truncated, info

    # ----- internal --------------------------------------------------------
    def _reset_state(self) -> None:
        self._step_idx = 0
        self.position = 0.0
        self.equity = self.cfg.initial_equity
        self.peak_equity = self.cfg.initial_equity
        self._return_buf: list[float] = []
        self._trade_pnls: list[float] = []
        self._trade_open_equity = self.cfg.initial_equity

    def _obs(self) -> np.ndarray:
        i = self._step_idx
        window = self.features[i : i + self.cfg.window_size].ravel()
        port_state = np.array(
            [
                self.position / max(self.cfg.max_position, 1e-9),
                (self.equity - self.cfg.initial_equity) / self.cfg.initial_equity,
                (self.equity - self.peak_equity) / max(self.peak_equity, 1e-9),
            ],
            dtype=np.float32,
        )
        return np.concatenate([window, port_state]).astype(np.float32)

    def _info(self) -> dict:
        return {
            "equity": self.equity,
            "position": self.position,
            "peak_equity": self.peak_equity,
            "n_trades": len(self._trade_pnls),
        }

    @property
    def trade_pnls(self) -> list[float]:
        return list(self._trade_pnls)
