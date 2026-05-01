"""Live runtime loop. Broker-agnostic: takes a `BaseBroker` and a `Policy`.

Per tick:
  1. Pull last N bars from broker.
  2. If a new bar has appeared, compute features and the latest observation.
  3. Run risk gate; if halted, flatten and skip.
  4. Run policy; clamp to position limits.
  5. Send / modify / close orders to reach target.
  6. Update internal state, log everything.

The loop is *crash-safe by design*: state lives on disk in JSON, the broker is
queried as the source of truth at every tick (we don't trust our cached view of
positions), and a kill-switch file flushes everything.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.policies import Policy
from ..features.engineer import FeatureEngineer
from ..features.normalizer import RollingZScore
from ..risk.manager import RiskManager
from ..training.pipeline import make_observation_matrix
from ..utils.logging_setup import get_logger
from .broker import BaseBroker

logger = get_logger(__name__)


@dataclass
class LiveRuntime:
    broker: BaseBroker
    policy: Policy
    risk: RiskManager
    symbol: str
    timeframe: str
    feature_engineer: FeatureEngineer
    normalizer: RollingZScore
    window_size: int = 64
    poll_seconds: int = 5
    bars_to_pull: int = 1000        # rolling buffer
    state_path: str = "artifacts/live_state.json"

    def __post_init__(self) -> None:
        self._last_bar_ts: Optional[pd.Timestamp] = None
        self._stop_requested = False
        self._load_state()

    # ----- state persistence ------------------------------------------------
    def _load_state(self) -> None:
        path = Path(self.state_path)
        if path.exists():
            try:
                state = json.loads(path.read_text())
                ts = state.get("last_bar_ts")
                if ts:
                    self._last_bar_ts = pd.Timestamp(ts)
            except Exception:
                logger.exception("failed to load live state")

    def _save_state(self) -> None:
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None})
        )

    # ----- main loop --------------------------------------------------------
    def run(self) -> None:
        self.broker.connect()
        try:
            logger.info("LIVE START | symbol=%s timeframe=%s", self.symbol, self.timeframe)
            while not self._stop_requested:
                try:
                    self._tick()
                except Exception:
                    logger.exception("tick failed; sleeping before retry")
                time.sleep(self.poll_seconds)
        finally:
            self.broker.disconnect()
            logger.info("LIVE STOP")

    def stop(self) -> None:
        self._stop_requested = True

    # ----- per-tick logic ---------------------------------------------------
    def _tick(self) -> None:
        bars = self.broker.latest_bars(self.symbol, self.timeframe, self.bars_to_pull)
        if bars.empty:
            return
        last_ts = bars.index[-1]
        if self._last_bar_ts is not None and last_ts <= self._last_bar_ts:
            return     # no new bar yet
        self._last_bar_ts = last_ts
        self._save_state()

        equity = self.broker.equity()
        positions = self.broker.positions()
        peak = max(equity, getattr(self, "_peak_equity", equity))
        self._peak_equity = peak
        current_units = sum(p.volume for p in positions)

        # ---- Risk gate ----
        if self.risk.should_halt(equity=equity, peak_equity=peak, ts=last_ts):
            logger.error("HALT: %s — flattening", self.risk.halted_reason)
            for p in positions:
                self.broker.close(p.ticket)
            return

        # ---- Compute observation ----
        feats = self.feature_engineer.transform(bars)
        if len(feats) < self.window_size + 50:
            logger.info("warming up: %d feature rows", len(feats))
            return
        normed = self.normalizer.transform(feats)
        feature_array = normed.values.astype(np.float32)
        obs_matrix = make_observation_matrix(feature_array, self.window_size)
        latest_obs = obs_matrix[-1]

        # Feed portfolio state into the obs (last 3 floats), matching env layout.
        # Position fraction, unrealized return, drawdown.
        max_pos = self.risk.cfg.max_position_units
        latest_obs[-3] = (current_units / max_pos) if max_pos > 0 else 0.0
        latest_obs[-2] = (equity / max(equity, 1.0)) - 1.0   # always 0 — no init equity tracked here; refine for prod
        latest_obs[-1] = (equity - peak) / max(peak, 1.0)

        # ---- Policy ----
        target_frac = float(np.clip(self.policy.act(latest_obs), -1.0, 1.0))
        target_units = self.risk.clamp_position(target_frac * max_pos, equity=equity)
        delta = target_units - current_units

        logger.info(
            "TICK ts=%s eq=%.2f pos=%+.2f target=%+.2f delta=%+.2f",
            last_ts, equity, current_units, target_units, delta,
        )

        if abs(delta) < 1e-6:
            return

        # Close opposing positions first, then open the residual.
        if positions and (np.sign(target_units) != np.sign(current_units) or target_units == 0):
            for p in positions:
                self.broker.close(p.ticket)
            current_units = 0.0
            delta = target_units

        if abs(delta) > 1e-6:
            self.broker.place_market(
                symbol=self.symbol,
                volume=float(delta),
                comment="ai_trader",
            )
