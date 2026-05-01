"""Feature engineering — explicitly avoids lagging indicators (RSI, MACD, etc.).

We use:
    * Multi-horizon log returns
    * Realized volatility at multiple windows (rolling std of returns)
    * Range / body / wick ratios — single-bar microstructure
    * Volume z-score (volume-of-trade proxy)
    * High-low range (intra-bar volatility)
    * Skew of recent returns (asymmetry of move)
    * Hour-of-day and day-of-week (cyclical encoding)

Reasoning: any deterministic transform of price already in the input is recoverable
by a sufficiently expressive model. Adding RSI just consumes capacity. Microstructure
features (range, volume) carry information *not* in close-to-close returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureEngineer:
    """Stateless feature builder. Apply with `.transform(df)`."""

    return_horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 15, 60])
    vol_windows: List[int] = field(default_factory=lambda: [10, 30, 90])
    include_time_features: bool = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        c = df["close"].astype(float)
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        v = df["volume"].astype(float)

        log_close = np.log(c)

        # 1. Multi-horizon log returns.  These are the primary inputs.
        for h_ in self.return_horizons:
            out[f"ret_{h_}"] = log_close.diff(h_)

        # 2. Realized volatility at several windows.
        ret_1 = out["ret_1"]
        for w in self.vol_windows:
            out[f"rvol_{w}"] = ret_1.rolling(w, min_periods=max(2, w // 2)).std()

        # 3. Microstructure ratios (intra-bar shape).
        rng = (h - l).replace(0, np.nan)
        body = (c - o)
        out["body_ratio"] = body / rng
        out["upper_wick_ratio"] = (h - np.maximum(c, o)) / rng
        out["lower_wick_ratio"] = (np.minimum(c, o) - l) / rng

        # 4. Bar range normalized by close (a clean proxy for instantaneous vol).
        out["range_pct"] = (h - l) / c

        # 5. Volume regime — z-score over the past 96 bars (~ 1 day on M15).
        v_log = np.log(v.replace(0, np.nan))
        out["vol_z"] = (v_log - v_log.rolling(96, min_periods=10).mean()) / (
            v_log.rolling(96, min_periods=10).std() + 1e-9
        )

        # 6. Skewness of last 30 returns — asymmetry of recent moves.
        out["ret_skew_30"] = ret_1.rolling(30, min_periods=10).skew()

        # 7. Trend strength: |return over 60| / sum of |returns over 60|.
        # This is a Sharpe-like signal-to-noise ratio of the move, NOT a lagging trend.
        ret_60_signed = log_close.diff(60)
        ret_1_abs = ret_1.abs()
        out["trend_strength"] = ret_60_signed / (
            ret_1_abs.rolling(60, min_periods=10).sum() + 1e-9
        )

        # 8. Spread proxy from high/low/close — useful when no L1 data.
        out["spread_proxy"] = (h - l) / (c + 1e-9)

        # 9. Cyclical time features. Markets have hour-of-day effects.
        if self.include_time_features:
            hour = df.index.hour + df.index.minute / 60.0
            out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            dow = df.index.dayofweek
            out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # Drop the warmup rows where rolling features are NaN.
        before = len(out)
        out = out.dropna()
        logger.info(
            "Feature engineering: %d → %d rows, %d features",
            before, len(out), out.shape[1],
        )
        return out

    def feature_names(self) -> List[str]:
        names = [f"ret_{h}" for h in self.return_horizons]
        names += [f"rvol_{w}" for w in self.vol_windows]
        names += [
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "range_pct", "vol_z", "ret_skew_30", "trend_strength", "spread_proxy",
        ]
        if self.include_time_features:
            names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        return names
