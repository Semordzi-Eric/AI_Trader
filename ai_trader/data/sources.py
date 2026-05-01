"""Pluggable data sources.

All sources return a `pd.DataFrame` indexed by UTC timestamp with columns:
    open, high, low, close, volume

The synthetic source is deliberately rich — regime switches, fat-tailed innovations,
intraday seasonality — so the rest of the pipeline can be exercised without paid data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------
class DataSource(ABC):
    """Abstract data source. `load` is the only required method."""

    OHLCV = ["open", "high", "low", "close", "volume"]

    @abstractmethod
    def load(self) -> pd.DataFrame: ...

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in DataSource.OHLCV if c not in df.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex")
        # Drop any rows where high < low (data corruption) or with NaNs.
        bad = (df["high"] < df["low"]) | df[DataSource.OHLCV].isna().any(axis=1)
        if bad.any():
            logger.warning("Dropping %d corrupt OHLCV rows", int(bad.sum()))
            df = df.loc[~bad]
        return df.sort_index()


# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------
@dataclass
class CSVSource(DataSource):
    """Read OHLCV from a CSV file. Expected columns (case-insensitive):
    timestamp, open, high, low, close, volume.
    """
    path: str | Path
    timestamp_col: str = "timestamp"
    start: Optional[str] = None
    end: Optional[str] = None

    def load(self) -> pd.DataFrame:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        # Be permissive: if the first column is unnamed (pandas-default index
        # column from to_csv), promote it to the configured timestamp column.
        if self.timestamp_col not in df.columns:
            unnamed = [c for c in df.columns if c.startswith("unnamed")]
            if unnamed:
                df = df.rename(columns={unnamed[0]: self.timestamp_col})
            else:
                raise ValueError(f"Missing timestamp column '{self.timestamp_col}'")
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)
        df = df.set_index(self.timestamp_col)
        df = self._validate(df)
        if self.start:
            df = df.loc[df.index >= pd.Timestamp(self.start, tz="UTC")]
        if self.end:
            df = df.loc[df.index <= pd.Timestamp(self.end, tz="UTC")]
        logger.info("CSV loaded: %s rows from %s", f"{len(df):,}", path.name)
        return df


# -----------------------------------------------------------------------------
# Synthetic
# -----------------------------------------------------------------------------
@dataclass
class SyntheticSource(DataSource):
    """Generate synthetic OHLCV with regime switching for offline testing.

    Model: piecewise GBM with three regimes (trend up / mean-reverting / volatile),
    fat-tailed innovations (Student-t, df=5), and intraday seasonality on volume.
    """
    n_bars: int = 50_000
    timeframe_minutes: int = 15
    start: str = "2018-01-01"
    seed: int = 42
    initial_price: float = 1.10
    regime_persistence: float = 0.998      # P(stay in regime) per bar

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n_bars
        # Regime parameters: (drift_per_bar, vol_per_bar, mean_reversion_strength)
        regimes = np.array([
            [+1.5e-5, 4.0e-4, 0.00],   # bull
            [-0.5e-5, 3.5e-4, 0.10],   # mean-reverting
            [+0.0e-5, 9.0e-4, 0.00],   # volatile chop
        ])

        # Sample regime sequence (Markov)
        state = 0
        states = np.empty(n, dtype=np.int8)
        switch_p = 1.0 - self.regime_persistence
        for t in range(n):
            if rng.random() < switch_p:
                state = (state + rng.integers(1, 3)) % 3
            states[t] = state

        # Innovations: Student-t for fat tails
        innov = rng.standard_t(df=5.0, size=n) / np.sqrt(5.0 / 3.0)  # rescale to ~unit var
        log_close = np.empty(n)
        log_close[0] = np.log(self.initial_price)
        prev_log = log_close[0]
        for t in range(1, n):
            mu, sigma, mr = regimes[states[t]]
            # Mean reversion pulls log price toward initial_price
            mr_term = -mr * (prev_log - np.log(self.initial_price))
            log_close[t] = prev_log + mu + mr_term + sigma * innov[t]
            prev_log = log_close[t]
        close = np.exp(log_close)

        # Build OHLC from intra-bar walk
        open_ = np.empty(n)
        high = np.empty(n)
        low = np.empty(n)
        open_[0] = close[0]
        for t in range(n):
            o = close[t - 1] if t > 0 else close[0]
            c = close[t]
            sigma = regimes[states[t], 1] * o
            wick_up = abs(rng.normal(0, sigma))
            wick_dn = abs(rng.normal(0, sigma))
            high[t] = max(o, c) + wick_up
            low[t] = min(o, c) - wick_dn
            open_[t] = o

        # Volume with intraday seasonality (U-shape)
        ts_index = pd.date_range(
            start=self.start, periods=n, freq=f"{self.timeframe_minutes}min", tz="UTC"
        )
        hour = ts_index.hour + ts_index.minute / 60.0
        seasonality = 1.0 + 0.6 * np.cos((hour - 12) / 12.0 * np.pi)
        base_vol = rng.lognormal(mean=8.0, sigma=0.4, size=n)
        # Volume spikes when |return| is large
        ret_abs = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))
        vol = base_vol * seasonality * (1.0 + 5.0 * ret_abs / (ret_abs.std() + 1e-12))

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=ts_index,
        )
        df = self._validate(df)
        logger.info(
            "Synthetic data generated: %s bars, %s timeframe",
            f"{len(df):,}",
            f"{self.timeframe_minutes}min",
        )
        return df


# -----------------------------------------------------------------------------
# MetaTrader 5
# -----------------------------------------------------------------------------
@dataclass
class MT5Source(DataSource):
    """Pull bars from MetaTrader 5. Import-guarded so non-Windows hosts can still
    load this module (for tests, dashboards, etc.). Real fetch happens at .load()."""
    symbol: str
    timeframe: str = "M15"          # M1, M5, M15, H1, H4, D1
    n_bars: int = 50_000
    end: Optional[str] = None

    _TF_MAP = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440,
    }

    def load(self) -> pd.DataFrame:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MetaTrader5 package not available. Install on a Windows host with "
                "the MT5 terminal installed."
            ) from exc

        if not mt5.initialize():
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

        try:
            tf_const = getattr(mt5, f"TIMEFRAME_{self.timeframe}")
            end_dt = pd.Timestamp(self.end, tz="UTC").to_pydatetime() if self.end else None
            if end_dt:
                rates = mt5.copy_rates_from(self.symbol, tf_const, end_dt, self.n_bars)
            else:
                rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, self.n_bars)
            if rates is None or len(rates) == 0:
                raise RuntimeError(f"No bars returned for {self.symbol}")
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"tick_volume": "volume"})
            df = df.set_index("time")[self.OHLCV]
            return self._validate(df)
        finally:
            mt5.shutdown()
