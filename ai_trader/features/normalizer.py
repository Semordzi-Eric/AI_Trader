"""Rolling z-score normalization. Critical: uses only past statistics so we never
leak future information at training time. Catching this bug is the entire reason
the project has a *separate* event-driven backtester downstream — they should agree."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RollingZScore:
    """z = (x - rolling_mean) / rolling_std, using a trailing window."""

    window: int = 500
    min_periods: int = 50
    eps: float = 1e-9

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df.rolling(self.window, min_periods=self.min_periods).mean()
        std = df.rolling(self.window, min_periods=self.min_periods).std()
        z = (df - mean) / (std + self.eps)
        # Clip extreme values (helps stability for nets); 6 sigma is generous.
        z = z.clip(lower=-6.0, upper=6.0)
        return z.dropna()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)
