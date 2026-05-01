"""Time-based splitting. Two flavors:

1. `TimeSplitter` — simple train/val/test cut by date. Used during early development.
2. `walk_forward_windows` — generator yielding rolling (train, val, test) windows.
   Used for honest out-of-sample evaluation. K-fold is *not* offered: it's wrong
   for time series, full stop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd


@dataclass
class SplitWindows:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"SplitWindows(train={len(self.train):,}, "
            f"val={len(self.val):,}, test={len(self.test):,})"
        )


@dataclass
class TimeSplitter:
    """Cut a single dataframe into train/val/test by timestamp."""
    train_end: str
    val_end: str

    def split(self, df: pd.DataFrame) -> SplitWindows:
        train_end = pd.Timestamp(self.train_end, tz="UTC")
        val_end = pd.Timestamp(self.val_end, tz="UTC")
        train = df.loc[df.index <= train_end]
        val = df.loc[(df.index > train_end) & (df.index <= val_end)]
        test = df.loc[df.index > val_end]
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(
                f"Empty split: train={len(train)}, val={len(val)}, test={len(test)}"
            )
        return SplitWindows(train=train, val=val, test=test)


def walk_forward_windows(
    df: pd.DataFrame,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    step_bars: Optional[int] = None,
) -> Iterator[SplitWindows]:
    """Yield rolling (train, val, test) windows. `step_bars` defaults to `test_bars`
    (non-overlapping test segments)."""
    n = len(df)
    step = step_bars or test_bars
    start = 0
    while start + train_bars + val_bars + test_bars <= n:
        train = df.iloc[start : start + train_bars]
        val = df.iloc[start + train_bars : start + train_bars + val_bars]
        test = df.iloc[
            start + train_bars + val_bars : start + train_bars + val_bars + test_bars
        ]
        yield SplitWindows(train=train, val=val, test=test)
        start += step
