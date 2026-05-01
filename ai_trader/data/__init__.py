"""Data ingestion: synthetic, CSV, MT5 sources + time-based splitter."""
from .sources import DataSource, CSVSource, SyntheticSource, MT5Source
from .splitter import TimeSplitter, SplitWindows

__all__ = [
    "DataSource",
    "CSVSource",
    "SyntheticSource",
    "MT5Source",
    "TimeSplitter",
    "SplitWindows",
]
