"""Regime detection via a Gaussian Hidden Markov Model.

We fit on the joint distribution of (1-bar return, realized vol over 30 bars) — the
typical 'risk premium / uncertainty' plane — and label each bar with the most likely
hidden state. The state IDs are then renamed by mean return so they are interpretable:
    * Lowest mean return  → 'bear' (state 0)
    * Middle              → 'chop' (state 1)
    * Highest mean return → 'bull' (state 2)
This makes regime IDs comparable across walk-forward windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class HMMRegimeDetector:
    """Two-feature Gaussian HMM. `n_states=3` is the usual sweet spot for FX."""

    n_states: int = 3
    covariance_type: str = "diag"
    n_iter: int = 100
    random_state: int = 42

    def __post_init__(self) -> None:
        self._model: Optional[object] = None
        self._state_order: Optional[np.ndarray] = None  # original_id → ranked_id

    def fit(self, returns: pd.Series, vol: pd.Series) -> "HMMRegimeDetector":
        from hmmlearn.hmm import GaussianHMM  # heavy import deferred

        x = np.column_stack([returns.values, vol.values]).astype(np.float64)
        x = x[~np.isnan(x).any(axis=1)]
        if x.shape[0] < self.n_states * 50:
            logger.warning("HMM fit on only %d points; results may be unstable", x.shape[0])

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(x)

        # Sort states by mean return so state 0 == bearish, state n-1 == bullish.
        means = model.means_[:, 0]
        order = np.argsort(means)              # original ids sorted by mean
        rename = np.empty_like(order)
        rename[order] = np.arange(self.n_states)  # rename[orig] = ranked_id

        self._model = model
        self._state_order = rename
        logger.info("HMM fitted, mean returns by state: %s", np.sort(means))
        return self

    def predict(self, returns: pd.Series, vol: pd.Series) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")
        x = np.column_stack([returns.values, vol.values]).astype(np.float64)
        mask = ~np.isnan(x).any(axis=1)
        labels = np.full(x.shape[0], -1, dtype=np.int64)
        if mask.any():
            preds = self._model.predict(x[mask])
            labels[mask] = self._state_order[preds]    # remap to ranked
        return pd.Series(labels, index=returns.index, name="regime")
