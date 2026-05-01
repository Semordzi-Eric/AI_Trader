"""Meta-controller: chooses which agent to run based on current market features.

Setup:
    * Train a specialist agent per regime (or per walk-forward window).
    * For each historical bar in validation, evaluate every agent's PnL on the
      next H bars. Label that bar with `argmax_agent`.
    * Train a gradient-boosted classifier mapping (features → best_agent).
    * At inference, the classifier picks an agent for the next H bars.

This is a soft form of mixture-of-experts. It assumes the labels are reasonably
stable through time — checked empirically by holding out the last walk-forward
window for the meta-model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class MetaController:
    """LightGBM-based agent selector. Feature columns are passed at fit time."""

    feature_cols: List[str]
    n_agents: int

    def __post_init__(self) -> None:
        self.model: Optional[object] = None

    def fit(
        self,
        meta_features: pd.DataFrame,
        agent_pnls: np.ndarray,
        horizon: int = 100,
    ) -> "MetaController":
        """`agent_pnls` has shape (T, n_agents) — per-bar PnL of each agent if it
        had been the only one trading. We build the label as the argmax over a
        forward `horizon` window."""
        import lightgbm as lgb

        if agent_pnls.shape[1] != self.n_agents:
            raise ValueError(f"agent_pnls has {agent_pnls.shape[1]} cols, expected {self.n_agents}")
        if len(meta_features) != agent_pnls.shape[0]:
            raise ValueError("meta_features and agent_pnls must align in length")

        # Forward-rolling PnL for each agent — the label looks `horizon` bars ahead.
        cum = np.cumsum(agent_pnls, axis=0)
        # PnL[t : t+horizon] = cum[t+horizon] - cum[t]
        end = np.minimum(np.arange(len(cum)) + horizon, len(cum) - 1)
        forward_pnl = cum[end] - cum
        labels = forward_pnl.argmax(axis=1)

        valid = ~meta_features[self.feature_cols].isna().any(axis=1).values
        X = meta_features.loc[valid, self.feature_cols].values
        y = labels[valid]

        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=50,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)
        logger.info("Meta-controller trained on %d samples, %d agents", len(X), self.n_agents)
        return self

    def select(self, meta_features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MetaController not fitted")
        X = meta_features[self.feature_cols].values
        return self.model.predict(X)

    def select_proba(self, meta_features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("MetaController not fitted")
        X = meta_features[self.feature_cols].values
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_cols, "n_agents": self.n_agents}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MetaController":
        import joblib
        blob = joblib.load(path)
        mc = cls(feature_cols=blob["features"], n_agents=blob["n_agents"])
        mc.model = blob["model"]
        return mc
