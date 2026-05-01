"""Reusable feature → observations pipeline.

The contract here matters: the backtester and the RL env must agree on what each
observation represents. We define `obs[i]` as: 'the flattened window ending at bar
`i + window_size - 1`, plus three portfolio-state placeholders'. Building the
observation matrix once and feeding it to both keeps them in lock-step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.engineer import FeatureEngineer
from ..features.normalizer import RollingZScore
from ..features.regime import HMMRegimeDetector
from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class FeaturePipelineOutput:
    prices: pd.DataFrame                    # aligned to features (warmup dropped)
    features: pd.DataFrame                  # normalized features
    feature_array: np.ndarray               # float32 (T, F)
    regimes: pd.Series | None               # length T, integer state ids
    feature_names: list[str]
    window_size: int


def build_features_pipeline(
    df: pd.DataFrame,
    return_horizons: list[int],
    vol_windows: list[int],
    window_size: int,
    normalize_window: int = 500,
    use_regime: bool = True,
    n_regimes: int = 3,
    fit_regime_on: pd.DataFrame | None = None,
) -> FeaturePipelineOutput:
    """Run the full feature → normalize → align pipeline.

    `fit_regime_on`, if given, is the dataframe used to fit the HMM (typically the
    training split). Avoids leaking validation data into the regime model.
    """
    fe = FeatureEngineer(return_horizons=return_horizons, vol_windows=vol_windows)
    feats = fe.transform(df)

    # Align prices to features after warmup drop.
    prices_aligned = df.loc[feats.index]

    norm = RollingZScore(window=normalize_window, min_periods=max(50, normalize_window // 10))
    feats_norm = norm.fit_transform(feats)
    prices_aligned = prices_aligned.loc[feats_norm.index]

    regimes: pd.Series | None = None
    if use_regime:
        # If a fit set is supplied, fit on it; else fit on the first 60% of the data.
        ret_full = feats_norm["ret_1"]
        vol_full = feats_norm.get(f"rvol_{vol_windows[1]}", feats_norm[f"rvol_{vol_windows[0]}"])
        hmm = HMMRegimeDetector(n_states=n_regimes)
        if fit_regime_on is not None:
            fit_feats = fe.transform(fit_regime_on)
            fit_feats_norm = norm.fit_transform(fit_feats)
            hmm.fit(fit_feats_norm["ret_1"], fit_feats_norm[f"rvol_{vol_windows[1]}"])
        else:
            cutoff = int(len(feats_norm) * 0.6)
            hmm.fit(ret_full.iloc[:cutoff], vol_full.iloc[:cutoff])
        regimes = hmm.predict(ret_full, vol_full)

    feature_array = feats_norm.values.astype(np.float32)
    return FeaturePipelineOutput(
        prices=prices_aligned,
        features=feats_norm,
        feature_array=feature_array,
        regimes=regimes,
        feature_names=list(feats_norm.columns),
        window_size=window_size,
    )


def make_observation_matrix(
    feature_array: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Stack flattened windows for use with the backtester. Portfolio state is
    appended as zeros; the backtester does not need them when policies don't use
    them, but RL policies expect the same shape they trained on.
    Output shape: (T - window_size, window_size * F + 3)
    """
    T, F = feature_array.shape
    n = T - window_size
    if n <= 0:
        raise ValueError("not enough data for one full window")
    out = np.empty((n, window_size * F + 3), dtype=np.float32)
    for i in range(n):
        flat = feature_array[i : i + window_size].ravel()
        # Portfolio state placeholders: filled with zeros for backtest;
        # the backtester is stateful but doesn't recompute these.
        out[i, :-3] = flat
        out[i, -3:] = 0.0
    return out
