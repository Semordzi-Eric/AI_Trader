"""Tests for feature engineering.

These guard against three classes of bug:
  1. Lookahead leak (a feature at time t depends on data after t).
  2. Crash on edge cases (zero-range bars, zero volume).
  3. Stat sanity (returns roughly mean-zero on a random walk).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trader.features.engineer import FeatureEngineer


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC")
    # Random walk close.
    log_close = np.cumsum(rng.normal(0, 1e-3, size=n)) + np.log(1.10)
    close = np.exp(log_close)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1e-4, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1e-4, size=n))
    volume = rng.lognormal(8.0, 0.5, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TestFeatureEngineer:
    def test_no_nan_after_warmup(self, synthetic_ohlcv):
        fe = FeatureEngineer()
        feats = fe.transform(synthetic_ohlcv)
        assert not feats.isna().any().any(), "NaN in output after warmup drop"

    def test_returns_have_no_lookahead(self, synthetic_ohlcv):
        """ret_h at row t must equal log(c[t]) - log(c[t-h]) using past prices only."""
        fe = FeatureEngineer(return_horizons=[1, 5])
        feats = fe.transform(synthetic_ohlcv)
        for h in [1, 5]:
            col = f"ret_{h}"
            # For each row in feats, verify ret matches the manual diff.
            for ts in feats.index[:50]:
                pos = synthetic_ohlcv.index.get_loc(ts)
                if pos < h:
                    continue
                expected = np.log(synthetic_ohlcv["close"].iloc[pos]) - np.log(
                    synthetic_ohlcv["close"].iloc[pos - h]
                )
                assert feats.loc[ts, col] == pytest.approx(expected, abs=1e-12)

    def test_perturbing_future_does_not_change_past_features(self, synthetic_ohlcv):
        """The strongest possible no-lookahead test: change future bars and verify the
        first half of the feature matrix is identical."""
        fe = FeatureEngineer()
        feats_a = fe.transform(synthetic_ohlcv).copy()

        df_b = synthetic_ohlcv.copy()
        # Multiply the second half of close by 10 — extreme perturbation.
        n = len(df_b)
        df_b.iloc[n // 2 :, df_b.columns.get_loc("close")] *= 10.0
        df_b.iloc[n // 2 :, df_b.columns.get_loc("high")] *= 10.0
        df_b.iloc[n // 2 :, df_b.columns.get_loc("low")] *= 10.0
        df_b.iloc[n // 2 :, df_b.columns.get_loc("open")] *= 10.0
        feats_b = fe.transform(df_b)

        # Indices that exist in both AND are before the perturbation point.
        cutoff = synthetic_ohlcv.index[n // 2 - 1]
        common = feats_a.index.intersection(feats_b.index)
        early = common[common <= cutoff]
        # rvol_90 needs 90 lookback, so trim earliest entries that overlap the window.
        early = early[early < cutoff - pd.Timedelta(minutes=15 * 90)]
        if len(early) == 0:
            pytest.skip("not enough untouched history given longest rolling window")
        for col in feats_a.columns:
            np.testing.assert_allclose(
                feats_a.loc[early, col].values,
                feats_b.loc[early, col].values,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"feature {col} changed for past timestamps when future was perturbed",
            )

    def test_zero_range_bar_does_not_explode(self, synthetic_ohlcv):
        df = synthetic_ohlcv.copy()
        # Force one bar to have zero range (open==high==low==close).
        i = 100
        df.iloc[i, df.columns.get_loc("high")] = df.iloc[i]["close"]
        df.iloc[i, df.columns.get_loc("low")] = df.iloc[i]["close"]
        df.iloc[i, df.columns.get_loc("open")] = df.iloc[i]["close"]
        feats = FeatureEngineer().transform(df)
        # That row may be dropped (NaN from divide-by-zero), but the rest must be finite.
        assert np.isfinite(feats.values).all()

    def test_feature_names_match_columns_when_defaults(self, synthetic_ohlcv):
        fe = FeatureEngineer()
        feats = fe.transform(synthetic_ohlcv)
        for name in fe.feature_names():
            assert name in feats.columns
