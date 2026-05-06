"""Walk-forward training & evaluation driver.

Trains an agent on a rolling train window, validates on the immediately-following
val window (used for early stopping / model selection), then evaluates on the
out-of-sample test window. Repeats over the dataset, producing a stitched OOS
equity curve and per-window metrics.

This is the only out-of-sample story we trust. Reported metrics from full-sample
training on the same data they were trained on tell you nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

from ..backtest.engine import EventDrivenBacktester
from ..backtest.policies import RLPolicy
from ..data.splitter import walk_forward_windows
from ..rl.agent import PPOAgent
from ..rl.env import EnvConfig, TradingEnv
from ..utils.logging_setup import get_logger
from ..utils.metrics import summary_stats
from .pipeline import build_features_pipeline, make_observation_matrix

logger = get_logger(__name__)


@dataclass
class WalkForwardResult:
    fold_summaries: List[dict]
    oos_equity: pd.Series
    oos_trade_pnls: np.ndarray
    overall_summary: dict


def walk_forward_run(
    df: pd.DataFrame,
    cfg: dict,
    save_dir: str | Path = "artifacts/models/walk_forward",
    timesteps_per_fold: int = 50_000,
) -> WalkForwardResult:
    """Run walk-forward training on the dataframe. Returns stitched OOS metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    win_cfg = cfg["walk_forward"]
    feat_cfg = cfg["features"]
    bt_cfg = cfg["backtest"]
    rl_cfg = cfg["rl"]

    fold_summaries: List[dict] = []
    oos_equity_segments: List[pd.Series] = []
    all_trade_pnls: List[float] = []

    fold_id = 0
    for split in walk_forward_windows(
        df,
        train_bars=win_cfg["train_bars"],
        val_bars=win_cfg["val_bars"],
        test_bars=win_cfg["test_bars"],
        step_bars=win_cfg["step_bars"],
    ):
        fold_id += 1
        logger.info("===== walk-forward fold %d =====", fold_id)
        logger.info(
            "train: %s rows  val: %s rows  test: %s rows",
            f"{len(split.train):,}", f"{len(split.val):,}", f"{len(split.test):,}",
        )

        # Build features for the FULL fold range, fitting normalizers/regimes only on train.
        full_fold = pd.concat([split.train, split.val, split.test])
        out = build_features_pipeline(
            full_fold,
            return_horizons=feat_cfg["return_horizons"],
            vol_windows=feat_cfg["vol_windows"],
            window_size=feat_cfg["window_size"],
            normalize_window=feat_cfg.get("normalize_window", 500),
            use_regime=feat_cfg.get("regime", {}).get("enabled", True),
            n_regimes=feat_cfg.get("regime", {}).get("n_states", 3),
            fit_regime_on=split.train,
        )

        # Slice to train segment for env
        train_mask = out.prices.index <= split.train.index[-1]
        train_prices = out.prices.loc[train_mask]
        train_features = out.feature_array[train_mask]

        env_cfg = EnvConfig(
            window_size=feat_cfg["window_size"],
            initial_equity=bt_cfg["initial_equity"],
            contract_size=bt_cfg["contract_size"],
            max_position=rl_cfg["action"]["max_position"],
            spread_pips=bt_cfg["spread_pips"],
            commission_per_lot=bt_cfg["commission_per_lot"],
            slippage_pips=bt_cfg["slippage_pips"],
            point_value=bt_cfg["point_value"],
            pnl_weight=rl_cfg["reward"]["pnl_weight"],
            drawdown_penalty=rl_cfg["reward"]["drawdown_penalty"],
            vol_penalty=rl_cfg["reward"]["vol_penalty"],
            turnover_penalty=rl_cfg["reward"]["turnover_penalty"],
            holding_bonus=rl_cfg["reward"]["holding_bonus"],
        )

        def env_factory(_p=train_prices, _f=train_features, _c=env_cfg):
            return TradingEnv(_p, _f, _c)

        agent = PPOAgent(
            env_factory=env_factory,
            n_envs=rl_cfg["n_envs"],
            total_timesteps=timesteps_per_fold,
            learning_rate=rl_cfg["learning_rate"],
            n_steps=rl_cfg["n_steps"],
            batch_size=rl_cfg["batch_size"],
            n_epochs=rl_cfg["n_epochs"],
            gamma=rl_cfg["gamma"],
            gae_lambda=rl_cfg["gae_lambda"],
            ent_coef=rl_cfg["ent_coef"],
            clip_range=rl_cfg["clip_range"],
        )
        model_path = save_dir / f"fold_{fold_id:03d}.zip"
        agent.train(save_path=model_path)

        # Build OOS observations on the test segment (using normalizer fit on full fold).
        # We pull the test slice of the prepared output.
        test_mask = out.prices.index > split.val.index[-1]
        test_prices = out.prices.loc[test_mask]
        test_features = out.feature_array[test_mask]

        if len(test_features) <= feat_cfg["window_size"] + 2:
            logger.warning("test fold too small after warmup, skipping fold %d", fold_id)
            continue

        obs_matrix = make_observation_matrix(test_features, feat_cfg["window_size"])

        bt = EventDrivenBacktester(
            initial_equity=bt_cfg["initial_equity"],
            contract_size=bt_cfg["contract_size"],
            spread_pips=bt_cfg["spread_pips"],
            commission_per_lot=bt_cfg["commission_per_lot"],
            slippage_pips=bt_cfg["slippage_pips"],
            point_value=bt_cfg["point_value"],
            latency_bars=bt_cfg["latency_bars"],
            max_position=rl_cfg["action"]["max_position"],
        )
        result = bt.run(test_prices, obs_matrix, RLPolicy(agent), feat_cfg["window_size"])
        s = result.summary()
        s["fold"] = fold_id
        fold_summaries.append(s)
        oos_equity_segments.append(result.equity)
        all_trade_pnls.extend(result.trade_pnls.tolist())
        logger.info("fold %d OOS sharpe=%.2f maxDD=%.2f%%", fold_id, s["sharpe"], s["max_drawdown_pct"])

    if not oos_equity_segments:
        raise RuntimeError("No completed walk-forward folds — data too short for config.")

    # Stitch equity by chaining returns (each fold starts at initial_equity again).
    init = bt_cfg["initial_equity"]
    pieces = []
    running = init
    for seg in oos_equity_segments:
        rets = (seg / seg.shift(1)).fillna(seg.iloc[0] / init)
        compounded = running * rets.cumprod()
        pieces.append(compounded)
        running = float(compounded.iloc[-1])
    stitched = pd.concat(pieces).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]

    overall = summary_stats(stitched, np.array(all_trade_pnls))
    return WalkForwardResult(
        fold_summaries=fold_summaries,
        oos_equity=stitched,
        oos_trade_pnls=np.array(all_trade_pnls),
        overall_summary=overall,
    )
