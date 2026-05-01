"""Train a PPO agent on the TradingEnv.

Operational notes:
  * We only train on the train split (defined by config dates or 70%
    fractional fallback). Val/test segments are not seen by training.
  * `n_envs` controls vectorization. With CPU-bound envs, 4-8 is the
    typical sweet spot. Use `--use-subproc` on machines where the GIL is
    a bottleneck.
  * SB3 logs to stdout; the package logger writes to disk.

Example:
    python scripts/train_rl.py \\
        --config configs/default.yaml \\
        --csv artifacts/data/synthetic.csv \\
        --timesteps 100000 \\
        --out artifacts/models/ppo_agent.zip
"""
from __future__ import annotations

# --- bootstrap: make the package importable when running this file directly ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parents[1]
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
# --- end bootstrap ---

import argparse
from pathlib import Path

import pandas as pd

from ai_trader.data.sources import CSVSource, SyntheticSource
from ai_trader.data.splitter import TimeSplitter
from ai_trader.rl.agent import PPOAgent
from ai_trader.rl.env import EnvConfig, TradingEnv
from ai_trader.training.pipeline import build_features_pipeline
from ai_trader.utils.config import load_config
from ai_trader.utils.logging_setup import get_logger
from ai_trader.utils.seeding import seed_everything

logger = get_logger(__name__)


def _load_data(args, cfg) -> pd.DataFrame:
    if args.csv:
        return CSVSource(path=args.csv).load()
    return SyntheticSource(n_bars=20_000, seed=int(cfg.get("seed", 42))).load()


def _train_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    train_end = data_cfg.get("train_end")
    val_end = data_cfg.get("val_end")
    if train_end and val_end:
        try:
            return TimeSplitter(train_end=train_end, val_end=val_end).split(df).train
        except ValueError:
            logger.warning("Date split outside data; falling back to 70%% fraction.")
    return df.iloc[: int(len(df) * 0.70)]


def _make_env_config(cfg: dict) -> EnvConfig:
    rl = cfg.get("rl", {})
    bt = cfg.get("backtest", {})
    feat = cfg.get("features", {})
    reward = rl.get("reward", {})
    action = rl.get("action", {})
    return EnvConfig(
        window_size=int(feat.get("window_size", 64)),
        initial_equity=float(bt.get("initial_equity", 100_000.0)),
        contract_size=float(bt.get("contract_size", 100_000.0)),
        max_position=float(action.get("max_position", 1.0)),
        spread_pips=float(bt.get("spread_pips", 0.8)),
        commission_per_lot=float(bt.get("commission_per_lot", 7.0)),
        slippage_pips=float(bt.get("slippage_pips", 0.3)),
        point_value=float(bt.get("point_value", 1e-4)),
        pnl_weight=float(reward.get("pnl_weight", 1.0)),
        drawdown_penalty=float(reward.get("drawdown_penalty", 0.5)),
        vol_penalty=float(reward.get("vol_penalty", 0.1)),
        turnover_penalty=float(reward.get("turnover_penalty", 5e-4)),
        holding_bonus=float(reward.get("holding_bonus", 0.0)),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Train a PPO agent on the trading env.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--out", type=str, default="artifacts/models/ppo_agent.zip")
    p.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    p.add_argument("--n-envs", type=int, default=None, help="Override number of parallel envs.")
    p.add_argument("--use-subproc", action="store_true", help="Use SubprocVecEnv (CPU speedup).")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    df = _load_data(args, cfg)
    train_df = _train_split(df, cfg)

    feat = cfg["features"]
    pipe = build_features_pipeline(
        df=train_df,
        return_horizons=feat["return_horizons"],
        vol_windows=feat["vol_windows"],
        window_size=int(feat["window_size"]),
        normalize_window=int(feat.get("normalize_window", 500)),
        use_regime=bool(feat.get("regime", {}).get("enabled", True)),
        n_regimes=int(feat.get("regime", {}).get("n_states", 3)),
    )

    env_cfg = _make_env_config(cfg)

    def env_factory():
        return TradingEnv(
            prices=pipe.prices,
            features=pipe.feature_array,
            config=env_cfg,
            feature_names=pipe.feature_names,
        )

    rl_cfg = cfg.get("rl", {})
    agent = PPOAgent(
        env_factory=env_factory,
        n_envs=int(args.n_envs or rl_cfg.get("n_envs", 8)),
        total_timesteps=int(args.timesteps or rl_cfg.get("total_timesteps", 200_000)),
        learning_rate=float(rl_cfg.get("learning_rate", 3e-4)),
        n_steps=int(rl_cfg.get("n_steps", 1024)),
        batch_size=int(rl_cfg.get("batch_size", 256)),
        n_epochs=int(rl_cfg.get("n_epochs", 10)),
        gamma=float(rl_cfg.get("gamma", 0.995)),
        gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
        ent_coef=float(rl_cfg.get("ent_coef", 0.01)),
        clip_range=float(rl_cfg.get("clip_range", 0.2)),
        seed=int(cfg.get("seed", 42)),
        use_subproc=bool(args.use_subproc),
    )
    agent.train(save_path=args.out)
    logger.info("Training complete. Model saved to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
