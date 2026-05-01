"""Run a backtest end-to-end: load data → build features → run a Policy.

Supported policies:
  baseline  - always flat (Sharpe should be ~0)
  random    - uniform random position
  trend     - sign-of-recent-return rule (non-ML baseline)
  rl        - load a saved PPO agent and run it deterministically

All policies share the same observation layout produced by
`make_observation_matrix`; only what they do with it differs.

Example:
    python scripts/run_backtest.py \\
        --config configs/default.yaml \\
        --csv artifacts/data/synthetic.csv \\
        --policy trend \\
        --report artifacts/reports/runs/trend.json
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
import json
from pathlib import Path

import pandas as pd

from ai_trader.backtest.engine import EventDrivenBacktester
from ai_trader.backtest.policies import (
    BaselinePolicy,
    Policy,
    RandomPolicy,
    RLPolicy,
    TrendFollowPolicy,
)
from ai_trader.data.sources import CSVSource, SyntheticSource
from ai_trader.risk.manager import RiskConfig, RiskManager
from ai_trader.training.pipeline import build_features_pipeline, make_observation_matrix
from ai_trader.utils.config import load_config
from ai_trader.utils.logging_setup import get_logger
from ai_trader.utils.seeding import seed_everything

logger = get_logger(__name__)


def _load_data(args, cfg) -> pd.DataFrame:
    if args.csv:
        return CSVSource(path=args.csv).load()
    # Fallback so CLI runs out-of-the-box.
    return SyntheticSource(
        n_bars=20_000,
        timeframe_minutes=15,
        seed=int(cfg.get("seed", 42)),
    ).load()


def _make_policy(name: str, agent_path: str | None, feature_names: list[str],
                 window_size: int) -> Policy:
    name = name.lower()
    if name == "baseline":
        return BaselinePolicy()
    if name == "random":
        return RandomPolicy(seed=0)
    if name == "trend":
        try:
            idx = feature_names.index("trend_strength")
        except ValueError:
            # Fall back to a long-horizon return as the trend proxy.
            try:
                idx = feature_names.index("ret_60")
            except ValueError:
                idx = next((i for i, n in enumerate(feature_names) if n.startswith("ret_")), 0)
        return TrendFollowPolicy(
            trend_idx=idx,
            n_features=len(feature_names),
            window_size=window_size,
            threshold=0.1,
        )
    if name == "rl":
        if not agent_path:
            raise SystemExit("--agent is required for --policy rl")
        # Lazy import: avoid pulling torch/SB3 unless we actually need them.
        from ai_trader.rl.agent import PPOAgent
        agent = PPOAgent(env_factory=lambda: None)  # type: ignore[arg-type]
        agent.load(agent_path)
        return RLPolicy(agent=agent, deterministic=True)
    raise SystemExit(f"unknown policy: {name}")


def main() -> int:
    p = argparse.ArgumentParser(description="Run an event-driven backtest.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--csv", type=str, default=None, help="OHLCV CSV; if absent, synthetic.")
    p.add_argument("--policy", type=str, default="trend",
                   choices=["baseline", "random", "trend", "rl"])
    p.add_argument("--agent", type=str, default=None, help="Path to PPO model zip (policy=rl).")
    p.add_argument("--report", type=str, default=None, help="Optional JSON report path.")
    p.add_argument("--no-risk", action="store_true", help="Disable the RiskManager overlay.")
    p.add_argument("--no-regime", action="store_true",
                   help="Skip the HMM regime model (useful when hmmlearn not installed).")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    df = _load_data(args, cfg)
    feat = cfg["features"]
    use_regime = bool(feat.get("regime", {}).get("enabled", True)) and not args.no_regime
    pipe = build_features_pipeline(
        df=df,
        return_horizons=feat["return_horizons"],
        vol_windows=feat["vol_windows"],
        window_size=int(feat["window_size"]),
        normalize_window=int(feat.get("normalize_window", 500)),
        use_regime=use_regime,
        n_regimes=int(feat.get("regime", {}).get("n_states", 3)),
    )
    obs = make_observation_matrix(pipe.feature_array, pipe.window_size)

    policy = _make_policy(args.policy, args.agent, pipe.feature_names, pipe.window_size)

    bt_cfg = cfg["backtest"]
    risk: RiskManager | None = None
    if not args.no_risk:
        rc = cfg.get("risk", {})
        risk = RiskManager(RiskConfig(
            max_position_units=float(rc.get("max_position_units", 1.0)),
            max_open_positions=int(rc.get("max_open_positions", 1)),
            max_daily_loss_pct=float(rc.get("max_daily_loss_pct", 2.0)),
            max_drawdown_pct=float(rc.get("max_drawdown_pct", 15.0)),
            max_consecutive_losses=int(rc.get("max_consecutive_losses", 5)),
            circuit_breaker_vol_mult=float(rc.get("circuit_breaker_vol_mult", 5.0)),
            kill_switch_path=rc.get("kill_switch_path", "artifacts/KILL"),
        ))

    max_pos = float(cfg.get("rl", {}).get("action", {}).get("max_position", 1.0))
    bt = EventDrivenBacktester(
        initial_equity=float(bt_cfg.get("initial_equity", 100_000.0)),
        contract_size=float(bt_cfg.get("contract_size", 100_000.0)),
        spread_pips=float(bt_cfg.get("spread_pips", 0.8)),
        commission_per_lot=float(bt_cfg.get("commission_per_lot", 7.0)),
        slippage_pips=float(bt_cfg.get("slippage_pips", 0.3)),
        point_value=float(bt_cfg.get("point_value", 1e-4)),
        latency_bars=int(bt_cfg.get("latency_bars", 1)),
        max_position=max_pos,
        risk=risk,
    )

    result = bt.run(
        prices=pipe.prices,
        observations=obs,
        policy=policy,
        window_size=pipe.window_size,
    )
    summary = result.summary()
    print(json.dumps({"policy": args.policy, **summary}, indent=2, default=str))

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        equity_csv = out.with_suffix(".equity.csv")
        result.equity.to_csv(equity_csv)
        with out.open("w") as fh:
            json.dump(
                {
                    "policy": args.policy,
                    "summary": summary,
                    "equity_csv": str(equity_csv),
                    "metadata": result.metadata,
                },
                fh,
                indent=2,
                default=str,
            )
        logger.info("Wrote report → %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
