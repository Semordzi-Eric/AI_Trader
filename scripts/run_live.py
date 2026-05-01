"""Run the live trading loop against MetaTrader 5 (or paper).

This is the only script in the repo that touches a real broker.
Pre-flight checks before running on a real account:

  * The Windows host has MT5 installed and the python `MetaTrader5` package.
  * MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_TERMINAL_PATH env vars are set.
  * The agent zip referenced by `--agent` was *trained on the same symbol &
    timeframe* and uses the same feature set.
  * `artifacts/KILL` does NOT exist (kill-switch off). Touching that file at
    any time during the run halts trading and flattens positions.
  * You have run `--paper` for at least 24h and reviewed the trade ledger.

Example (paper, default):
    python scripts/run_live.py --config configs/live.yaml --agent artifacts/models/ppo_agent.zip --paper

Example (real money):
    python scripts/run_live.py --config configs/live.yaml --agent artifacts/models/ppo_agent.zip
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
import sys

from ai_trader.backtest.policies import BaselinePolicy, RLPolicy
from ai_trader.features.engineer import FeatureEngineer
from ai_trader.features.normalizer import RollingZScore
from ai_trader.live.broker import MT5Broker, PaperBroker
from ai_trader.live.runtime import LiveRuntime
from ai_trader.risk.manager import RiskConfig, RiskManager
from ai_trader.rl.agent import PPOAgent
from ai_trader.utils.config import load_config
from ai_trader.utils.logging_setup import get_logger
from ai_trader.utils.seeding import seed_everything

logger = get_logger(__name__)


def _confirm_real_money(symbol: str) -> None:
    """Last-line-of-defence before sending real orders."""
    sys.stdout.write(
        f"\n⚠️  You are about to run LIVE on {symbol} against a real broker.\n"
        f"    Type 'I ACCEPT' (exact, all caps) to continue, anything else to abort: "
    )
    sys.stdout.flush()
    line = sys.stdin.readline().strip()
    if line != "I ACCEPT":
        print("Aborted.")
        raise SystemExit(2)


def main() -> int:
    p = argparse.ArgumentParser(description="Run the live trading runtime.")
    p.add_argument("--config", type=str, default="configs/live.yaml")
    p.add_argument("--agent", type=str, default=None,
                   help="PPO model zip; if absent, BaselinePolicy (always flat).")
    p.add_argument("--paper", action="store_true",
                   help="Use PaperBroker simulator instead of MT5.")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip the type-to-confirm prompt (paper only).")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    live_cfg = cfg.get("live", {})
    symbol = live_cfg.get("symbol", "EURUSD")
    timeframe = cfg.get("data", {}).get("timeframe", "M15")
    feat = cfg["features"]

    if not args.paper and not args.no_confirm:
        _confirm_real_money(symbol)

    # Broker
    if args.paper:
        broker = PaperBroker(symbol=symbol)
        logger.info("Using PaperBroker (simulation)")
    else:
        broker = MT5Broker(symbol=symbol)
        logger.info("Using MT5Broker → real account")
    broker.connect()

    # Policy
    if args.agent:
        agent = PPOAgent(env_factory=lambda: None)  # type: ignore[arg-type]
        agent.load(args.agent)
        policy = RLPolicy(agent=agent, deterministic=True)
        logger.info("Loaded RL agent from %s", args.agent)
    else:
        policy = BaselinePolicy()
        logger.warning("No --agent supplied; using BaselinePolicy (always flat).")

    # Risk
    rc = cfg.get("risk", {})
    risk = RiskManager(RiskConfig(
        max_position_units=float(rc.get("max_position_units", 1.0)),
        max_open_positions=int(rc.get("max_open_positions", 1)),
        max_daily_loss_pct=float(rc.get("max_daily_loss_pct", 2.0)),
        max_drawdown_pct=float(rc.get("max_drawdown_pct", 10.0)),
        max_consecutive_losses=int(rc.get("max_consecutive_losses", 5)),
        circuit_breaker_vol_mult=float(rc.get("circuit_breaker_vol_mult", 5.0)),
        kill_switch_path=rc.get("kill_switch_path", "artifacts/KILL"),
    ))

    # Features
    fe = FeatureEngineer(
        return_horizons=feat["return_horizons"],
        vol_windows=feat["vol_windows"],
    )
    norm = RollingZScore(window=int(feat.get("normalize_window", 500)))

    runtime = LiveRuntime(
        broker=broker,
        policy=policy,
        risk=risk,
        symbol=symbol,
        timeframe=timeframe,
        feature_engineer=fe,
        normalizer=norm,
        window_size=int(feat["window_size"]),
        poll_seconds=int(live_cfg.get("poll_seconds", 5)),
        bars_to_pull=int(live_cfg.get("bars_to_pull", 1000)),
        state_path=live_cfg.get("state_path", "artifacts/live_state.json"),
    )

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; flushing.")
    finally:
        broker.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
