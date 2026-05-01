"""Run a full walk-forward evaluation: rolling train → val → test windows.

This is the only honest evaluation. Single-split numbers are not enough
for a stochastic, regime-shifting market. The output is a stitched
out-of-sample equity curve with per-fold summaries, written to a JSON
report and a CSV.

Example:
    python scripts/walk_forward.py \\
        --config configs/default.yaml \\
        --csv artifacts/data/synthetic.csv \\
        --timesteps-per-fold 30000 \\
        --out artifacts/reports/walk_forward.json
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

from ai_trader.data.sources import CSVSource, SyntheticSource
from ai_trader.training.walk_forward import walk_forward_run
from ai_trader.utils.config import load_config
from ai_trader.utils.logging_setup import get_logger
from ai_trader.utils.seeding import seed_everything

logger = get_logger(__name__)


def _load_data(args, cfg) -> pd.DataFrame:
    if args.csv:
        return CSVSource(path=args.csv).load()
    syn = cfg.get("data", {}).get("synthetic", {})
    return SyntheticSource(
        n_bars=int(syn.get("n_bars", 20_000)),
        timeframe_minutes=int(cfg.get("data", {}).get("timeframe_minutes", 15)),
        start=syn.get("start", "2018-01-01"),
        seed=int(syn.get("seed", 42)),
        initial_price=float(syn.get("initial_price", 1.10)),
    ).load()


def main() -> int:
    p = argparse.ArgumentParser(description="Run walk-forward evaluation.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="artifacts/models/walk_forward")
    p.add_argument("--timesteps-per-fold", type=int, default=30_000)
    p.add_argument("--out", type=str, default="artifacts/reports/walk_forward.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    df = _load_data(args, cfg)

    result = walk_forward_run(
        df=df,
        cfg=cfg,
        save_dir=args.save_dir,
        timesteps_per_fold=args.timesteps_per_fold,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    equity_csv = out.with_suffix(".equity.csv")
    result.oos_equity.to_csv(equity_csv)
    with out.open("w") as fh:
        json.dump(
            {
                "fold_summaries": result.fold_summaries,
                "overall": result.overall_summary,
                "n_oos_trades": int(len(result.oos_trade_pnls)),
                "equity_csv": str(equity_csv),
            },
            fh,
            indent=2,
            default=str,
        )

    print(json.dumps({"overall": result.overall_summary, "n_folds": len(result.fold_summaries)}, indent=2, default=str))
    logger.info("Wrote walk-forward report → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
