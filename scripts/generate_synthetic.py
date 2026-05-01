"""Generate synthetic OHLCV bars and write them to CSV.

Useful for offline development, smoke tests, and CI. The generator is the
3-regime Markov GBM with Student-t innovations defined in
`ai_trader.data.sources.SyntheticSource`.

Example:
    python scripts/generate_synthetic.py --bars 50000 --out artifacts/data/synth.csv
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

from ai_trader.data.sources import SyntheticSource
from ai_trader.utils.logging_setup import get_logger
from ai_trader.utils.seeding import seed_everything

logger = get_logger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic OHLCV CSV.")
    p.add_argument("--bars", type=int, default=50_000, help="Number of bars to generate.")
    p.add_argument("--timeframe", type=int, default=15, help="Timeframe in minutes.")
    p.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD).")
    p.add_argument("--initial-price", type=float, default=1.10, help="Starting price.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--out",
        type=str,
        default="artifacts/data/synthetic.csv",
        help="Output CSV path (parents created if missing).",
    )
    args = p.parse_args()

    seed_everything(args.seed)
    src = SyntheticSource(
        n_bars=args.bars,
        timeframe_minutes=args.timeframe,
        start=args.start,
        seed=args.seed,
        initial_price=args.initial_price,
    )
    df = src.load()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Name the index column 'timestamp' so CSVSource picks it up cleanly.
    df.index.name = "timestamp"
    df.to_csv(out_path)
    logger.info("Wrote %s bars to %s", f"{len(df):,}", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
