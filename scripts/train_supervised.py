"""Train the supervised Transformer forecaster on sign-of-return.

The model learns *direction* by default. Sign-of-return is harder than it
sounds: the no-information baseline is ~50% accuracy, and a model that
hits 53% on out-of-sample is already non-trivial. We do not chase
regression of returns directly — that's a noise-fitting trap on
financial data.

Train/val splits are taken time-wise from the data using the dates in
the config (`data.train_end`, `data.val_end`).

Example:
    python scripts/train_supervised.py \\
        --config configs/default.yaml \\
        --csv artifacts/data/synthetic.csv \\
        --out artifacts/models/transformer.pt
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

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ai_trader.data.sources import CSVSource, SyntheticSource
from ai_trader.data.splitter import TimeSplitter
from ai_trader.models.dataset import WindowDataset
from ai_trader.models.trainer import SupervisedTrainer
from ai_trader.models.transformer import TransformerForecaster
from ai_trader.training.pipeline import build_features_pipeline
from ai_trader.utils.config import load_config
from ai_trader.utils.logging_setup import get_logger
from ai_trader.utils.seeding import seed_everything

logger = get_logger(__name__)


def _load_data(args, cfg) -> pd.DataFrame:
    if args.csv:
        return CSVSource(path=args.csv).load()
    return SyntheticSource(n_bars=20_000, seed=int(cfg.get("seed", 42))).load()


def _split_by_dates_or_fraction(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Try date-based split (configs/default.yaml convention); fall back to 70/15
    fraction when the dates fall outside the data (e.g. synthetic).
    """
    data_cfg = cfg.get("data", {})
    train_end = data_cfg.get("train_end")
    val_end = data_cfg.get("val_end")
    if train_end and val_end:
        try:
            sw = TimeSplitter(train_end=train_end, val_end=val_end).split(df)
            return sw.train, sw.val
        except ValueError:
            logger.warning("Date split outside data range; falling back to fractional split.")
    n = len(df)
    return df.iloc[: int(n * 0.70)], df.iloc[int(n * 0.70) : int(n * 0.85)]


def main() -> int:
    p = argparse.ArgumentParser(description="Train the supervised Transformer.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--out", type=str, default="artifacts/models/transformer.pt")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    df = _load_data(args, cfg)
    train_df, val_df = _split_by_dates_or_fraction(df, cfg)

    feat = cfg["features"]
    train_pipe = build_features_pipeline(
        df=train_df,
        return_horizons=feat["return_horizons"],
        vol_windows=feat["vol_windows"],
        window_size=int(feat["window_size"]),
        normalize_window=int(feat.get("normalize_window", 500)),
        use_regime=False,
    )
    val_pipe = build_features_pipeline(
        df=val_df,
        return_horizons=feat["return_horizons"],
        vol_windows=feat["vol_windows"],
        window_size=int(feat["window_size"]),
        normalize_window=int(feat.get("normalize_window", 500)),
        use_regime=False,
    )

    # Target: 1-bar log return. BCE will use sign(y).
    def _bar_log_returns(prices: pd.DataFrame) -> np.ndarray:
        lp = np.log(prices["close"].to_numpy())
        return np.diff(lp, prepend=lp[0]).astype(np.float32)

    y_train = _bar_log_returns(train_pipe.prices)
    y_val = _bar_log_returns(val_pipe.prices)

    sup = cfg.get("supervised", {})
    ws = int(feat["window_size"])
    horizon = int(sup.get("horizon", 1))
    train_ds = WindowDataset(train_pipe.feature_array, y_train, window_size=ws, horizon=horizon)
    val_ds = WindowDataset(val_pipe.feature_array, y_val, window_size=ws, horizon=horizon)

    bs = int(args.batch_size or sup.get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = TransformerForecaster(
        n_features=train_pipe.feature_array.shape[1],
        d_model=int(sup.get("d_model", 64)),
        n_heads=int(sup.get("n_heads", 4)),
        n_layers=int(sup.get("n_layers", 3)),
        dropout=float(sup.get("dropout", 0.1)),
    )
    trainer = SupervisedTrainer(
        model=model,
        target_kind=sup.get("target", "sign_return"),
        lr=float(sup.get("lr", 3e-4)),
        weight_decay=float(sup.get("weight_decay", 1e-5)),
        epochs=int(args.epochs or sup.get("epochs", 20)),
        early_stopping_patience=int(sup.get("early_stopping_patience", 5)),
    )
    result = trainer.fit(train_loader, val_loader, save_path=args.out)
    print(json.dumps({"best_val_loss": result["best_val_loss"]}, indent=2))

    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as fh:
        json.dump(
            {"history": result["history"], "n_features": int(train_pipe.feature_array.shape[1])},
            fh,
            indent=2,
        )
    logger.info("Wrote training meta → %s", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
