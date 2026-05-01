# Architecture

This document describes how the system is structured, why each piece exists,
and how data and control flow through it. If you're going to modify the code,
read this first.

## High-level shape

```
                 ┌──────────────────────────────────────────────────────┐
                 │                      DATA LAYER                       │
                 │  CSVSource │ SyntheticSource │ MT5Source              │
                 └──────────────────────┬───────────────────────────────┘
                                        │ pd.DataFrame (OHLCV, UTC index)
                                        ▼
                 ┌──────────────────────────────────────────────────────┐
                 │                    FEATURES LAYER                     │
                 │  FeatureEngineer → RollingZScore → HMMRegimeDetector  │
                 └──────────────────────┬───────────────────────────────┘
                                        │ (T, F) float32 + regimes
                                        ▼
                 ┌──────────────────────┬───────────────────────────────┐
                 │       MODELS         │            RL                  │
                 │  TransformerForecaster│   TradingEnv → PPOAgent       │
                 │  (sign-of-return)     │   (continuous position)        │
                 └──────────────────────┴───────────────┬───────────────┘
                                                        │ Policy
                                                        ▼
                 ┌──────────────────────────────────────────────────────┐
                 │                    BACKTEST LAYER                     │
                 │  EventDrivenBacktester + RiskManager overlay          │
                 └──────────────────────┬───────────────────────────────┘
                                        │ BacktestResult
                                        ▼
            ┌───────────────────────────┼───────────────────────────────┐
            │                           │                                │
            ▼                           ▼                                ▼
    META-CONTROLLER             DRIFT DETECTOR                    DASHBOARD
   (LightGBM agent picker)    (Page-Hinkley + KS)            (Streamlit reports)

                                ┌──────────────────────┐
                                │    LIVE LAYER        │
                                │  LiveRuntime →       │
                                │  MT5Broker / Paper   │
                                └──────────────────────┘
```

## Directory map

```
ai_trader/
├── utils/         logging, config (YAML w/ inheritance), seeding, metrics
├── data/          DataSource ABC + CSV / Synthetic / MT5 implementations; splitters
├── features/      raw-price feature engineering; rolling z-score; HMM regimes
├── models/        Transformer forecaster + supervised trainer + windowed dataset
├── rl/            Gymnasium TradingEnv + PPOAgent (SB3 wrapper)
├── backtest/      Policy ABC + reference policies; EventDrivenBacktester
├── risk/          RiskConfig + RiskManager (kill-switch, DD, daily loss, vol breaker)
├── meta/          MetaController (regime → agent picker) + DriftDetector
├── training/      build_features_pipeline, walk_forward_run
├── live/          MT5Broker + PaperBroker; LiveRuntime poll loop
└── dashboard/     Streamlit app (backtest viewer, compare, live monitor, controls)
```

## Module-by-module rationale

### `utils/`

Boring, but used everywhere. The `Config` class supports a `inherits:` key so
`live.yaml` can override only the few values that differ from `default.yaml`.
`seed_everything` sets random/numpy/torch seeds *and* enables cuDNN determinism;
without that last bit, two runs of the same code produce different numbers and
debugging takes weeks.

### `data/`

Three sources behind a single `DataSource` ABC. `CSVSource` is the fast path
for offline work. `SyntheticSource` is a 3-regime Markov-switching GBM with
Student-t innovations (df=5) and intraday volume seasonality — it's enough
structure that strategies that don't work on it definitely won't work on real
data, and it costs nothing to generate. `MT5Source` is import-guarded so the
package loads on Linux.

### `features/`

Deliberately *not* using RSI/MACD/Bollinger/etc. The reasoning is in the module
docstring, but briefly: any deterministic transform of past prices is recoverable
by a sufficiently expressive model. Adding RSI just adds noise and consumes
representational capacity. We use multi-horizon log returns, realized vol at
several windows, intra-bar microstructure (body/wick/range), volume z-score,
return skew, a Sharpe-like trend strength, a spread proxy, and cyclical time
encodings.

`RollingZScore` normalizes using *only past* statistics. Past-only statistics
are a hard requirement; using full-sample mean/std is the most common silent
lookahead bug in published trading research.

`HMMRegimeDetector` (Gaussian HMM on returns and realized vol) labels each bar
with a discrete regime. We reorder the states by mean return so state IDs are
comparable across walk-forward windows.

### `models/`

A small Transformer (3 layers, 4 heads, d_model=64 by default). Input projection
→ sinusoidal positional encoding → encoder stack → mean pool over time → MLP →
scalar. Default target is sign-of-return with BCE loss. The trainer does early
stopping on val loss, AdamW + cosine LR, gradient clipping. Nothing exotic —
the value is in the *features* and the *evaluation*, not the architecture.

### `rl/`

`TradingEnv` (Gymnasium) exposes a continuous action in [-1, 1] interpreted as
the target position fraction. The reward is

```
reward = pnl_weight*ret − drawdown_penalty*max(0, dd − dd_threshold)
       − vol_penalty*recent_vol − turnover_penalty*|Δposition|
       + holding_bonus*|position|
```

The shaping matters: a pure-PnL reward learns reckless leverage. The drawdown
threshold is asymmetric (only penalize past 5%), which leaves room for natural
fluctuation while still discouraging blow-up trajectories.

`PPOAgent` is a thin wrapper around stable-baselines3 PPO. It supports
`DummyVecEnv` and `SubprocVecEnv` for parallel rollouts.

### `backtest/`

This is the only module where we trust the numbers. The training env makes
shortcuts for speed; the backtester walks bars sequentially, executes at the
*next* bar's open, charges spread + slippage + commission per side, supports
configurable latency in bars, and integrates the `RiskManager` overlay.

The strongest correctness check we have is: *the env and the backtester should
report the same Sharpe to within a few percent on the same data with the same
policy.* If they diverge by an order of magnitude, somebody is peeking at the
future.

### `risk/`

Conservative by design. Halts are sticky. The kill-switch is just "does the
file `artifacts/KILL` exist?" — touchable from any process, any user, any
language. Daily loss resets at the UTC day boundary. The vol circuit breaker
trips when realized vol over the last few bars exceeds N× the baseline. Every
halt records a reason string for the dashboard.

### `meta/`

`MetaController` is a LightGBM classifier that picks which agent to use given
the current feature vector. It's trained on labels of the form "the agent that
produced the highest forward-window PnL" computed from the walk-forward
results. This is one of the few places where a small, fast, gradient-boosted
model beats a deep one — the input is dense, low-dimensional, and the
relationship is non-monotonic.

`DriftDetector` combines two signals: Page-Hinkley on the PnL stream (online
change-point detection that auto-resets after a hit) and a Kolmogorov-Smirnov
two-sample test comparing the current feature distribution to a reference
window. When either fires, the runtime logs it and (optionally) triggers a
retrain.

### `training/`

`build_features_pipeline` is the function called from every script. It does
features → normalize → optional regimes, fitting the regime model only on the
training segment (or a default 60% prefix). `make_observation_matrix` builds
the flat-window observation layout used by both the env and the backtester —
both must see the same shape, and this function is the contract.

`walk_forward_run` is the only honest evaluation. K-fold cross-validation is
not provided because it leaks future data into training. Single-split metrics
are reported for development convenience but should not be the basis for a
production decision.

### `live/`

`BaseBroker` is the contract; `MT5Broker` is the real implementation; `PaperBroker`
is a deterministic simulator that lets you exercise the runtime loop without
sending real orders.

`LiveRuntime` polls every `poll_seconds`. Each iteration:

1. Pull the last N bars from the broker.
2. If a new bar has appeared, recompute features and the latest observation.
3. Ask the `RiskManager` whether to halt; if so, flatten and skip.
4. Ask the policy for a target position; clamp to limits.
5. Send / modify / close orders against the broker to reach target.
6. Persist state to `artifacts/live_state.json`. The broker's reported positions
   are the source of truth — we don't trust our cache if the process crashed.

### `dashboard/`

A Streamlit app with four pages: Backtest viewer (load a JSON report, see
equity + drawdown + per-trade stats), Compare (overlay two reports), Live
monitor (read `artifacts/live_state.json` and show position + recent trades),
and Controls (toggle the kill-switch, view halt reasons).

## End-to-end data flow (offline)

1. `scripts/generate_synthetic.py` → CSV at `artifacts/data/synthetic.csv`.
2. `scripts/run_backtest.py --policy trend` runs a non-ML baseline.
3. `scripts/train_rl.py` trains a PPO agent on the train split.
4. `scripts/run_backtest.py --policy rl --agent <path>` evaluates the trained agent.
5. `scripts/walk_forward.py` runs the full rolling evaluation.
6. The Streamlit dashboard renders the reports.

## End-to-end data flow (live)

1. Pre-trained agent zip on disk.
2. `scripts/run_live.py --paper` for shakedown.
3. `scripts/run_live.py` (real broker) after pre-flight checks (see `SAFETY.md`).
4. `LiveRuntime` polls bars, computes features the same way training did,
   asks the policy for a target position, applies risk, places/modifies/closes
   orders against MT5.
5. `artifacts/live_state.json` updates each tick; the dashboard reflects it.

## Invariants the code relies on

* All timestamps are UTC. The split logic, the daily loss reset, and the HMM
  state stability assume UTC.
* The observation layout is `[flat_window, position, equity_norm, drawdown]`
  in exactly that order, and the env and backtester both produce it.
* Walk-forward folds fit normalizers and HMMs on the *train* segment of each
  fold only. Anything else is a leak.
* The kill-switch path in `RiskConfig` is the same path the dashboard's
  Controls page touches.
