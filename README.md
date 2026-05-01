# AI Trader

A modular, production-grade research framework for building adaptive algorithmic trading
systems. It combines a supervised directional-bias model (Transformer) with a
Reinforcement Learning policy (PPO) for execution decisions, walk-forward validation,
event-driven backtesting with realistic frictions, a meta-controller for regime-aware
agent selection, drift detection, and a Streamlit dashboard. A MetaTrader 5 live-trading
adapter with a hard kill-switch is included.

## Honest expectations

This repository is a **framework**, not a money-printer. No code can guarantee profits
in markets — alpha comes from data, hypotheses, and disciplined evaluation, not
architecture alone. Use this as a rigorous harness for your own research.

What's actually here, working end-to-end on synthetic data out of the box:

- Data pipeline with a synthetic OHLCV generator (drop-in `CSVSource` for real data)
- Microstructure-flavored features (no lagging RSI/MACD): returns at multiple horizons,
  realized volatility, range / body / wick ratios, volume-of-trade proxies, regime
  labels via a Hidden Markov Model on returns
- Transformer encoder for directional bias (PyTorch)
- Custom Gymnasium trading environment with continuous position sizing,
  transaction costs, slippage, and drawdown-aware rewards
- PPO agent (Stable-Baselines3) with vectorized environments
- Walk-forward training with regime-segmented ensemble
- Meta-model (gradient-boosted) selecting the best agent for current conditions
- Event-driven backtester independent of training (catches lookahead bugs)
- Performance analytics: Sharpe, Sortino, Calmar, max drawdown, profit factor,
  expectancy, turnover
- Drift detection (Page–Hinkley + KS test on feature distributions)
- Streamlit dashboard for backtests, comparison, and controls
- MT5 live adapter with position limits, daily loss cutoff, latency-aware fills, and
  a kill-switch
- Configurable via YAML; reproducible with pinned seeds

## Project layout

```
ai_trader/
├── ai_trader/
│   ├── data/           # data sources, loaders, splitters
│   ├── features/       # feature engineering, regime detection, normalization
│   ├── models/         # supervised models (Transformer)
│   ├── rl/             # gym env, PPO wrapper, reward shaping
│   ├── training/       # walk-forward, ensemble training
│   ├── meta/           # meta-controller for agent selection, drift detection
│   ├── backtest/       # event-driven engine, execution simulation
│   ├── risk/           # position limits, kill-switch, circuit breaker
│   ├── live/           # MT5 adapter, runtime loop
│   ├── dashboard/      # Streamlit app
│   └── utils/          # config, logging, metrics, seeding
├── configs/            # YAML configs
├── scripts/            # entry-point CLIs
├── tests/              # unit tests
└── artifacts/          # models, logs, reports (gitignored)
```

## Quick start

```bash
# 1. Install (the lighter modules — utils, features, risk, backtester baselines —
#    only need numpy/pandas/scipy/PyYAML and run without torch/SB3/hmmlearn).
pip install -r requirements.txt

# 2. Generate synthetic data
python scripts/generate_synthetic.py --bars 50000 --out artifacts/data/synth.csv

# 3. Smoke test: baseline backtest end-to-end (no training, no heavy deps required
#    if you pass --no-regime to skip the HMM)
python scripts/run_backtest.py \
    --csv artifacts/data/synth.csv \
    --policy baseline \
    --no-risk --no-regime

# 4. A non-ML baseline that actually trades
python scripts/run_backtest.py \
    --csv artifacts/data/synth.csv \
    --policy trend \
    --no-regime \
    --report artifacts/reports/runs/trend.json

# 5. Train the supervised directional model
python scripts/train_supervised.py --csv artifacts/data/synth.csv

# 6. Train the RL agent
python scripts/train_rl.py --csv artifacts/data/synth.csv --timesteps 50000

# 7. Walk-forward evaluation (the only honest evaluation)
python scripts/walk_forward.py --csv artifacts/data/synth.csv --timesteps-per-fold 30000

# 8. Launch dashboard
streamlit run ai_trader/dashboard/app.py

# 9. Paper-trade against a simulated broker before going live
python scripts/run_live.py --paper --agent artifacts/models/ppo_agent.zip

# 10. Live (requires MT5 + env vars set; read docs/SAFETY.md FIRST)
python scripts/run_live.py --agent artifacts/models/ppo_agent.zip
```

### Running the tests

The metrics, features, and risk tests don't need torch / SB3 / gymnasium:

```bash
pip install pytest
pytest tests/ -v
```

## Methodology notes

**Why no RSI/MACD?** Lagging indicators are deterministic functions of price already in
the input; a sufficiently expressive model recovers them for free. Including them adds
noise and false confidence. We use raw returns at multiple horizons and microstructure
proxies instead.

**Why walk-forward over k-fold?** Markets are non-stationary. K-fold leaks future
information into training. Walk-forward replicates the only validation regime that
matters: train on the past, evaluate on the future, repeat.

**Why a meta-controller?** A single policy underfits across regimes. We train
specialists per regime (trending / mean-reverting / volatile) and let a meta-model
arbitrate based on current features. This is closer to how desks actually allocate.

**Why a separate event-driven backtester?** Training environments are convenient but
notorious for subtle lookahead leaks (e.g., normalizing with statistics from the full
window). Re-evaluating with an independent engine that consumes data tick-by-tick is
the only honest test.

## Safety

The live adapter enforces, at minimum: max position size, max daily loss (% of equity),
max consecutive losses, max open positions, and a manual kill-switch file
(`artifacts/KILL`). Touch that file from anywhere — the loop halts on the next tick and
flattens. Read `docs/SAFETY.md` before going live with real capital.

## License

MIT. See `LICENSE`.
