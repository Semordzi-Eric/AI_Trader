# Safety: pre-live checklist

This document is the gate between *this is a working research framework* and
*real money is at risk*. The framework is a tool. The decisions about whether
to deploy, how much capital to risk, and what counts as evidence of edge are
yours.

Read every section. Do not skip ahead.

---

## 1. The honest expectation

A randomly-initialized RL agent has no edge over the spread. A trained agent
that looks profitable in-sample has, with overwhelming probability, no edge
out-of-sample. A trained agent that looks profitable in walk-forward backtests
on synthetic data has not been tested against real markets.

The default position is *the system is not profitable* until proven otherwise
on real, held-out data, repeatedly, across regimes, after costs.

If you are looking at backtest Sharpe > 2 on your first run, something is wrong.
The most likely thing wrong is a lookahead leak. Investigate before deploying.

---

## 2. Pre-live checklist

Do not run `scripts/run_live.py` against a real account until every box is
checked. Print this section. Tick it with a pen.

### Data

- [ ] Live data feed is the *same* symbol, timeframe, and source as training
  data. A model trained on Dukascopy EURUSD M15 deployed against MT5 EURUSD M15
  is two different distributions.
- [ ] Spread and commission in `configs/*.yaml` reflect your actual broker.
  Off-by-2x on cost assumptions is enough to flip a profitable strategy to
  unprofitable.
- [ ] You have at least 2 years of historical training data. Less than that
  and the agent has not seen enough regime variation.

### Model validation

- [ ] Walk-forward results show **out-of-sample** Sharpe > 0.5 across the
  majority of folds — not just one lucky fold.
- [ ] Out-of-sample profit factor > 1.2 across folds.
- [ ] The OOS equity curve does not have a single fold contributing >50% of
  total return.
- [ ] You have compared OOS metrics to the `BaselinePolicy` (flat) and
  `RandomPolicy` baselines. The agent beats both meaningfully, not by noise.
- [ ] You have eyeballed the trade ledger from a representative OOS window.
  The trades make sense — the agent is not trading on weekends, not flipping
  every bar, not accumulating one direction blindly.

### Engineering

- [ ] All tests in `tests/` pass: `pytest tests/`.
- [ ] You have run `scripts/run_live.py --paper` for at least 24 hours of
  market time and reviewed every order it placed.
- [ ] The kill-switch works: while paper-running, `touch artifacts/KILL` and
  verify the agent flattens within one poll cycle.
- [ ] `artifacts/live_state.json` persists across restarts: kill the runtime
  mid-session, restart it, confirm it picks up where it left off.
- [ ] Logs are being written to `artifacts/logs/` and contain every order.

### Risk

- [ ] `max_drawdown_pct` and `max_daily_loss_pct` in `configs/live.yaml` are
  values you can *actually* lose without it affecting your finances or judgment.
- [ ] The MT5 account has hard limits set at the broker level too — do not
  rely on the application-level risk gate alone.
- [ ] You have a separate account for live deployment, not your main account.
- [ ] Position sizing is small enough that a 100% loss on this account is
  recoverable.

### Operations

- [ ] You know how to reach the kill-switch from your phone (SSH, Tailscale,
  whatever — practice it before you need it).
- [ ] You have a written plan for what triggers a manual halt: drawdown
  thresholds, news events, weekend overnight risk, broker maintenance windows.
- [ ] Someone other than you knows the kill-switch path and the broker login,
  in case you are unreachable.

If any box is unchecked, do not deploy.

---

## 3. The kill-switch

The kill-switch is intentionally simple: a file on disk at the path in
`configs/live.yaml` (default `artifacts/KILL`). At every tick, the
`RiskManager` checks for the file. If present:

1. Trading is halted.
2. The runtime sends a flatten order for any open position.
3. No new orders are placed until the file is removed AND `RiskManager.reset_kill()`
   is called from a controlled context (the dashboard or a maintenance script).

To trigger:

```bash
touch /path/to/artifacts/KILL
```

To clear (only after you've reviewed why it was triggered):

```bash
rm /path/to/artifacts/KILL
```

The file mechanism is deliberately language- and process-agnostic. You can
trigger it from anything that can write a file: a cron job, a phone over SSH,
a Streamlit button, a panicked finger.

---

## 4. What the system will not do

- It will not size positions based on Kelly, Markowitz, or volatility targeting
  by default. The agent learns a position fraction in [-1, 1]; the
  `RiskManager` only applies a hard cap.
- It will not retrain itself live. Drift detection (`ai_trader.meta.drift`) can
  flag drift, but the retraining loop is not auto-triggered. This is
  intentional: a model that retrains on its own losing trades during a regime
  shift can amplify losses. Retraining is a manual operation.
- It will not avoid news events on its own. Calendar-aware halts are a future
  extension. For now, halt manually around scheduled high-impact news.
- It will not survive a broker outage gracefully beyond the next reconnect.
  The state file is restored on restart, but if the broker drops mid-execution,
  manual reconciliation may be required.

---

## 5. Failure modes to expect

These are not edge cases. These will happen.

- **Distribution shift**: Real markets in 2026 are not the markets you trained
  on. Sharpe ratios decay. Re-evaluate every month.
- **Cost creep**: Spreads widen during news. Commissions change. Your
  configured costs will drift from real costs.
- **Slippage on large positions**: The backtester models slippage as a
  constant pip adjustment. Real slippage is non-linear in position size.
- **Broker quirks**: Stop levels, freeze levels, weekend rollover charges,
  swap rates, dividend adjustments — none of these are in the backtester.
- **Model degradation**: A model that worked for 6 months will eventually
  stop working. Drift detection helps spot this; it does not prevent it.

---

## 6. If the system loses money

The first time, it is not necessarily a bug. Markets can take from any agent
that has positive expected return.

The conditions that should trigger investigation, not just acceptance:

- A drawdown deeper than the worst observed in walk-forward, on
  comparable-length data.
- A loss of statistical confidence: rolling 30-trade Sharpe goes negative.
- Any divergence between backtest and live performance that is not explained
  by costs.
- A kill-switch triggered for any reason other than your finger.

When investigating, start at the bottom of the stack:

1. Are live data and training data the same? Print the feature vectors from
   live and from a backtest of the same period — diff them.
2. Are costs as configured? Pull a recent trade and compare commission and
   spread to broker reports.
3. Is the agent doing what it did in backtest? Plot live actions against the
   actions a fresh backtest would have produced on the same bars.
4. Is the regime different? Run drift detection on the live feature
   distribution against the training reference.

Only after these are ruled out should you suspect the model itself.
