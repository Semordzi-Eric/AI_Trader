"""AI Trader dashboard.

Run with:
    streamlit run ai_trader/dashboard/app.py

Pages:
    1. Backtest — run a backtest from the UI, view equity / drawdown / trades.
    2. Compare — load saved backtest results and compare side-by-side.
    3. Live monitor — show latest live state, recent trades, kill-switch.
    4. Strategy controls — toggle config, kill-switch, retrain triggers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make the package importable when running `streamlit run path/to/app.py`
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trader.backtest.engine import EventDrivenBacktester
from ai_trader.backtest.policies import BaselinePolicy, RandomPolicy, RLPolicy
from ai_trader.data.sources import CSVSource, SyntheticSource
from ai_trader.rl.agent import PPOAgent
from ai_trader.rl.env import EnvConfig, TradingEnv
from ai_trader.training.pipeline import build_features_pipeline, make_observation_matrix
from ai_trader.utils.config import load_config
from ai_trader.utils.metrics import drawdown_series

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Trader",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — restrained editorial / terminal aesthetic
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', 'Courier New', monospace; letter-spacing: -0.02em; }
    .metric-card {
        background: #1a1f2e; border: 1px solid #2a3040; padding: 1rem;
        border-radius: 4px; margin-bottom: 0.5rem;
    }
    .stMetric { background: #1a1f2e; padding: 0.75rem; border-radius: 4px; }
    .small-muted { color: #8b95a8; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("◈ AI Trader")
page = st.sidebar.radio(
    "Page",
    ["Backtest", "Compare", "Live monitor", "Controls"],
    label_visibility="collapsed",
)

config_path = st.sidebar.text_input("Config", value="configs/default.yaml")
try:
    cfg = load_config(ROOT / config_path)
    st.sidebar.success(f"Loaded {config_path}")
except Exception as exc:
    st.sidebar.error(f"Config error: {exc}")
    cfg = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_data(source: str, csv_path: str, n_bars: int, seed: int) -> pd.DataFrame:
    if source == "synthetic":
        return SyntheticSource(n_bars=n_bars, seed=seed).load()
    return CSVSource(path=csv_path).load()


def _equity_chart(equity: pd.Series, dd: np.ndarray, title: str = "") -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=("Equity", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, mode="lines",
                   line=dict(color="#4ade80", width=2), name="Equity"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=equity.index, y=dd * 100, mode="lines",
                   line=dict(color="#f87171", width=1.5),
                   fill="tozeroy", fillcolor="rgba(248,113,113,0.2)",
                   name="Drawdown %"),
        row=2, col=1,
    )
    fig.update_layout(
        height=540, template="plotly_dark", showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20), title=title,
        font=dict(family="IBM Plex Mono, monospace"),
    )
    return fig


def _metrics_grid(summary: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Final equity", f"${summary.get('final_equity', 0):,.0f}",
                   delta=f"{summary.get('total_return_pct', 0):.2f}%")
    cols[1].metric("Sharpe", f"{summary.get('sharpe', 0):.2f}")
    cols[2].metric("Sortino", f"{summary.get('sortino', 0):.2f}")
    cols[3].metric("Max DD", f"{summary.get('max_drawdown_pct', 0):.2f}%")

    cols = st.columns(4)
    cols[0].metric("Calmar", f"{summary.get('calmar', 0):.2f}")
    cols[1].metric("Profit factor", f"{summary.get('profit_factor', 0):.2f}")
    cols[2].metric("Win rate", f"{summary.get('win_rate_pct', 0):.1f}%")
    cols[3].metric("Trades", f"{summary.get('n_trades', 0)}")


# -----------------------------------------------------------------------------
# Backtest page
# -----------------------------------------------------------------------------
def page_backtest() -> None:
    st.title("Backtest")
    st.caption("Run a backtest with the selected policy. Results are cached for the session.")

    if cfg is None:
        st.error("Fix the config path in the sidebar before continuing.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        source = st.selectbox("Data source", ["synthetic", "csv"], index=0)
    with c2:
        n_bars = st.number_input("Bars (synthetic only)", value=20_000, min_value=1_000, step=1_000)
    with c3:
        seed = st.number_input("Seed", value=42, step=1)

    csv_path = st.text_input("CSV path", value=cfg.get_path("data.csv_path", ""))
    policy_kind = st.selectbox("Policy", ["baseline (flat)", "random", "trained_rl"])

    rl_path = st.text_input(
        "RL model path",
        value=str(ROOT / "artifacts/models/best_agent.zip"),
        disabled=(policy_kind != "trained_rl"),
    )

    if not st.button("Run backtest", type="primary"):
        return

    with st.spinner("Loading data..."):
        df = _load_data(source, csv_path, int(n_bars), int(seed))
    st.success(f"Loaded {len(df):,} bars")

    with st.spinner("Building features..."):
        out = build_features_pipeline(
            df,
            return_horizons=cfg["features"]["return_horizons"],
            vol_windows=cfg["features"]["vol_windows"],
            window_size=cfg["features"]["window_size"],
            normalize_window=cfg["features"].get("normalize_window", 500),
            use_regime=cfg["features"].get("regime", {}).get("enabled", True),
            n_regimes=cfg["features"].get("regime", {}).get("n_states", 3),
        )

    obs_matrix = make_observation_matrix(out.feature_array, out.window_size)

    if policy_kind == "baseline (flat)":
        policy = BaselinePolicy()
    elif policy_kind == "random":
        policy = RandomPolicy(seed=int(seed))
    else:
        if not Path(rl_path).exists():
            st.error(f"Model not found: {rl_path}. Train one first.")
            return

        def _stub():
            return TradingEnv(out.prices, out.feature_array, EnvConfig(window_size=out.window_size))

        agent = PPOAgent(env_factory=_stub).load(rl_path)
        policy = RLPolicy(agent)

    with st.spinner("Running event-driven backtest..."):
        bt = EventDrivenBacktester(
            initial_equity=cfg["backtest"]["initial_equity"],
            contract_size=cfg["backtest"]["contract_size"],
            spread_pips=cfg["backtest"]["spread_pips"],
            commission_per_lot=cfg["backtest"]["commission_per_lot"],
            slippage_pips=cfg["backtest"]["slippage_pips"],
            point_value=cfg["backtest"]["point_value"],
            latency_bars=cfg["backtest"]["latency_bars"],
            max_position=cfg["rl"]["action"]["max_position"],
        )
        result = bt.run(out.prices, obs_matrix, policy, out.window_size)

    summary = result.summary()
    st.subheader("Performance summary")
    _metrics_grid(summary)
    st.plotly_chart(
        _equity_chart(result.equity, drawdown_series(result.equity.values)),
        width="stretch",
    )

    if result.trades:
        with st.expander(f"Trade ledger ({len(result.trades)} trades)"):
            trades_df = pd.DataFrame(
                [
                    {
                        "open": t.open_time,
                        "close": t.close_time,
                        "side": "LONG" if t.direction > 0 else "SHORT",
                        "open_px": t.open_price,
                        "close_px": t.close_price,
                        "units": t.units,
                        "pnl": t.pnl,
                    }
                    for t in result.trades
                ]
            )
            st.dataframe(trades_df, width="stretch", height=300)

    # Save run for the comparison page
    runs_dir = ROOT / "artifacts/reports/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    name = st.text_input("Save as", value=f"{policy_kind}_{source}")
    if st.button("Save this run"):
        path = runs_dir / f"{name}.json"
        path.write_text(json.dumps({
            "summary": summary,
            "equity": {ts.isoformat(): float(v) for ts, v in result.equity.items()},
        }))
        st.success(f"Saved → {path.name}")


# -----------------------------------------------------------------------------
# Compare page
# -----------------------------------------------------------------------------
def page_compare() -> None:
    st.title("Compare runs")
    st.caption("Load multiple saved backtests and compare metrics + equity.")
    runs_dir = ROOT / "artifacts/reports/runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        st.info("No saved runs yet. Run and save a backtest first.")
        return
    chosen = st.multiselect("Runs", [f.stem for f in files], default=[f.stem for f in files[:3]])
    if not chosen:
        return

    summaries = []
    fig = go.Figure()
    for name in chosen:
        data = json.loads((runs_dir / f"{name}.json").read_text())
        s = data.get("summary", {})
        s["name"] = name
        summaries.append(s)

        if "equity" in data:
            eq_df = pd.DataFrame(list(data["equity"].items()), columns=["ts", "equity"])
            eq_df["ts"] = pd.to_datetime(eq_df["ts"])
        elif "equity_csv" in data:
            csv_p = ROOT / data["equity_csv"]
            if csv_p.exists():
                # Load from CSV (CLI format)
                eq_df = pd.read_csv(csv_p, index_col=0)
                eq_df.index.name = "ts"
                eq_df = eq_df.reset_index()
                eq_df.columns = ["ts", "equity"]
                eq_df["ts"] = pd.to_datetime(eq_df["ts"])
            else:
                st.warning(f"Equity CSV not found for {name}: {csv_p}")
                continue
        else:
            st.warning(f"No equity data found for {name}")
            continue

        fig.add_trace(go.Scatter(x=eq_df["ts"], y=eq_df["equity"], mode="lines", name=name))

    fig.update_layout(template="plotly_dark", height=480, margin=dict(l=20, r=20, t=30, b=20),
                      font=dict(family="IBM Plex Mono, monospace"))
    st.plotly_chart(fig, width="stretch")

    df_sum = pd.DataFrame(summaries).set_index("name")
    cols_show = [c for c in [
        "final_equity", "total_return_pct", "sharpe", "sortino", "calmar",
        "max_drawdown_pct", "profit_factor", "win_rate_pct", "n_trades",
    ] if c in df_sum.columns]
    st.dataframe(df_sum[cols_show].style.format({
        "final_equity": "{:,.0f}", "total_return_pct": "{:.2f}",
        "sharpe": "{:.2f}", "sortino": "{:.2f}", "calmar": "{:.2f}",
        "max_drawdown_pct": "{:.2f}", "profit_factor": "{:.2f}",
        "win_rate_pct": "{:.2f}",
    }), width="stretch")


# -----------------------------------------------------------------------------
# Live monitor page
# -----------------------------------------------------------------------------
def page_live() -> None:
    st.title("Live monitor")
    state_path = ROOT / "artifacts/live_state.json"
    log_path = ROOT / "artifacts/logs/live.log"

    cols = st.columns(3)
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            cols[0].metric("Last bar", state.get("last_bar_ts", "—"))
        except Exception as exc:
            cols[0].error(f"state error: {exc}")
    else:
        cols[0].info("No live state yet")

    kill = ROOT / cfg.get_path("risk.kill_switch_path", "artifacts/KILL") if cfg else None
    if kill and kill.exists():
        cols[1].error("KILL-SWITCH ACTIVE")
    else:
        cols[1].success("Kill-switch inactive")

    cols[2].info("Live process runs separately (`scripts/run_live.py`)")

    st.subheader("Recent log")
    if log_path.exists():
        text = log_path.read_text().splitlines()
        st.code("\n".join(text[-200:]), language="text")
    else:
        st.info("No live log yet")


# -----------------------------------------------------------------------------
# Controls page
# -----------------------------------------------------------------------------
def page_controls() -> None:
    st.title("Strategy controls")

    st.subheader("Kill-switch")
    st.caption("Engaging the kill-switch flattens all positions and halts trading on the next tick.")
    kill_path = ROOT / cfg.get_path("risk.kill_switch_path", "artifacts/KILL") if cfg else None
    if kill_path is None:
        st.error("Cannot resolve kill-switch path — fix config.")
        return

    c1, c2 = st.columns(2)
    if not kill_path.exists():
        if c1.button("Engage kill-switch", type="primary"):
            kill_path.parent.mkdir(parents=True, exist_ok=True)
            kill_path.write_text("HALTED")
            st.success(f"Kill-switch engaged ({kill_path})")
    else:
        if c2.button("Release kill-switch"):
            kill_path.unlink()
            st.success("Kill-switch released")

    st.subheader("Config preview")
    if cfg is not None:
        st.json(dict(cfg))


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if page == "Backtest":
    page_backtest()
elif page == "Compare":
    page_compare()
elif page == "Live monitor":
    page_live()
else:
    page_controls()
