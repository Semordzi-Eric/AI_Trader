"""Tests for the RiskManager. Halt logic must be correct — bugs here are
the kind that lose real money."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from ai_trader.risk.manager import RiskConfig, RiskManager


@pytest.fixture
def cfg(tmp_path):
    return RiskConfig(
        max_daily_loss_pct=2.0,
        max_drawdown_pct=10.0,
        max_consecutive_losses=3,
        circuit_breaker_vol_mult=5.0,
        kill_switch_path=str(tmp_path / "KILL"),
    )


class TestRiskHalt:
    def test_no_halt_when_healthy(self, cfg):
        rm = RiskManager(cfg)
        ts = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        assert rm.should_halt(equity=100_000, peak_equity=100_000, ts=ts) is False
        assert rm.halted_reason is None

    def test_kill_switch_file_halts_immediately(self, cfg):
        Path(cfg.kill_switch_path).write_text("stop")
        rm = RiskManager(cfg)
        ts = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        assert rm.should_halt(equity=100_000, peak_equity=100_000, ts=ts) is True
        assert "manual_kill_switch" in rm.halted_reason

    def test_max_drawdown_halts(self, cfg):
        rm = RiskManager(cfg)
        ts = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        # Peak 100k, equity 89k → -11% drawdown, exceeds 10% limit.
        assert rm.should_halt(equity=89_000, peak_equity=100_000, ts=ts) is True
        assert "max_drawdown" in rm.halted_reason

    def test_drawdown_at_limit_does_not_halt(self, cfg):
        rm = RiskManager(cfg)
        ts = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        # Exactly -10% — below the strict-less-than threshold.
        assert rm.should_halt(equity=90_000, peak_equity=100_000, ts=ts) is False

    def test_daily_loss_halts(self, cfg):
        rm = RiskManager(cfg)
        ts0 = pd.Timestamp("2024-01-01 09:00", tz="UTC")
        # First call sets day_start_equity = 100k.
        rm.should_halt(equity=100_000, peak_equity=100_000, ts=ts0)
        ts1 = pd.Timestamp("2024-01-01 14:00", tz="UTC")
        # 97_000 = -3% from day_start_equity → exceeds 2% daily limit.
        assert rm.should_halt(equity=97_000, peak_equity=100_000, ts=ts1) is True
        assert "daily_loss" in rm.halted_reason

    def test_daily_reset_at_new_day(self, cfg):
        rm = RiskManager(cfg)
        # Day 1 starts at 100k, ends at 99k (no halt).
        rm.should_halt(equity=100_000, peak_equity=100_000,
                       ts=pd.Timestamp("2024-01-01 09:00", tz="UTC"))
        rm.should_halt(equity=99_000, peak_equity=100_000,
                       ts=pd.Timestamp("2024-01-01 23:00", tz="UTC"))
        # Day 2 starts: day_start_equity should reset to 99k.
        rm.should_halt(equity=99_000, peak_equity=100_000,
                       ts=pd.Timestamp("2024-01-02 00:00", tz="UTC"))
        # Now -2.1% from 99k = ~96920. Halt at 97k? That's -2.02% from 99k, which trips.
        ts1 = pd.Timestamp("2024-01-02 12:00", tz="UTC")
        assert rm.should_halt(equity=96_900, peak_equity=100_000, ts=ts1) is True

    def test_consecutive_losses_halts(self, cfg):
        rm = RiskManager(cfg)
        # Establish day so per-day reset doesn't zero the counter.
        rm.should_halt(equity=100_000, peak_equity=100_000,
                       ts=pd.Timestamp("2024-01-01 09:00", tz="UTC"))
        for _ in range(cfg.max_consecutive_losses):
            rm.record_trade_result(-50.0)
        ts = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        assert rm.should_halt(equity=100_000, peak_equity=100_000, ts=ts) is True
        assert "consecutive_losses" in rm.halted_reason

    def test_winning_trade_resets_consecutive_losses(self, cfg):
        rm = RiskManager(cfg)
        rm.record_trade_result(-50)
        rm.record_trade_result(-50)
        rm.record_trade_result(+10)
        assert rm.consecutive_losses == 0


class TestClampPosition:
    def test_clamps_above_cap(self, cfg):
        rm = RiskManager(cfg)
        assert rm.clamp_position(2.5, equity=100_000) == cfg.max_position_units

    def test_clamps_below_negative_cap(self, cfg):
        rm = RiskManager(cfg)
        assert rm.clamp_position(-2.5, equity=100_000) == -cfg.max_position_units

    def test_passes_through_in_range(self, cfg):
        rm = RiskManager(cfg)
        assert rm.clamp_position(0.5, equity=100_000) == pytest.approx(0.5)
