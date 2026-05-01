"""Broker adapters.

`BaseBroker` defines the contract; everything in `live/runtime.py` uses only that.
`MT5Broker` is the real one — it is import-guarded so non-Windows hosts can still
load this module for tests. `PaperBroker` is a deterministic simulator used to
validate the runtime loop end-to-end without sending real orders.

The MT5 implementation uses positions (not orders) and `position_modify` for
SL/TP. We tag every order with `magic_number` so parallel manual trades on the
same account aren't disturbed.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------------
@dataclass
class Position:
    ticket: int
    symbol: str
    volume: float                       # in lots, signed
    open_price: float
    sl: Optional[float] = None
    tp: Optional[float] = None


class BaseBroker(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def equity(self) -> float: ...
    @abstractmethod
    def positions(self) -> List[Position]: ...
    @abstractmethod
    def latest_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame: ...
    @abstractmethod
    def place_market(self, symbol: str, volume: float, sl: Optional[float] = None,
                     tp: Optional[float] = None, comment: str = "") -> Position: ...
    @abstractmethod
    def close(self, ticket: int) -> None: ...
    @abstractmethod
    def modify(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> None: ...


# -----------------------------------------------------------------------------
# MT5
# -----------------------------------------------------------------------------
@dataclass
class MT5Broker(BaseBroker):
    symbol: str
    magic_number: int = 240501
    login_env: str = "MT5_LOGIN"
    password_env: str = "MT5_PASSWORD"
    server_env: str = "MT5_SERVER"
    path_env: str = "MT5_TERMINAL_PATH"

    _TF_MAP = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440,
    }

    def __post_init__(self) -> None:
        self._mt5 = None
        self._connected = False

    def connect(self) -> None:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MetaTrader5 package not available. Install on a Windows host."
            ) from exc
        self._mt5 = mt5

        kwargs = {}
        path = os.environ.get(self.path_env)
        if path:
            kwargs["path"] = path
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")

        login = os.environ.get(self.login_env)
        password = os.environ.get(self.password_env)
        server = os.environ.get(self.server_env)
        if login and password and server:
            ok = mt5.login(int(login), password=password, server=server)
            if not ok:
                raise RuntimeError(f"mt5.login failed: {mt5.last_error()}")
            logger.info("MT5 logged in to %s account=%s", server, login)
        else:
            logger.info("MT5 connected via existing session (no env credentials)")

        # Make sure symbol is selected and visible in Market Watch
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"symbol_select({self.symbol}) failed: {mt5.last_error()}")
        self._connected = True

    def disconnect(self) -> None:
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False

    def equity(self) -> float:
        info = self._mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info failed: {self._mt5.last_error()}")
        return float(info.equity)

    def positions(self) -> List[Position]:
        all_pos = self._mt5.positions_get(symbol=self.symbol) or []
        out: List[Position] = []
        for p in all_pos:
            if p.magic != self.magic_number:
                continue          # ignore positions opened outside this system
            sign = 1 if p.type == self._mt5.POSITION_TYPE_BUY else -1
            out.append(
                Position(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    volume=sign * p.volume,
                    open_price=p.price_open,
                    sl=p.sl if p.sl > 0 else None,
                    tp=p.tp if p.tp > 0 else None,
                )
            )
        return out

    def latest_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        tf_const = getattr(self._mt5, f"TIMEFRAME_{timeframe}")
        rates = self._mt5.copy_rates_from_pos(symbol, tf_const, 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no bars for {symbol} {timeframe}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df.set_index("time")[["open", "high", "low", "close", "volume"]]

    def place_market(self, symbol: str, volume: float, sl: Optional[float] = None,
                     tp: Optional[float] = None, comment: str = "") -> Position:
        mt5 = self._mt5
        if abs(volume) < 1e-9:
            raise ValueError("volume must be non-zero")
        order_type = mt5.ORDER_TYPE_BUY if volume > 0 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if volume > 0 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(abs(volume)),
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(
                f"order_send failed: retcode={getattr(result, 'retcode', None)}, "
                f"comment={getattr(result, 'comment', None)}, last_error={mt5.last_error()}"
            )
        sign = 1 if volume > 0 else -1
        return Position(
            ticket=result.order,
            symbol=symbol,
            volume=sign * float(abs(volume)),
            open_price=result.price,
            sl=sl, tp=tp,
        )

    def close(self, ticket: int) -> None:
        mt5 = self._mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return
        p = positions[0]
        order_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"close failed: {mt5.last_error()}")

    def modify(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> None:
        mt5 = self._mt5
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return
        p = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": p.symbol,
            "sl": float(sl) if sl is not None else p.sl,
            "tp": float(tp) if tp is not None else p.tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"modify failed: {mt5.last_error()}")


# -----------------------------------------------------------------------------
# Paper broker (simulator)
# -----------------------------------------------------------------------------
@dataclass
class PaperBroker(BaseBroker):
    """Deterministic simulator. Used to validate the runtime loop without real money."""
    symbol: str
    initial_equity: float = 100_000.0
    spread_pips: float = 0.8
    point_value: float = 1e-4
    contract_size: float = 100_000.0

    _equity: float = field(init=False)
    _positions: List[Position] = field(default_factory=list)
    _next_ticket: int = field(default=1, init=False)
    _bars: Optional[pd.DataFrame] = field(default=None, init=False)
    _bar_idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._equity = self.initial_equity

    def connect(self) -> None:
        logger.info("PaperBroker connected (initial equity %.2f)", self.initial_equity)

    def disconnect(self) -> None:
        pass

    def feed_bars(self, df: pd.DataFrame) -> None:
        """Provide the bar data this paper broker will replay."""
        self._bars = df.copy()
        self._bar_idx = 0

    def equity(self) -> float:
        return self._equity

    def positions(self) -> List[Position]:
        return list(self._positions)

    def latest_bars(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:  # noqa: ARG002
        if self._bars is None:
            raise RuntimeError("feed_bars() not called")
        end = min(self._bar_idx + 1, len(self._bars))
        start = max(0, end - n)
        return self._bars.iloc[start:end]

    def place_market(self, symbol: str, volume: float, sl: Optional[float] = None,
                     tp: Optional[float] = None, comment: str = "") -> Position:  # noqa: ARG002
        if self._bars is None:
            raise RuntimeError("feed_bars() not called")
        price = float(self._bars.iloc[self._bar_idx]["close"])
        spread_cost = abs(volume) * self.spread_pips * self.point_value * self.contract_size
        self._equity -= spread_cost
        ticket = self._next_ticket
        self._next_ticket += 1
        pos = Position(ticket=ticket, symbol=symbol, volume=volume,
                       open_price=price, sl=sl, tp=tp)
        self._positions.append(pos)
        return pos

    def close(self, ticket: int) -> None:
        if self._bars is None:
            return
        price = float(self._bars.iloc[self._bar_idx]["close"])
        kept: List[Position] = []
        for p in self._positions:
            if p.ticket == ticket:
                pnl = p.volume * (price - p.open_price) * self.contract_size
                self._equity += pnl
            else:
                kept.append(p)
        self._positions = kept

    def modify(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> None:
        for p in self._positions:
            if p.ticket == ticket:
                if sl is not None:
                    p.sl = sl
                if tp is not None:
                    p.tp = tp

    def advance_bar(self) -> None:
        """Move forward one bar and mark-to-market open positions."""
        if self._bars is None or self._bar_idx >= len(self._bars) - 1:
            return
        prev_close = float(self._bars.iloc[self._bar_idx]["close"])
        self._bar_idx += 1
        new_close = float(self._bars.iloc[self._bar_idx]["close"])
        for p in self._positions:
            self._equity += p.volume * (new_close - prev_close) * self.contract_size
