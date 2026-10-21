"""Positions, account state and fills from the broker.

A websocket subscription to trade updates delivers fills as they happen; each
fill updates the tracked position, the trade journal and the circuit breaker
(with the account's fresh equity). Per-position tracking keeps entry time and
price, stop level, holding period and the regime at entry vs now. On startup,
`reconcile()` syncs the tracked book with what Alpaca actually holds.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from broker.alpaca_client import StreamSupervisor, _enum_value
from monitoring.logger import get_logger

log = get_logger(__name__)


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


@dataclass
class TrackedPosition:
    symbol: str
    qty: float
    entry_time: Optional[pd.Timestamp]
    entry_price: float
    current_price: float
    stop_level: Optional[float] = None
    trade_id: Optional[str] = None
    regime_at_entry: Optional[str] = None
    regime_current: Optional[str] = None

    @property
    def unrealized_pl(self) -> float:
        return (self.current_price - self.entry_price) * self.qty

    @property
    def unrealized_pl_pct(self) -> float:
        return self.current_price / self.entry_price - 1 if self.entry_price else 0.0

    @property
    def holding_period(self) -> Optional[pd.Timedelta]:
        return _now() - self.entry_time if self.entry_time is not None else None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = str(self.entry_time) if self.entry_time is not None else None
        d.update(unrealized_pl=round(self.unrealized_pl, 2), unrealized_pl_pct=round(self.unrealized_pl_pct, 5),
                 holding_days=round(self.holding_period / pd.Timedelta(days=1), 2) if self.holding_period else None,
                 regime_changed=bool(self.regime_at_entry and self.regime_current
                                     and self.regime_at_entry != self.regime_current))
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrackedPosition":
        fields = {k: d.get(k) for k in cls.__dataclass_fields__}
        fields["entry_time"] = pd.Timestamp(d["entry_time"]) if d.get("entry_time") else None
        return cls(**fields)


@dataclass
class TradeContext:
    """What the executor knew when it sent an order, keyed by trade_id."""
    trade_id: str
    symbol: str
    regime: Optional[str] = None
    stop_loss: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class PositionTracker:
    def __init__(self, client, state_file: Optional[Path] = None, risk=None, journal=None,
                 timezone_name: str = "America/New_York"):
        from zoneinfo import ZoneInfo

        self.client = client
        self.tz = ZoneInfo(timezone_name)       # breaker day/week boundaries, same clock as the live loop
        self.state_file = state_file
        self.risk = risk                       # RiskManager whose breaker sees every fill
        self.journal = journal                 # TradeJournal for the fill stage
        self.account: Dict[str, Any] = {}
        self.positions: Dict[str, Dict] = {}   # raw broker positions
        self.tracked: Dict[str, TrackedPosition] = {}
        self.contexts: Dict[str, TradeContext] = {}
        self.updated: Optional[datetime] = None
        self.on_fill: List[Callable[[Dict[str, Any]], None]] = []
        self.on_breaker: List[Callable[[list], None]] = []
        self._lock = threading.RLock()
        self._stream: Optional[StreamSupervisor] = None

    # ------------------------------------------------------------- refresh
    def refresh(self) -> None:
        account, positions = self.client.get_account(), self.client.get_positions()
        with self._lock:
            self.account = account
            self.positions = {p["symbol"]: p for p in positions}
            for sym, p in self.positions.items():
                if sym in self.tracked:
                    t = self.tracked[sym]
                    t.current_price, t.qty, t.entry_price = p["current_price"], p["qty"], p["avg_entry_price"]
            self.updated = datetime.now()

    @property
    def equity(self) -> float:
        return self.account.get("equity", 0.0)

    @property
    def day_pnl_pct(self) -> float:
        last = self.account.get("last_equity") or 0.0
        return self.equity / last - 1 if last else 0.0

    def qty(self, symbol: str) -> float:
        return self.positions.get(symbol, {}).get("qty", 0.0)

    def weights(self) -> Dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {}
        return {s: p["market_value"] / eq for s, p in self.positions.items()}

    @property
    def leverage(self) -> float:
        return sum(abs(w) for w in self.weights().values())

    def portfolio_state(self):
        return self.risk.portfolio_state(self.equity, self.weights(), cash=self.account.get("cash"),
                                         buying_power=self.account.get("buying_power"))

    # ------------------------------------------------------------ tracking
    def expect(self, ctx: TradeContext) -> None:
        """Register an order's context so its fills can be attributed (called by the executor)."""
        with self._lock:
            self.contexts[ctx.trade_id] = ctx

    def set_stop(self, symbol: str, stop: Optional[float]) -> None:
        with self._lock:
            if symbol in self.tracked:
                self.tracked[symbol].stop_level = stop
                self._save()

    def set_regime(self, label: str) -> None:
        with self._lock:
            for pos in self.tracked.values():
                pos.regime_current = label
            self._save()

    def handle_fill(self, symbol: str, side: str, qty: float, price: float, position_qty: float,
                    timestamp: Optional[pd.Timestamp] = None, client_order_id: Optional[str] = None,
                    event: str = "fill") -> Dict[str, Any]:
        """Apply one (partial) fill: position, journal, account refresh and circuit breaker."""
        ts = pd.Timestamp(timestamp) if timestamp is not None else _now()
        trade_id = client_order_id.rsplit("-", 1)[0] if client_order_id else None
        with self._lock:
            ctx = self.contexts.get(trade_id) if trade_id else None
            pos = self.tracked.get(symbol)
            if position_qty <= 1e-9:
                if pos is not None:
                    log.info("position closed", extra={"symbol": symbol, "exit_price": price,
                                                       "realized_pl_est": round((price - pos.entry_price) * qty, 2),
                                                       "held_days": round(pos.holding_period / pd.Timedelta(days=1), 2)
                                                       if pos.holding_period else None})
                self.tracked.pop(symbol, None)
            elif pos is None:
                self.tracked[symbol] = TrackedPosition(
                    symbol, position_qty, ts, price, price, ctx.stop_loss if ctx else None, trade_id,
                    ctx.regime if ctx else None, ctx.regime if ctx else None)
            else:
                pos.qty, pos.current_price = position_qty, price     # entry price: broker average, on refresh
            self._save()
        fill = {"trade_id": trade_id, "symbol": symbol, "side": side, "qty": qty, "price": price,
                "position_qty": position_qty, "timestamp": str(ts), "event": event}
        if self.journal is not None and trade_id:
            self.journal.record(trade_id, "fill", **fill)
        log.info("fill", extra=fill)

        self.refresh()
        if self.risk is not None:
            held = [s for s, p in self.positions.items() if abs(p["qty"]) > 1e-9]
            events = self.risk.breaker.update(self.equity, pd.Timestamp(datetime.now(self.tz)).tz_localize(None), held)
            if events:
                for cb in self.on_breaker:
                    cb(events)
        for cb in self.on_fill:
            cb(fill)
        return fill

    def handle_trade_update(self, update: Any) -> Optional[Dict[str, Any]]:
        """alpaca-py TradeUpdate -> handle_fill for fill / partial_fill events."""
        event = _enum_value(update.event)
        if event not in ("fill", "partial_fill"):
            return None
        order = update.order
        return self.handle_fill(
            symbol=order.symbol, side=_enum_value(order.side), qty=float(update.qty or 0.0),
            price=float(update.price or 0.0), position_qty=float(update.position_qty or 0.0),
            timestamp=update.timestamp, client_order_id=order.client_order_id, event=event)

    # -------------------------------------------------------------- stream
    def start_stream(self) -> StreamSupervisor:
        def make():
            stream = self.client.trading_stream()

            async def on_update(data):
                try:
                    self.handle_trade_update(data)
                except Exception:
                    log.exception("trade update handler failed")

            stream.subscribe_trade_updates(on_update)
            return stream

        self._stream = StreamSupervisor("trade_updates", make).start()
        return self._stream

    def stop_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    # ----------------------------------------------------------- reconcile
    def reconcile(self) -> Dict[str, List[str]]:
        """Sync tracked positions with the broker: adopt untracked, drop vanished, fix quantities."""
        self._load()
        self.refresh()
        report: Dict[str, List[str]] = {"adopted": [], "dropped": [], "qty_fixed": [], "ok": []}
        stops = self._open_stops()
        with self._lock:
            for sym in list(self.tracked):
                if sym not in self.positions:
                    log.warning("tracked position not held at broker: dropping", extra={"symbol": sym})
                    self.tracked.pop(sym)
                    report["dropped"].append(sym)
            for sym, p in self.positions.items():
                pos = self.tracked.get(sym)
                if pos is None:
                    self.tracked[sym] = TrackedPosition(sym, p["qty"], self._entry_time(sym), p["avg_entry_price"],
                                                        p["current_price"], stops.get(sym))
                    log.warning("untracked broker position: adopting", extra={"symbol": sym, "qty": p["qty"]})
                    report["adopted"].append(sym)
                    continue
                if abs(pos.qty - p["qty"]) > 1e-6:
                    log.warning("position quantity mismatch: using broker",
                                extra={"symbol": sym, "tracked": pos.qty, "broker": p["qty"]})
                    report["qty_fixed"].append(sym)
                else:
                    report["ok"].append(sym)
                pos.qty, pos.entry_price, pos.current_price = p["qty"], p["avg_entry_price"], p["current_price"]
                pos.stop_level = stops.get(sym, pos.stop_level)
            self._save()
        log.info("positions reconciled", extra=report)
        return report

    def _open_stops(self) -> Dict[str, float]:
        stops = {}
        for o in self.client.open_orders():
            legs = [o] + list(getattr(o, "legs", None) or [])
            for leg in legs:
                if _enum_value(leg.side) == "sell" and getattr(leg, "stop_price", None):
                    stops[leg.symbol] = max(stops.get(leg.symbol, 0.0), float(leg.stop_price))
        return stops

    def _entry_time(self, symbol: str) -> Optional[pd.Timestamp]:
        """Most recent filled buy for `symbol` (approximate entry time for adopted positions)."""
        try:
            for o in self.client.get_order_history(limit=50, symbols=[symbol], status="closed"):
                if _enum_value(o.side) == "buy" and getattr(o, "filled_at", None):
                    return pd.Timestamp(o.filled_at)
        except Exception as exc:
            log.warning("could not read order history", extra={"symbol": symbol, "error": str(exc)})
        return None

    # --------------------------------------------------------- persistence
    def _save(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({s: p.as_dict() for s, p in self.tracked.items()}, indent=2))

    def _load(self) -> None:
        if self.state_file is None or not self.state_file.exists():
            return
        data = json.loads(self.state_file.read_text())
        with self._lock:
            self.tracked = {s: TrackedPosition.from_dict(d) for s, d in data.items()}

    def snapshot(self) -> Dict:
        weights = self.weights()
        positions: List[Dict] = []
        for s, p in self.positions.items():
            row = dict(p, weight=weights.get(s, 0.0))
            if s in self.tracked:
                t = self.tracked[s].as_dict()
                row.update(stop_level=t["stop_level"], holding_days=t["holding_days"],
                           regime_at_entry=t["regime_at_entry"], regime_current=t["regime_current"])
            positions.append(row)
        return {
            "equity": self.equity,
            "cash": self.account.get("cash"),
            "day_pnl_pct": self.day_pnl_pct,
            "leverage": self.leverage,
            "positions": positions,
            "updated": f"{self.updated:%Y-%m-%d %H:%M:%S}" if self.updated else None,
        }
