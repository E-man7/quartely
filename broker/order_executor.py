"""Order placement for the allocation strategy, with protective stops.

* `submit_order`: LIMIT at +/- `limit_offset_pct` of the current price; if not
  filled within `fill_timeout_seconds` it is cancelled and (optionally) the
  remainder is sent at market.
* `submit_bracket_order`: entry + stop (+ take-profit if the signal has one) as
  one Alpaca bracket (OTO when there is no take-profit).
* `modify_stop`: stops only ever tighten.
* Every rebalance gets a `trade_id`; the journal links signal -> risk decision
  -> orders -> fills (client_order_id = "<trade_id>-<n>").

Alpaca reserves shares covered by an open stop-sell, so a rebalance sell first
cancels that symbol's protective stop and re-places it for what remains.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from broker.alpaca_client import _enum_value
from broker.position_tracker import PositionTracker, TradeContext
from monitoring.logger import get_logger

log = get_logger(__name__)

TERMINAL = ("filled", "canceled", "cancelled", "expired", "rejected", "done_for_day")


def new_trade_id() -> str:
    return f"qt-{uuid.uuid4().hex[:16]}"


class TradeJournal:
    """Append-only JSONL log of every stage of a trade, keyed by trade_id."""

    def __init__(self, path: Optional[Path]):
        self.path = path

    def record(self, trade_id: str, stage: str, /, **data: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(timezone.utc).isoformat(), **data, "trade_id": trade_id, "stage": stage}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def read(self, trade_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return [r for r in rows if trade_id is None or r["trade_id"] == trade_id]


@dataclass
class PlannedOrder:
    symbol: str
    side: str          # buy | sell
    qty: float
    price: float

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class OrderResult:
    trade_id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float = 0.0
    avg_price: Optional[float] = None
    order_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OrderExecutor:
    def __init__(
        self,
        settings: Dict,
        client,
        tracker: PositionTracker,
        price_fn: Callable[[str], float],
        on_reject: Optional[Callable[[str, Dict], None]] = None,
        risk=None,
        quote_fn: Optional[Callable[[str], Optional[Tuple[float, float]]]] = None,
        journal: Optional[TradeJournal] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        cfg = settings.get("broker", {})
        self.client = client
        self.tracker = tracker
        self.price_fn = price_fn
        self.on_reject = on_reject
        self.risk = risk              # RiskManager: validates every order before submission
        self.quote_fn = quote_fn      # symbol -> (bid, ask), for the spread check
        self.journal = journal or TradeJournal(None)
        self.sleep, self.clock = sleep, clock
        self.order_type = cfg.get("order_type", "limit")
        self.limit_offset = cfg.get("limit_offset_pct", 0.001)
        self.fill_timeout = cfg.get("fill_timeout_seconds", 30)
        self.retry_at_market = cfg.get("retry_at_market", True)
        self.poll = cfg.get("poll_seconds_fill", 1.0)
        self.tif = cfg.get("time_in_force", "day")
        self.min_notional = cfg.get("min_order_notional", 10.0)
        self.fractional = cfg.get("fractional", False)
        self.protective_stops = cfg.get("protective_stops", True)
        self.dry_run = cfg.get("dry_run", False)

    # ------------------------------------------------------------- helpers
    def _limit_price(self, side: str, price: float) -> float:
        return round(price * (1 + self.limit_offset if side == "buy" else 1 - self.limit_offset), 2)

    def _tif(self, gtc: bool = False):
        from alpaca.trading.enums import TimeInForce

        if self.fractional:
            return TimeInForce.DAY            # Alpaca: fractional orders must be DAY
        return TimeInForce.GTC if gtc else TimeInForce(self.tif)

    @staticmethod
    def _side(side: str):
        from alpaca.trading.enums import OrderSide

        return OrderSide.BUY if side == "buy" else OrderSide.SELL

    def _reject(self, info: Dict) -> None:
        log.error("order rejected", extra=info)
        if self.on_reject:
            self.on_reject("order_rejected", info)

    def _validate(self, symbol: str, side: str, qty: float, price: float) -> Optional[str]:
        if self.risk is None:
            return None
        state = self.tracker.portfolio_state()
        quote = self.quote_fn(symbol) if self.quote_fn else None
        check = self.risk.validate_order(symbol, side, qty * price, state, quote,
                                         tradable=self.client.is_tradable(symbol))
        return None if check.approved else check.reason

    def wait_for_fill(self, order_id: str, timeout: Optional[float] = None) -> Any:
        """Poll until the order is terminal or `timeout` passes; returns the last order snapshot."""
        deadline = self.clock() + (self.fill_timeout if timeout is None else timeout)
        order = self.client.get_order(order_id)
        while _enum_value(order.status) not in TERMINAL and self.clock() < deadline:
            self.sleep(self.poll)
            order = self.client.get_order(order_id)
        return order

    @staticmethod
    def _filled(order: Any) -> Tuple[float, Optional[float]]:
        qty = float(getattr(order, "filled_qty", 0) or 0)
        px = getattr(order, "filled_avg_price", None)
        return qty, float(px) if px else None

    # ------------------------------------------------------------ planning
    def plan(self, target_weights: Dict[str, float]) -> List[PlannedOrder]:
        equity = self.tracker.equity
        symbols = set(target_weights) | set(self.tracker.positions)
        orders: List[PlannedOrder] = []
        for sym in symbols:
            price = self.price_fn(sym)
            target_qty = target_weights.get(sym, 0.0) * equity / price
            delta = target_qty - self.tracker.qty(sym)
            if not self.fractional:
                delta = float(math.trunc(delta))
            if abs(delta * price) < self.min_notional:
                continue
            orders.append(PlannedOrder(sym, "buy" if delta > 0 else "sell", round(abs(delta), 6), price))
        # Sells first so their proceeds fund the buys
        return sorted(orders, key=lambda o: o.side != "sell")

    # -------------------------------------------------------------- orders
    def submit_order(self, signal=None, *, symbol: Optional[str] = None, side: Optional[str] = None,
                     qty: Optional[float] = None, trade_id: Optional[str] = None) -> OrderResult:
        """Limit order at +/- limit_offset; cancel after the timeout, then optionally finish at market.

        With a signal, symbol/side default to the signal and qty to the change
        from the current position to the signal's target weight.
        """
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        trade_id = trade_id or new_trade_id()
        symbol = symbol or signal.symbol
        price = self.price_fn(symbol)
        if qty is None:
            target = signal.target_weight * self.tracker.equity / price
            delta = target - self.tracker.qty(symbol)
            qty, side = abs(delta), side or ("buy" if delta > 0 else "sell")
        side = side or "buy"
        if not self.fractional:
            qty = float(math.trunc(qty))
        result = OrderResult(trade_id, symbol, side, qty, "planned")
        if qty <= 0:
            result.status = "nothing_to_do"
            return result
        reason = self._validate(symbol, side, qty, price)
        if reason:
            result.status, result.error = "rejected_by_risk", reason
            self._reject(result.as_dict())
            return result
        self.tracker.expect(TradeContext(trade_id, symbol, signal.regime_name if signal else None,
                                         signal.stop_loss if signal else None))
        limit = self._limit_price(side, price) if self.order_type == "limit" else None
        self.journal.record(trade_id, "order", symbol=symbol, side=side, qty=qty, type=self.order_type,
                            limit_price=limit, reference_price=price)
        if self.dry_run:
            log.info("dry-run order", extra={"trade_id": trade_id, "symbol": symbol, "side": side, "qty": qty,
                                             "limit_price": limit})
            result.status = "dry_run"
            return result

        n = 1
        try:
            if limit is not None:
                req = LimitOrderRequest(symbol=symbol, qty=qty, side=self._side(side), time_in_force=self._tif(),
                                        limit_price=limit, client_order_id=f"{trade_id}-{n}")
            else:
                req = MarketOrderRequest(symbol=symbol, qty=qty, side=self._side(side), time_in_force=self._tif(),
                                         client_order_id=f"{trade_id}-{n}")
            order = self.client.submit(req)
            result.order_ids.append(str(order.id))
            order = self.wait_for_fill(str(order.id))
            filled, avg = self._filled(order)
            status = _enum_value(order.status)
            if status != "filled" and limit is not None:
                self.client.cancel(str(order.id))
                order = self.wait_for_fill(str(order.id), timeout=5)
                filled, avg = self._filled(order)
                remaining = qty - filled
                log.warning("limit order not filled in time", extra={"trade_id": trade_id, "symbol": symbol,
                                                                     "filled": filled, "remaining": remaining})
                if remaining > 0 and self.retry_at_market:
                    n += 1
                    mkt = self.client.submit(MarketOrderRequest(
                        symbol=symbol, qty=remaining, side=self._side(side), time_in_force=self._tif(),
                        client_order_id=f"{trade_id}-{n}"))
                    result.order_ids.append(str(mkt.id))
                    mkt = self.wait_for_fill(str(mkt.id))
                    m_qty, m_px = self._filled(mkt)
                    if m_qty:
                        avg = ((avg or 0) * filled + (m_px or 0) * m_qty) / (filled + m_qty)
                        filled += m_qty
                    status = _enum_value(mkt.status)
                else:
                    status = "partially_filled" if filled else "canceled"
            result.status, result.filled_qty, result.avg_price = status, filled, avg
        except Exception as exc:
            result.status, result.error = "rejected", str(exc)
            self._reject(result.as_dict())
        self.journal.record(trade_id, "order_status", **result.as_dict())
        return result

    def submit_bracket_order(self, signal, qty: Optional[float] = None, trade_id: Optional[str] = None) -> OrderResult:
        """Entry + stop-loss (+ take-profit) in one order. Whole shares, GTC so the stop persists."""
        from alpaca.trading.enums import OrderClass
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest

        trade_id = trade_id or new_trade_id()
        price = self.price_fn(signal.symbol)
        if qty is None:
            qty = signal.target_weight * self.tracker.equity / price - self.tracker.qty(signal.symbol)
        qty = float(math.trunc(qty))
        result = OrderResult(trade_id, signal.symbol, "buy", qty, "planned")
        if signal.stop_loss is None or not math.isfinite(signal.stop_loss) or signal.stop_loss >= price:
            result.status, result.error = "rejected", "bracket needs a stop below the entry price"
            self._reject(result.as_dict())
            return result
        if qty <= 0:
            result.status = "nothing_to_do"
            return result
        reason = self._validate(signal.symbol, "buy", qty, price)
        if reason:
            result.status, result.error = "rejected_by_risk", reason
            self._reject(result.as_dict())
            return result
        limit = self._limit_price("buy", price)
        stop = round(signal.stop_loss, 2)
        tp = round(signal.take_profit, 2) if signal.take_profit else None
        self.tracker.expect(TradeContext(trade_id, signal.symbol, signal.regime_name, stop))
        self.journal.record(trade_id, "order", symbol=signal.symbol, side="buy", qty=qty, type="bracket",
                            limit_price=limit, stop_loss=stop, take_profit=tp)
        if self.dry_run:
            log.info("dry-run bracket", extra={"trade_id": trade_id, "symbol": signal.symbol, "qty": qty,
                                               "limit_price": limit, "stop": stop, "take_profit": tp})
            result.status = "dry_run"
            return result
        try:
            req = LimitOrderRequest(
                symbol=signal.symbol, qty=qty, side=self._side("buy"), time_in_force=self._tif(gtc=True),
                limit_price=limit, client_order_id=f"{trade_id}-1",
                order_class=OrderClass.BRACKET if tp else OrderClass.OTO,
                stop_loss=StopLossRequest(stop_price=stop),
                take_profit=TakeProfitRequest(limit_price=tp) if tp else None,
            )
            order = self.client.submit(req)
            result.order_ids.append(str(order.id))
            order = self.wait_for_fill(str(order.id))
            filled, avg = self._filled(order)
            status = _enum_value(order.status)
            if status != "filled":
                # Cancelling the parent cancels its legs; protect what did fill separately
                self.client.cancel(str(order.id))
                order = self.wait_for_fill(str(order.id), timeout=5)
                filled, avg = self._filled(order)
                status = "partially_filled" if filled else "canceled"
                if filled:
                    self.ensure_stop(signal.symbol, stop, trade_id)
            result.status, result.filled_qty, result.avg_price = status, filled, avg
            self.tracker.set_stop(signal.symbol, stop)
        except Exception as exc:
            result.status, result.error = "rejected", str(exc)
            self._reject(result.as_dict())
        self.journal.record(trade_id, "order_status", **result.as_dict())
        return result

    # --------------------------------------------------------------- stops
    def _stop_orders(self, symbol: str) -> List[Any]:
        """Open stop-sells for `symbol`, standalone or as legs of a bracket/OTO parent."""
        out: Dict[str, Any] = {}
        for o in self.client.open_orders(symbol):
            for leg in [o] + list(getattr(o, "legs", None) or []):
                if (leg.symbol == symbol and _enum_value(leg.side) == "sell" and getattr(leg, "stop_price", None)
                        and _enum_value(leg.status) not in TERMINAL):
                    out[str(leg.id)] = leg
        return list(out.values())

    def modify_stop(self, symbol: str, new_stop: float) -> bool:
        """Raise the protective stop. Never widens: a lower stop than the current one is refused."""
        new_stop = round(new_stop, 2)
        stops = self._stop_orders(symbol)
        if not stops:
            return self.ensure_stop(symbol, new_stop)
        current = max(float(o.stop_price) for o in stops)
        if new_stop <= current:
            log.info("stop not widened", extra={"symbol": symbol, "current": current, "requested": new_stop})
            return False
        if self.dry_run:
            log.info("dry-run modify stop", extra={"symbol": symbol, "from": current, "to": new_stop})
            return True
        for o in stops:
            self.client.replace(str(o.id), stop_price=new_stop)
        self.tracker.set_stop(symbol, new_stop)
        log.info("stop tightened", extra={"symbol": symbol, "from": current, "to": new_stop})
        return True

    def ensure_stop(self, symbol: str, stop: float, trade_id: Optional[str] = None) -> bool:
        """One GTC stop-sell covering the whole-share position, at the tighter of the existing and requested stop."""
        from alpaca.trading.requests import StopOrderRequest

        self.tracker.refresh()
        qty = float(math.trunc(self.tracker.qty(symbol)))
        stops = self._stop_orders(symbol)
        stop = round(max([stop] + [float(o.stop_price) for o in stops]), 2)   # only tighten
        covered = sum(float(o.qty) for o in stops)
        if qty <= 0:
            for o in stops:
                self.client.cancel(str(o.id))
            return False
        if self.tracker.qty(symbol) - qty > 1e-9:
            log.warning("fractional shares not covered by stop", extra={"symbol": symbol,
                                                                         "uncovered": self.tracker.qty(symbol) - qty})
        if len(stops) == 1 and abs(covered - qty) < 1e-9 and abs(float(stops[0].stop_price) - stop) < 0.005:
            return True
        if self.dry_run:
            log.info("dry-run stop", extra={"symbol": symbol, "qty": qty, "stop": stop})
            return True
        if len(stops) == 1 and getattr(stops[0], "parent_id", None) is None:
            self.client.replace(str(stops[0].id), qty=qty, stop_price=stop)
        else:
            for o in stops:
                self.client.cancel(str(o.id))
            trade_id = trade_id or new_trade_id()
            self.client.submit(StopOrderRequest(symbol=symbol, qty=qty, side=self._side("sell"),
                                                time_in_force=self._tif(gtc=True), stop_price=stop,
                                                client_order_id=f"{trade_id}-stop"))
        self.tracker.set_stop(symbol, stop)
        log.info("protective stop set", extra={"symbol": symbol, "qty": qty, "stop": stop})
        return True

    # ------------------------------------------------------------- closing
    def cancel_order(self, order_id: str) -> None:
        if self.dry_run:
            log.info("dry-run cancel", extra={"order_id": order_id})
            return
        self.client.cancel(order_id)

    def cancel_open(self, symbol: Optional[str] = None, keep_stops: bool = True) -> int:
        """Cancel stale open orders (protective stops are kept unless keep_stops=False).

        The open-orders query also returns filled bracket parents whose stop leg is
        still live; only orders that are themselves open can be cancelled.
        """
        orders = {}
        for o in self.client.open_orders(symbol):
            for leg in [o] + list(getattr(o, "legs", None) or []):
                is_stop = _enum_value(leg.side) == "sell" and getattr(leg, "stop_price", None)
                if _enum_value(leg.status) not in TERMINAL and not (keep_stops and is_stop):
                    orders[str(leg.id)] = leg
        orders = list(orders.values())
        for o in orders:
            self.cancel_order(str(o.id))
        if orders:
            log.info("cancelled open orders", extra={"count": len(orders)})
        return len(orders)

    def close_position(self, symbol: str) -> None:
        if self.dry_run:
            log.warning("dry-run close position", extra={"symbol": symbol})
            return
        self.cancel_open(symbol, keep_stops=False)
        self.client.close_position(symbol)
        log.warning("position closed", extra={"symbol": symbol})

    def close_all_positions(self) -> None:
        """Emergency exit: cancel every order and liquidate everything."""
        if self.dry_run:
            log.warning("dry-run close all positions")
            return
        self.client.close_all_positions(cancel_orders=True)
        log.warning("all positions closed")

    flatten = close_all_positions

    # ------------------------------------------------------------ rebalance
    def rebalance(self, target_weights: Dict[str, float], signal=None, decision=None,
                  wait_for_fills: bool = True) -> List[Dict]:
        """Trade to `target_weights`; protect the signal's symbol with its (risk-approved) stop."""
        trade_id = new_trade_id()
        if signal is not None:
            self.journal.record(trade_id, "signal", **signal.as_dict())
        if decision is not None:
            self.journal.record(trade_id, "risk_decision", approved=decision.approved,
                                rejection_reason=decision.rejection_reason, modifications=decision.modifications)
        stop_sig = decision.modified_signal if decision is not None and decision.approved else signal
        self.tracker.refresh()
        self.cancel_open()
        results = []
        for order in self.plan(target_weights):
            is_entry = order.side == "buy" and self.tracker.qty(order.symbol) == 0
            if (is_entry and self.protective_stops and stop_sig is not None and order.symbol == stop_sig.symbol
                    and not self.fractional):
                res = self.submit_bracket_order(stop_sig, qty=order.qty, trade_id=trade_id)
            else:
                if order.side == "sell" and not self.dry_run:
                    for o in self._stop_orders(order.symbol):   # free shares the stop reserves
                        self.client.cancel(str(o.id))
                res = self.submit_order(stop_sig if stop_sig is not None and stop_sig.symbol == order.symbol else None,
                                        symbol=order.symbol, side=order.side, qty=order.qty, trade_id=trade_id)
            results.append(res.as_dict())
        if self.protective_stops and stop_sig is not None and not self.dry_run:
            self.ensure_stop(stop_sig.symbol, stop_sig.stop_loss, trade_id)
        return results
