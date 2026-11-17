"""Order handling for the Quarterly Theory strategy: sizing, shortability, bracket orders.

Orders are Alpaca bracket orders: market entry + take-profit limit + stop-loss
stop, in whole shares (Alpaca does not accept fractional quantities on bracket
orders), with `time_in_force` from config (default GTC so both legs survive the
close). client_order_id = "<prefix>-<SYMBOL>-<YYYYMMDD>-<n>" identifies this
strategy's orders (qt930 for triad930, qt3 for q3) and lets a restart count
today's trades.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.signals import SHORT, Decision

log = get_logger(__name__)

ORDER_PREFIX = "qt3"
PREFIXES = {"q3": ORDER_PREFIX, "triad930": "qt930", "nyopen": "qtny", "silverbullet": "qtsb"}


def position_size(equity: float, risk_pct: float, risk_per_share: float) -> int:
    """(equity x risk_pct / 100) / per-share risk, rounded down to whole shares; 0 if risk <= 0."""
    if risk_per_share is None or not math.isfinite(risk_per_share) or risk_per_share <= 0 or equity <= 0:
        return 0
    return int(math.floor(equity * risk_pct / 100.0 / risk_per_share))


def capped_size(qty: int, price: float, max_notional: float) -> int:
    """Whole shares, at most max_notional worth at `price` (the account's buying power / leverage cap)."""
    if price is None or price <= 0 or max_notional <= 0:
        return 0
    return min(qty, int(math.floor(max_notional / price)))


BP_HEADROOM = 0.95          # leave room for slippage between the signal close and the fill


def refit_qty(qty: int, error: str, headroom: float = 0.98) -> int:
    """Smaller size after an "insufficient buying power" rejection, from the broker's own figures.

    Alpaca checks market shorts against a lower buying power than the account shows and prices
    them with a ~3% buffer, so the first try can be refused; its error carries both numbers
    (buying_power, cost_basis). Returns 0 when the error is anything else.
    """
    import json

    try:
        body = json.loads(error[error.index("{"):])
        bp, cost = float(body["buying_power"]), float(body["cost_basis"])
    except (ValueError, KeyError, TypeError):
        return 0
    if "buying power" not in str(body.get("message", "")) or bp <= 0 or cost <= 0 or qty <= 0:
        return 0
    return min(qty - 1, int(math.floor(qty * bp / cost * headroom)))


def order_id(symbol: str, day: date, n: int, prefix: str = ORDER_PREFIX) -> str:
    return f"{prefix}-{symbol}-{day:%Y%m%d}-{n}"


def bracket_prices(side: str, stop: float, target: float) -> "tuple[float, float]":
    """Round to cents: the stop away from the entry (never tighter than the day's extreme), target to nearest."""
    stop_px = math.ceil(stop * 100 - 1e-9) / 100 if side == SHORT else math.floor(stop * 100 + 1e-9) / 100
    return round(stop_px, 2), round(target, 2)


@dataclass
class Outcome:
    symbol: str
    status: str                    # submitted | dry_run | skipped | rejected
    reason: str = ""
    qty: int = 0
    order_id: Optional[str] = None
    broker_id: Optional[str] = None

    def counts_as_trade(self) -> bool:
        return self.status in ("submitted", "dry_run")


class Executor:
    def __init__(self, cfg: QuarterlyConfig, client):
        self.cfg = cfg
        self.client = client                 # broker.alpaca_client.AlpacaClient
        self._assets: Dict[str, Dict[str, Any]] = {}
        self.prefix = PREFIXES[cfg.strategy]

    # -------------------------------------------------------------- state
    def equity(self) -> float:
        return float(self.client.get_account()["equity"])

    def positions(self) -> Dict[str, float]:
        return {p["symbol"]: p["qty"] for p in self.client.get_positions() if abs(p["qty"]) > 1e-9}

    def trades_today(self, day: date, since: pd.Timestamp) -> Dict[str, int]:
        """This strategy's entries for `day`, from the broker's order history (restart-safe)."""
        counts: Dict[str, int] = {}
        for o in self.client.get_order_history(limit=500, after=since.to_pydatetime(), status="all"):
            cid = getattr(o, "client_order_id", None) or ""
            parts = cid.split("-")
            if len(parts) == 4 and parts[0] == self.prefix and parts[2] == f"{day:%Y%m%d}":
                counts[parts[1]] = counts.get(parts[1], 0) + 1
        return counts

    def protective_stops(self, since: pd.Timestamp) -> Dict[str, float]:
        """symbol -> stop price of this strategy's latest bracket order since `since` whose stop leg has not
        filled (from the broker's order history, so it survives a restart)."""
        stops: Dict[str, "tuple[Any, float]"] = {}
        for o in self.client.get_order_history(limit=500, after=since.to_pydatetime(), status="all"):
            cid = getattr(o, "client_order_id", None) or ""
            if not cid.startswith(self.prefix + "-"):
                continue
            for leg in list(getattr(o, "legs", None) or []):
                sp = getattr(leg, "stop_price", None)
                status = str(getattr(leg.status, "value", leg.status)).lower()
                if sp is not None and status != "filled":
                    at = getattr(o, "submitted_at", None)
                    if o.symbol not in stops or (at is not None and stops[o.symbol][0] is not None
                                                 and at > stops[o.symbol][0]):
                        stops[o.symbol] = (at, float(sp))
        return {s: v[1] for s, v in stops.items()}

    def asset(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._assets:
            self._assets[symbol] = self.client.get_asset(symbol)
        return self._assets[symbol]

    def new_day(self) -> None:
        self._assets.clear()                 # shortability can change day to day

    # ------------------------------------------------------------- orders
    def execute(self, dec: Decision, day: date, n: int) -> Outcome:
        """Turn an entry decision into a bracket order (or a logged dry-run order)."""
        sym = dec.symbol
        a = self.asset(sym)
        if not a["tradable"]:
            return self._skip(dec, "asset not tradable")
        if dec.action == SHORT and not (a["shortable"] and a["easy_to_borrow"]):
            return self._skip(dec, f"cannot short {sym}: shortable={a['shortable']} easy_to_borrow={a['easy_to_borrow']}")
        acct = self.client.get_account()
        equity = float(acct["equity"])
        risk_qty = position_size(equity, self.cfg.risk_pct, dec.risk_per_share)
        max_notional = BP_HEADROOM * min(float(acct.get("buying_power") or 0.0), self.cfg.max_leverage * equity)
        qty = capped_size(risk_qty, dec.entry, max_notional)
        if qty <= 0:
            return self._skip(dec, f"size 0 shares (equity ${equity:,.0f}, risk {self.cfg.risk_pct}% , "
                                   f"${dec.risk_per_share:.4f}/share, max position ${max_notional:,.0f})")
        if qty < risk_qty:
            log.warning("position capped by buying power", extra={
                "symbol": sym, "risk_qty": risk_qty, "qty": qty, "max_notional": round(max_notional, 2),
                "risk_usd": round(qty * dec.risk_per_share, 2)})
        stop, target = bracket_prices(dec.action, dec.stop, dec.target)
        cid = order_id(sym, day, n, self.prefix)
        order = {"symbol": sym, "side": "sell" if dec.action == SHORT else "buy", "qty": qty, "type": "market",
                 "order_class": "bracket", "time_in_force": self.cfg.time_in_force, "stop_loss": stop,
                 "take_profit": target, "client_order_id": cid, "signal_close": dec.entry,
                 "risk_usd": round(qty * dec.risk_per_share, 2), "equity": round(equity, 2)}
        if self.cfg.dry_run:
            log.warning("DRY RUN: would submit bracket order", extra=order)
            return Outcome(sym, "dry_run", "dry run", qty, cid)
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        req = MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.SELL if dec.action == SHORT else OrderSide.BUY,
            time_in_force=TimeInForce(self.cfg.time_in_force), order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=target), stop_loss=StopLossRequest(stop_price=stop),
            client_order_id=cid)
        try:
            placed = self.client.submit(req)
        except Exception as exc:
            smaller = refit_qty(qty, str(exc))
            if smaller <= 0:
                log.error("bracket order rejected", extra={**order, "error": str(exc)})
                return Outcome(sym, "rejected", str(exc), qty, cid)
            # Rejected for buying power: retry once at the size the broker says fits
            log.warning("bracket order refused for buying power, retrying smaller",
                        extra={"symbol": sym, "qty": qty, "retry_qty": smaller, "error": str(exc)})
            qty = smaller
            order.update(qty=qty, risk_usd=round(qty * dec.risk_per_share, 2))
            req.qty = qty
            try:
                placed = self.client.submit(req)
            except Exception as exc2:
                log.error("bracket order rejected", extra={**order, "error": str(exc2)})
                return Outcome(sym, "rejected", str(exc2), qty, cid)
        log.warning("bracket order submitted", extra={**order, "broker_id": str(placed.id)})
        return Outcome(sym, "submitted", "", qty, cid, str(placed.id))

    def _skip(self, dec: Decision, reason: str) -> Outcome:
        log.warning("entry skipped", extra={"symbol": dec.symbol, "action": dec.action, "reason": reason})
        return Outcome(dec.symbol, "skipped", reason)

    def flatten(self, symbols: List[str], reason: str = "strategy flatten time") -> List[str]:
        """Close this strategy's open positions: cancel their bracket legs first (they reserve the shares)."""
        held = self.positions()
        closed = []
        for sym in symbols:
            if sym not in held:
                continue
            if self.cfg.dry_run:
                log.warning("DRY RUN: would close position", extra={"symbol": sym, "qty": held[sym]})
                continue
            for o in self.client.open_orders(sym):
                for leg in [o] + list(getattr(o, "legs", None) or []):
                    status = str(getattr(leg.status, "value", leg.status)).lower()
                    if status not in ("filled", "canceled", "cancelled", "expired", "rejected"):
                        self.client.cancel(str(leg.id))
            self.client.close_position(sym)
            closed.append(sym)
            log.warning("position closed", extra={"symbol": sym, "qty": held[sym], "reason": reason})
        return closed
