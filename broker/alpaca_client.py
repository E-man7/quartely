"""alpaca-py TradingClient wrapper: paper by default, health check, retries with reconnect.

Credentials come from the environment (`.env`), never from code or settings.
Live trading (`broker.paper_trading: false`) requires typing a confirmation
phrase at an interactive terminal; without one the client refuses to start.
"""
from __future__ import annotations

import random
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from monitoring.logger import get_logger

log = get_logger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
LIVE_CONFIRMATION = "YES I UNDERSTAND THE RISKS"
LIVE_PROMPT = f"⚠️ LIVE TRADING MODE. Type '{LIVE_CONFIRMATION}' to confirm: "


def _enum_value(x: Any) -> str:
    return str(getattr(x, "value", x)).lower()


def _status_code(exc: Exception) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if code is None and getattr(exc, "response", None) is not None:
        code = getattr(exc.response, "status_code", None)
    return code


def _is_connection_error(exc: Exception) -> bool:
    try:
        import requests

        if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, (ConnectionError, TimeoutError))


def confirm_live_trading(input_fn: Callable[[str], str] = input, interactive: Optional[bool] = None) -> None:
    """Raise unless the operator types the confirmation phrase at a terminal."""
    interactive = sys.stdin.isatty() if interactive is None else interactive
    if not interactive:
        raise RuntimeError("Live trading needs interactive confirmation; no terminal attached. "
                           "Set broker.paper_trading: true or run from a terminal.")
    if input_fn(LIVE_PROMPT).strip() != LIVE_CONFIRMATION:
        raise RuntimeError("Live trading not confirmed")
    log.warning("LIVE TRADING CONFIRMED BY OPERATOR")


def is_paper(settings: Dict) -> bool:
    cfg = settings.get("broker", {})
    return bool(cfg.get("paper_trading", cfg.get("paper", True)))


class AlpacaClient:
    def __init__(
        self,
        settings: Dict,
        credentials: Dict,
        confirm: Callable[[], None] = confirm_live_trading,
        client_factory: Optional[Callable[..., Any]] = None,
    ):
        creds = credentials.get("alpaca", {})
        if not creds.get("api_key") or not creds.get("secret_key"):
            raise RuntimeError("Alpaca credentials missing: set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        cfg = settings.get("broker", {})
        self.paper = is_paper(settings)
        if not self.paper:
            confirm()
        self.base_url = PAPER_URL if self.paper else LIVE_URL
        self._key, self._secret = creds["api_key"], creds["secret_key"]
        self.max_retries = cfg.get("max_retries", 5)
        self.backoff_base = cfg.get("backoff_base_seconds", 1.0)
        self.backoff_max = cfg.get("backoff_max_seconds", 30.0)
        self._factory = client_factory or self._default_factory
        self.client = self._connect()

    # ------------------------------------------------------------ connection
    def _default_factory(self):
        from alpaca.trading.client import TradingClient

        return TradingClient(self._key, self._secret, paper=self.paper, url_override=self.base_url)

    def _connect(self):
        return self._factory()

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter: base * 2^attempt, capped."""
        return min(self.backoff_max, self.backoff_base * 2 ** attempt) * (0.5 + random.random() / 2)

    def _call(self, fn: Callable[[Any], Any], attempts: Optional[int] = None, what: str = "alpaca call") -> Any:
        """Run fn(client) with retries. 4xx errors (except 429) are final; connection errors reconnect."""
        attempts = attempts or self.max_retries
        for i in range(attempts):
            try:
                return fn(self.client)
            except Exception as exc:
                code = _status_code(exc)
                if (code is not None and 400 <= code < 500 and code != 429) or i == attempts - 1:
                    raise
                wait = self.backoff(i)
                log.warning("alpaca call failed, retrying",
                            extra={"call": what, "error": str(exc), "status": code, "attempt": i + 1, "wait_s": round(wait, 2)})
                time.sleep(wait)
                if _is_connection_error(exc):
                    self.client = self._connect()

    def health_check(self) -> Dict[str, Any]:
        """Verify credentials, account status and clock on startup. Raises on failure."""
        account = self.get_account()
        clock = self.get_clock()
        if account["status"] != "active":
            raise RuntimeError(f"Alpaca account not active: {account['status']}")
        if account["trading_blocked"] or account["account_blocked"]:
            raise RuntimeError("Alpaca account is blocked from trading")
        info = {"paper": self.paper, "url": self.base_url, "equity": account["equity"],
                "buying_power": account["buying_power"], "market_open": clock["is_open"]}
        log.info("alpaca health check ok", extra=info)
        return info

    def trading_stream(self):
        from alpaca.trading.stream import TradingStream

        return TradingStream(self._key, self._secret, paper=self.paper)

    # --------------------------------------------------------------- account
    def get_account(self) -> Dict[str, Any]:
        a = self._call(lambda c: c.get_account(), what="get_account")
        f = lambda name: float(getattr(a, name, None) or 0.0)  # noqa: E731
        return {
            "status": _enum_value(a.status),
            "equity": f("equity"),
            "last_equity": f("last_equity"),
            "cash": f("cash"),
            "buying_power": f("buying_power"),
            "regt_buying_power": f("regt_buying_power"),
            "daytrading_buying_power": f("daytrading_buying_power"),
            "multiplier": f("multiplier"),
            "initial_margin": f("initial_margin"),
            "maintenance_margin": f("maintenance_margin"),
            "trading_blocked": bool(a.trading_blocked),
            "account_blocked": bool(getattr(a, "account_blocked", False)),
            "pattern_day_trader": bool(a.pattern_day_trader),
        }

    def get_available_margin(self) -> Dict[str, float]:
        """Margin headroom. Alpaca: 2x overnight (Reg T, $2k+), 4x intraday (PDT, $25k+)."""
        a = self.get_account()
        return {
            "multiplier": a["multiplier"],
            "buying_power": a["buying_power"],
            "regt_buying_power": a["regt_buying_power"],
            "daytrading_buying_power": a["daytrading_buying_power"],
            "excess_margin": a["equity"] - a["initial_margin"],
            "maintenance_cushion": a["equity"] - a["maintenance_margin"],
        }

    def get_clock(self) -> Dict[str, Any]:
        c = self._call(lambda cl: cl.get_clock(), what="get_clock")
        return {"is_open": bool(c.is_open), "next_open": c.next_open, "next_close": c.next_close, "timestamp": c.timestamp}

    def is_market_open(self) -> bool:
        return self.get_clock()["is_open"]

    def get_calendar(self, start, end) -> List[Any]:
        from alpaca.trading.requests import GetCalendarRequest

        return self._call(lambda c: c.get_calendar(GetCalendarRequest(start=start, end=end)), what="get_calendar")

    def is_tradable(self, symbol: str) -> bool:
        asset = self._call(lambda c: c.get_asset(symbol), what="get_asset")
        return bool(asset.tradable)

    # ------------------------------------------------------------- positions
    def get_positions(self) -> List[Dict[str, Any]]:
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price or 0.0),
                "market_value": float(p.market_value or 0.0),
                "unrealized_pl": float(p.unrealized_pl or 0.0),
                "unrealized_plpc": float(p.unrealized_plpc or 0.0),
            }
            for p in self._call(lambda c: c.get_all_positions(), what="get_positions")
        ]

    def close_position(self, symbol: str):
        return self._call(lambda c: c.close_position(symbol), attempts=1, what="close_position")

    def close_all_positions(self, cancel_orders: bool = True):
        return self._call(lambda c: c.close_all_positions(cancel_orders=cancel_orders), attempts=1,
                          what="close_all_positions")

    # ---------------------------------------------------------------- orders
    def submit(self, request) -> Any:
        # Never retried: a timeout after the broker accepted it would double the order.
        # A retry with a fresh client_order_id is the caller's decision.
        return self._call(lambda c: c.submit_order(request), attempts=1, what="submit_order")

    def get_order_history(self, limit: int = 100, after=None, until=None, symbols: Optional[List[str]] = None,
                          status: str = "all") -> List[Any]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus(status), limit=limit, after=after, until=until,
                               symbols=symbols, nested=True)
        return self._call(lambda c: c.get_orders(filter=req), what="get_orders")

    def open_orders(self, symbol: Optional[str] = None) -> List[Any]:
        return self.get_order_history(limit=500, symbols=[symbol] if symbol else None, status="open")

    def get_order(self, order_id: str) -> Any:
        return self._call(lambda c: c.get_order_by_id(order_id), what="get_order")

    def cancel(self, order_id: str) -> None:
        self._call(lambda c: c.cancel_order_by_id(order_id), what="cancel_order")

    def cancel_all(self) -> None:
        self._call(lambda c: c.cancel_orders(), what="cancel_orders")

    def replace(self, order_id: str, qty: Optional[float] = None, limit_price: Optional[float] = None,
                stop_price: Optional[float] = None) -> Any:
        from alpaca.trading.requests import ReplaceOrderRequest

        req = ReplaceOrderRequest(qty=qty, limit_price=limit_price, stop_price=stop_price)
        return self._call(lambda c: c.replace_order_by_id(order_id, req), attempts=1, what="replace_order")


# ---------------------------------------------------------------- streams
class StreamSupervisor:
    """Run an alpaca-py websocket stream in a daemon thread; restart it with backoff if it dies."""

    def __init__(self, name: str, make_stream: Callable[[], Any], backoff_base: float = 1.0,
                 backoff_max: float = 60.0):
        self.name = name
        self.make_stream = make_stream
        self.backoff_base, self.backoff_max = backoff_base, backoff_max
        self.stream = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "StreamSupervisor":
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-stream", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            started = time.time()
            try:
                self.stream = self.make_stream()
                log.info("stream connecting", extra={"stream": self.name})
                self.stream.run()                  # blocks; alpaca-py also reconnects internally
            except Exception as exc:
                log.warning("stream error", extra={"stream": self.name, "error": str(exc)})
            if self._stop.is_set():
                break
            attempt = 0 if time.time() - started > 300 else attempt + 1   # healthy for 5 min: reset backoff
            wait = min(self.backoff_max, self.backoff_base * 2 ** attempt)
            log.warning("stream disconnected, reconnecting", extra={"stream": self.name, "wait_s": wait})
            self._stop.wait(wait)

    def stop(self) -> None:
        self._stop.set()
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:  # pragma: no cover - best effort on shutdown
                pass
