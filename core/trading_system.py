"""Live trading system: startup, bar loop and shutdown.

Startup: connect + verify the account, check market hours, recover the
previous session's snapshot, initialise the risk manager from the live
portfolio, sync positions, load or train the HMM (retrain if older than
`execution.max_model_age_days`), start websocket feeds, "System online".

Every `bar_timeframe` bar (default 5Min, built from streamed minute bars):
    feed/API health -> circuit breakers on live equity -> per-regime trailing
    stop -> signal pipeline when a `signal_timeframe` bar completes ->
    status file for the dashboard.
Signal pipeline (default once per day, on completed daily bars, as backtested):
    features -> filtered HMM -> stability/flicker -> strategy -> risk veto ->
    orders. A data-feed outage pauses signals; stops stay in place.

Shutdown (SIGINT/SIGTERM): close websockets, keep positions (stops protect
them), save the snapshot, print a session summary.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config import PROJECT_ROOT
from monitoring import alerts as A
from monitoring.dashboard import write_status
from monitoring.logger import get_logger, set_context

log = get_logger(__name__)


def timeframe_minutes(tf: str) -> int:
    """'5Min' -> 5, '1Hour' -> 60, '1Day' -> 390 (one regular session)."""
    digits = "".join(c for c in tf if c.isdigit()) or "1"
    unit = tf[len(digits):].lower() if tf.startswith(digits) else tf.lower()
    per = {"min": 1, "hour": 60, "day": 390, "week": 1950}
    if unit not in per:
        raise ValueError(f"Unsupported timeframe {tf!r}")
    return int(digits) * per[unit]


class BarAggregator:
    """Roll streamed 1-minute bars into N-minute bars aligned to the clock (09:30, 09:35, ...).

    A bucket is emitted when its last minute arrives, or when a later minute
    shows the bucket is over (minutes without trades produce no bar).
    """

    def __init__(self, minutes: int, emit: Callable[[Dict[str, Any]], None]):
        self.minutes = minutes
        self.emit = emit
        self._cur: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_minute_bar(self, bar: Dict[str, Any]) -> None:
        ts = pd.Timestamp(bar["timestamp"])
        start = ts.floor(f"{self.minutes}min")
        sym = bar["symbol"]
        out = []
        with self._lock:
            cur = self._cur.get(sym)
            if cur is not None and cur["start"] != start:
                out.append(self._close(sym))
                cur = None
            if cur is None:
                cur = self._cur[sym] = {"symbol": sym, "start": start, "open": bar["open"], "high": bar["high"],
                                        "low": bar["low"], "close": bar["close"], "volume": bar["volume"]}
            else:
                cur["high"] = max(cur["high"], bar["high"])
                cur["low"] = min(cur["low"], bar["low"])
                cur["close"] = bar["close"]
                cur["volume"] += bar["volume"]
            if ts + pd.Timedelta(minutes=1) >= start + pd.Timedelta(minutes=self.minutes):
                out.append(self._close(sym))
        for b in out:
            self.emit(b)

    def _close(self, sym: str) -> Dict[str, Any]:
        b = self._cur.pop(sym)
        b["timestamp"] = b["start"] + pd.Timedelta(minutes=self.minutes)     # bar close time
        return b


class TradingSystem:
    def __init__(self, settings: Dict, credentials: Dict, dry_run: bool = False, client=None, market_data=None,
                 alerts=None, sleep: Callable[[float], None] = time.sleep):
        from core.signal_generator import SignalGenerator

        self.settings = settings
        self.creds = credentials
        if dry_run:
            settings["broker"]["dry_run"] = True
        self.dry_run = settings["broker"].get("dry_run", False)
        ex = settings.get("execution", {})
        self.ex = ex
        self.tz = ZoneInfo(ex.get("timezone", "America/New_York"))
        self.bar_tf = ex.get("bar_timeframe", "5Min")
        self.bar_minutes = timeframe_minutes(self.bar_tf)
        self.signal_tf = ex.get("signal_timeframe", "1Day")
        self.daily_signals = self.signal_tf.lower() in ("1day", "day")
        self.run_time = tuple(map(int, ex.get("run_time", "15:45").split(":")))
        self.max_model_age = ex.get("max_model_age_days", 7)
        self.feed_timeout = ex.get("feed_timeout_bars", 3) * self.bar_minutes * 60
        self.large_pnl = settings.get("monitoring", {}).get("alerts", {}).get("large_pnl_pct", 0.02)
        log_dir = PROJECT_ROOT / settings.get("monitoring", {}).get("log_dir", "logs")
        self.snapshot_path = PROJECT_ROOT / ex.get("state_file", "logs/state_snapshot.json")
        self.status_path = PROJECT_ROOT / settings.get("monitoring", {}).get("status_file", "logs/status.json")
        self.log_dir = log_dir
        self.sleep = sleep

        self.sig = SignalGenerator(settings)
        self.signal_symbol = settings["universe"]["signal_symbol"]
        self.symbols = sorted({s for s in (self.sig.risk_symbol, self.sig.safe_symbol) if s})
        self.client = client
        self.md = market_data
        self.alerts = alerts
        self.tracker = self.executor = None

        self.bars: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.streaming = False
        self.feed_down = False
        self.api_ok: Optional[bool] = None
        self.api_latency_ms: Optional[float] = None
        self.last_bar_at: Optional[float] = None
        self.last_decision: Optional[pd.Timestamp] = None
        self.last_signal = None
        self.stop_inputs: Optional[Dict[str, Any]] = None   # strategy + daily EMA/ATR for trailing stops
        self.signals: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.session = {"started": None, "start_equity": None, "bars": 0, "decisions": 0, "rebalances": 0,
                        "breaker_events": 0, "errors": 0}
        self._was_flickering = False
        self._clock_cache = (0.0, None)

    # ----------------------------------------------------------------- time
    def now(self) -> pd.Timestamp:
        """Wall-clock US/Eastern, tz-naive: the clock for breakers, decisions and logs."""
        return pd.Timestamp(datetime.now(self.tz)).tz_localize(None)

    def market_open(self) -> bool:
        cached_at, value = self._clock_cache
        if value is None or time.time() - cached_at > 60:
            value = self.client.is_market_open()
            self._clock_cache = (time.time(), value)
        return value

    # -------------------------------------------------------------- startup
    def startup(self) -> bool:
        """Run the startup sequence. Returns False if the market is closed and `when_closed: exit`."""
        from broker.alpaca_client import AlpacaClient
        from broker.order_executor import OrderExecutor, TradeJournal
        from broker.position_tracker import PositionTracker
        from data.market_data import MarketData
        from monitoring.alerts import AlertManager

        self.client = self.client or AlpacaClient(self.settings, self.creds)
        health = self.client.health_check()
        self.api_ok = True
        self.md = self.md or MarketData(self.settings, self.creds)
        self.alerts = self.alerts or AlertManager(self.settings, self.creds)

        if not health["market_open"]:
            if self.ex.get("when_closed", "wait") == "exit":
                log.warning("market closed: exiting")
                return False
            self.wait_for_open()
            if self.stop_event.is_set():
                return False

        snapshot = self.load_snapshot()
        risk = self.sig.risk
        risk.load_state(snapshot.get("risk", {}))
        journal = TradeJournal(self.log_dir / "trade_journal.jsonl")
        self.tracker = PositionTracker(self.client, self.log_dir / "positions.json", risk=risk, journal=journal,
                                       timezone_name=self.ex.get("timezone", "America/New_York"))
        self.executor = OrderExecutor(self.settings, self.client, self.tracker, self.md.latest_price,
                                      on_reject=self.alerts.send, risk=risk, quote_fn=self.md.get_latest_quote,
                                      journal=journal)
        self.tracker.reconcile()
        # Risk manager starts from the live portfolio (and the recovered peak / breaker state)
        held = [s for s, p in self.tracker.positions.items() if abs(p["qty"]) > 1e-9]
        events = risk.breaker.update(self.tracker.equity, self.now(), held,
                                     day_start=self.tracker.account.get("last_equity"))
        self.handle_breaker_events(events)
        self.tracker.on_breaker.append(self.handle_breaker_events)

        self.init_hmm()
        self.start_feeds()
        self.session.update(started=f"{self.now():%Y-%m-%d %H:%M:%S} ET", start_equity=self.tracker.equity)
        self.print_state()
        log.info("System online", extra={"paper": self.client.paper, "dry_run": self.dry_run,
                                         "bar_timeframe": self.bar_tf, "signal_timeframe": self.signal_tf})
        self.write_status()
        return True

    def wait_for_open(self) -> None:
        clock = self.client.get_clock()
        log.info("market closed: waiting for the open", extra={"next_open": str(clock["next_open"])})
        while not self.stop_event.is_set() and not self.client.is_market_open():
            self.stop_event.wait(min(300.0, max(5.0, (pd.Timestamp(clock["next_open"]) - pd.Timestamp.now(
                tz=pd.Timestamp(clock["next_open"]).tz)).total_seconds())) if clock.get("next_open") else 60.0)

    def signal_history(self) -> "tuple[pd.DataFrame, Optional[pd.DataFrame]]":
        """Completed signal-timeframe bars only (today's daily bar / the current intraday bar excluded)."""
        now = self.now()

        def fetch(symbol: str) -> pd.DataFrame:
            if self.daily_signals:
                df = self.md.history(symbol, use_cache=False)
                return df[df.index < now.normalize()]
            from data.feature_engineering import feature_warmup

            minutes = timeframe_minutes(self.signal_tf)
            needed = self.settings["hmm"].get("min_train_size", 504) + feature_warmup(self.settings.get("features"))
            days = int(needed * minutes / 390 * 1.5) + 7
            df = self.md.get_historical_bars(symbol, self.signal_tf, start=(now - pd.Timedelta(days=days)).strftime("%Y-%m-%d"))
            cutoff = now.floor(f"{minutes}min").tz_localize(self.tz)
            return df[df.index + pd.Timedelta(minutes=minutes) <= cutoff]

        bars = fetch(self.signal_symbol)
        risk_bars = fetch(self.sig.risk_symbol) if self.sig.risk_symbol != self.signal_symbol else None
        return bars, risk_bars

    def init_hmm(self) -> None:
        if self.sig.load_latest_model(self.max_model_age, self.now()):
            return
        bars, _ = self.signal_history()
        feats = self.sig.features(bars).dropna()
        self.sig.ensure_fitted(feats)
        self.alerts.send(A.HMM_RETRAINED, f"HMM trained: {self.sig.hmm.n_states} regimes", self.sig.hmm.metadata())

    def start_feeds(self) -> None:
        self.tracker.start_stream()
        if self.md.source == "alpaca":
            agg = BarAggregator(self.bar_minutes, self.bars.put)
            self.md.subscribe_bars(self.symbols, "1Min", agg.on_minute_bar)
            self.md.subscribe_quotes(self.symbols)
            self.streaming = True
        else:
            log.warning("no websocket bars for data.source != alpaca: polling on the bar clock")
        self.last_bar_at = time.time()

    def print_state(self) -> None:
        from rich.console import Console

        from monitoring.dashboard import render

        Console(stderr=True).print(render(self.status()))

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        while not self.stop_event.is_set():
            bar = self.next_bar()
            if bar is None:
                continue
            try:
                self.on_bar(bar)
            except Exception as exc:
                self.on_unhandled(exc)

    def next_bar(self) -> Optional[Dict[str, Any]]:
        """Block until the next bar of interest (or a periodic wake-up to check feed health)."""
        if self.streaming:
            try:
                bar = self.bars.get(timeout=1.0)
            except queue.Empty:
                self.check_feed()
                return None
            if bar["symbol"] != self.sig.risk_symbol:
                return None
            return bar
        # Polling: wake on the bar clock and build a bar from the latest data
        now = self.now()
        nxt = now.floor(f"{self.bar_minutes}min") + pd.Timedelta(minutes=self.bar_minutes)
        if self.stop_event.wait((nxt - now).total_seconds()):
            return None
        if not self.market_open():
            return None
        b = self.md.get_latest_bar(self.sig.risk_symbol)
        return {**b, "timestamp": nxt}

    def check_feed(self) -> None:
        if self.last_bar_at is None or self.feed_down:
            return
        try:
            is_open = self.market_open()
        except Exception as exc:
            self.mark_api_down(exc)
            return
        if is_open and time.time() - self.last_bar_at > self.feed_timeout:
            self.feed_down = True
            log.error("data feed down: signals paused, stops stay active",
                      extra={"silent_seconds": round(time.time() - self.last_bar_at)})
            self.alerts.send(A.DATA_FEED_DOWN, "No bars received: signals paused, stops remain in place")
            self.write_status()

    def mark_api_down(self, exc: Exception) -> None:
        if self.api_ok is not False:
            self.alerts.send(A.API_LOST, f"Alpaca API unreachable: {exc}")
        self.api_ok = False
        log.error("alpaca api error", extra={"error": str(exc)})

    def on_bar(self, bar: Dict[str, Any]) -> None:
        self.session["bars"] += 1
        self.last_bar_at = time.time()
        if self.feed_down:
            self.feed_down = False
            log.info("data feed restored")
        t0 = time.perf_counter()
        try:
            self.tracker.refresh()
        except Exception as exc:
            self.mark_api_down(exc)                 # skip this bar; broker-side stops still protect
            self.write_status()
            return
        self.api_latency_ms = (time.perf_counter() - t0) * 1000
        if self.api_ok is False:
            log.info("alpaca api restored")
        self.api_ok = True

        now = self.now()
        held = [s for s, p in self.tracker.positions.items() if abs(p["qty"]) > 1e-9]
        events = self.sig.risk.breaker.update(self.tracker.equity, now, held,
                                              day_start=self.tracker.account.get("last_equity"))
        self.handle_breaker_events(events)
        day_pnl = self.tracker.day_pnl_pct
        if abs(day_pnl) >= self.large_pnl:
            self.alerts.send(A.LARGE_PNL, f"Daily P&L {day_pnl:+.2%}", {"equity": self.tracker.equity})

        self.update_trailing_stop(float(bar["close"]))
        if self.decision_due(now):
            self.decide(now)
        self.update_context()
        self.write_status()

    def decision_due(self, now: pd.Timestamp) -> bool:
        if self.daily_signals:
            after_run_time = (now.hour, now.minute) >= self.run_time
            return after_run_time and (self.last_decision is None or self.last_decision.date() != now.date())
        bucket = now.floor(f"{timeframe_minutes(self.signal_tf)}min")
        return self.last_decision is None or bucket > self.last_decision

    # -------------------------------------------------------------- decide
    def decide(self, now: pd.Timestamp) -> None:
        if self.feed_down:
            log.warning("signal skipped: data feed down")
            return
        if self.sig.risk.breaker.check().flatten:
            log.warning("signal skipped: circuit breaker halt", extra={"breaker": str(self.sig.risk.breaker.check())})
            self.last_decision = now if self.daily_signals else now.floor(f"{timeframe_minutes(self.signal_tf)}min")
            return
        bars, risk_bars = self.signal_history()
        signal = self.sig.generate(bars, equity=self.tracker.equity, current_weights=self.tracker.weights(),
                                   risk_bars=risk_bars, day_start_equity=self.tracker.account.get("last_equity"),
                                   as_of=now, max_model_age_days=self.max_model_age)
        self.last_decision = now if self.daily_signals else now.floor(f"{timeframe_minutes(self.signal_tf)}min")
        self.last_signal = signal
        self.session["decisions"] += 1
        if self.sig.refit_happened:
            self.alerts.send(A.HMM_RETRAINED, f"HMM retrained: {self.sig.hmm.n_states} regimes", self.sig.hmm.metadata())
        if self.sig.hmm_error:
            self.alerts.send(A.ERROR, f"HMM error, holding regime {signal.regime.label}: {self.sig.hmm_error}")
        state = signal.regime
        if state.is_flickering and not self._was_flickering:
            self.alerts.send(A.FLICKER_EXCEEDED, f"Regime flicker {state.flicker_rate}/{self.sig.hmm.flicker_window}: "
                                                  "uncertainty mode (size halved, 1.0x)")
        self._was_flickering = state.is_flickering
        for event in signal.events:
            if event == "regime_change":
                self.alerts.send(A.REGIME_CHANGE, f"Regime now {state.label} ({state.probability:.0%})",
                                 {"strategy": signal.trade.strategy_name if signal.trade else None})
        self.tracker.set_regime(state.label)

        trade, decision = signal.trade, signal.risk_decision
        if trade is not None:
            self.stop_inputs = {"strategy": self.sig.strategy.strategy_for(state.state_id),
                                "ema": trade.metadata.get("ema_50"), "atr": trade.metadata.get("atr_14")}
        if decision is None:
            log.warning("no trade signal (insufficient history)")
        elif not decision.approved:
            log.warning("signal rejected by risk", extra={"reason": decision.rejection_reason})
        elif decision.modifications:
            log.info("signal modified by risk", extra={"modifications": decision.modifications})
        else:
            log.info("signal approved", extra={"symbol": trade.symbol, "target_weight": round(trade.target_weight, 4)})

        before = self.tracker.weights().get(self.sig.risk_symbol, 0.0)
        target = signal.target_weights.get(self.sig.risk_symbol, 0.0)
        if signal.rebalance:
            results = self.executor.rebalance(signal.target_weights, trade, decision)
            self.session["rebalances"] += 1
            self.signals.appendleft({
                "time": f"{now:%H:%M}", "symbol": self.sig.risk_symbol, "action": f"Rebalance {before:.0%}→{target:.0%}",
                "reason": (trade.strategy_name.replace("Strategy", "") if trade else "risk") +
                          (f" | {decision.rejection_reason}" if decision is not None and not decision.approved else ""),
                "orders": [f"{r['side']} {r['qty']:g} {r['symbol']} [{r['status']}]" for r in results]})
        elif decision is not None and decision.approved and self.tracker.qty(self.sig.risk_symbol) > 0:
            self.executor.modify_stop(self.sig.risk_symbol, decision.modified_signal.stop_loss)   # tighten only
        self.save_snapshot()

    def update_trailing_stop(self, price: float) -> None:
        """Re-apply the current regime's stop rule at the latest price; stops only ever tighten."""
        inp = self.stop_inputs
        sym = self.sig.risk_symbol
        if not inp or inp["ema"] is None or inp["atr"] is None or self.tracker.qty(sym) <= 0:
            return
        _, _, stop, _ = inp["strategy"].plan(price, inp["ema"], inp["atr"])
        current = (self.tracker.tracked.get(sym).stop_level if sym in self.tracker.tracked else None) or 0.0
        if stop < price and stop > current + 0.01:
            self.executor.modify_stop(sym, stop)

    # -------------------------------------------------------------- breakers
    def handle_breaker_events(self, events: List[Any]) -> None:
        if not events:
            return
        self.session["breaker_events"] += len(events)
        for e in events:
            self.alerts.send(A.CIRCUIT_BREAKER, f"{e.breaker} at {e.drawdown:.2%} (equity ${e.equity:,.0f})",
                             {"breaker": e.breaker, "hmm_regime": e.hmm_regime, "hmm_wrong": e.hmm_wrong},
                             key=f"{A.CIRCUIT_BREAKER}:{e.breaker}")
        if self.sig.risk.breaker.check().flatten and self.tracker.positions:
            self.executor.close_all_positions()
        self.save_snapshot()

    # ------------------------------------------------------- errors & state
    def on_unhandled(self, exc: Exception) -> None:
        self.session["errors"] += 1
        log.error("unhandled error in main loop", extra={"error": str(exc), "traceback": traceback.format_exc()})
        try:
            self.save_snapshot()
        finally:
            self.alerts.send(A.ERROR, f"Unhandled error: {exc}")

    def load_snapshot(self) -> Dict[str, Any]:
        if not self.snapshot_path.exists():
            return {}
        try:
            snap = json.loads(self.snapshot_path.read_text())
        except json.JSONDecodeError as exc:
            log.error("unreadable state snapshot: starting fresh", extra={"error": str(exc)})
            return {}
        log.info("recovered previous session", extra={"saved_at": snap.get("saved_at"),
                                                      "last_decision": snap.get("last_decision")})
        if snap.get("last_decision"):
            self.last_decision = pd.Timestamp(snap["last_decision"])
        return snap

    def save_snapshot(self) -> None:
        snap = {
            "saved_at": str(self.now()),
            "risk": self.sig.risk.state_dict(),
            "last_decision": str(self.last_decision) if self.last_decision is not None else None,
            "last_signal": self.last_signal.as_dict() if self.last_signal is not None else None,
            "positions": {s: p.as_dict() for s, p in self.tracker.tracked.items()} if self.tracker else {},
            "session": self.session,
        }
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.snapshot_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, indent=2, default=str))
        tmp.replace(self.snapshot_path)

    def update_context(self) -> None:
        state = self.last_signal.regime if self.last_signal is not None else None
        set_context(regime=state.label if state else None,
                    probability=round(state.probability, 4) if state else None,
                    equity=round(self.tracker.equity, 2),
                    positions={s: round(p["qty"], 4) for s, p in self.tracker.positions.items()},
                    daily_pnl=round(self.tracker.equity - (self.tracker.account.get("last_equity") or 0), 2))

    def status(self) -> Dict[str, Any]:
        t = self.tracker
        b = self.sig.risk.breaker
        cfg = self.settings.get("risk", {}).get("breakers", {})
        state = self.last_signal.regime if self.last_signal is not None else self.sig.last_state
        equity = t.equity if t else None
        last_eq = (t.account.get("last_equity") if t else None) or None
        weights = t.weights() if t else {}
        positions = []
        for sym, p in (t.positions.items() if t else []):
            tp = t.tracked.get(sym)
            held = tp.holding_period if tp else None
            positions.append({"symbol": sym, "side": "LONG" if p["qty"] > 0 else "SHORT", "price": p["current_price"],
                              "pnl_pct": p.get("unrealized_plpc"), "stop": tp.stop_level if tp else None,
                              "holding": None if held is None else
                              (f"{held / pd.Timedelta(hours=1):.0f}h" if held < pd.Timedelta(days=2)
                               else f"{held / pd.Timedelta(days=1):.0f}d")})
        return {
            "updated": datetime.now().isoformat(timespec="seconds"),
            "regime": {"label": state.label if state else None, "probability": state.probability if state else None,
                       "stability_bars": state.consecutive_bars if state else None,
                       "flicker_rate": state.flicker_rate if state else None,
                       "flicker_window": self.sig.hmm.flicker_window, "confirmed": state.is_confirmed if state else None,
                       "flickering": state.is_flickering if state else None,
                       "strategy": (self.last_signal.trade.strategy_name if self.last_signal is not None
                                    and self.last_signal.trade else None)},
            "portfolio": {"equity": equity, "daily_pnl": equity - last_eq if equity and last_eq else None,
                          "daily_pnl_pct": t.day_pnl_pct if t else None,
                          "allocation": weights.get(self.sig.risk_symbol), "leverage": t.leverage if t else None},
            "positions": positions,
            "signals": list(self.signals),
            "risk": {"daily_dd": max(0.0, -b.daily_dd), "daily_limit": cfg.get("daily_halt_dd", 0.03),
                     "weekly_dd": max(0.0, -b.weekly_dd), "weekly_limit": cfg.get("weekly_halt_dd", 0.07),
                     "peak_dd": max(0.0, -b.peak_dd), "peak_limit": cfg.get("peak_halt_dd", 0.10),
                     "breaker": str(b.check())},
            "system": {"data_ok": not self.feed_down, "api_ok": self.api_ok, "api_latency_ms": self.api_latency_ms,
                       "hmm_age_days": self.sig.model_age_days(self.now()),
                       "mode": "DRY-RUN" if self.dry_run else "PAPER" if (self.client is None or self.client.paper)
                       else "LIVE"},
        }

    def write_status(self) -> None:
        try:
            write_status(self.status_path, self.status())
        except Exception as exc:          # the dashboard must never break trading
            log.warning("status write failed", extra={"error": str(exc)})

    # -------------------------------------------------------------- shutdown
    def shutdown(self) -> Dict[str, Any]:
        """Close feeds, keep positions (stops in place), save the snapshot, print the session summary."""
        self.stop_event.set()
        if self.tracker is not None:
            self.tracker.stop_stream()
        if self.md is not None:
            self.md.stop_streams()
        summary = dict(self.session)
        if self.tracker is not None:
            try:
                self.tracker.refresh()
            except Exception as exc:
                log.warning("final refresh failed", extra={"error": str(exc)})
            summary.update(end_equity=self.tracker.equity,
                           session_pnl=(self.tracker.equity - (self.session["start_equity"] or self.tracker.equity)),
                           open_positions={s: p["qty"] for s, p in self.tracker.positions.items()})
            self.save_snapshot()
        summary["alerts"] = len(self.alerts.sent) if self.alerts is not None else 0
        log.info("session summary", extra=summary)
        print("\n== Session summary ==", *(f"  {k:<16} {v}" for k, v in summary.items()), sep="\n")
        self.write_status()
        return summary
