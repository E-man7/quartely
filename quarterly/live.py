"""Live loop for the Quarterly Theory strategy.

Wakes `loop_delay_seconds` after every minute closes, fetches the closed bars
of all symbols since the last one processed, feeds them minute by minute to the
signal engine and executes entries. On startup it rebuilds today's state (TDO,
8am range, session open, day/session ranges, pivots, SMT flags) by replaying
today's bars, and reads today's trade count from the broker, so a restart
mid-session is safe.

Delayed feeds (free plan: SIP premarket, boats overnight) deliver bars ~15
minutes late, and they replace the IEX bars that stood in for those minutes.
The loop keeps today's bars in memory; when a fetch revises or adds a bar it
has already processed, it replays the day into a fresh engine (silently, no
orders), then processes only the newly closed minutes. Entries are only ever
taken on the newest minutes, never on revised history.

Paper trading unless the environment has LIVE_TRADING=true (and the operator
then types the confirmation phrase). DRY_RUN logs orders without sending them.
"""
from __future__ import annotations

import logging
import signal as signal_mod
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.data import MinuteBars, empty_frame, iter_minutes
from quarterly.execution import Executor, bracket_prices
from quarterly.signals import LONG, SHORT
from quarterly.timemodel import NY, session_times, trading_day
from quarterly.triad import make_engine

log = get_logger(__name__)
COLS = ["symbol", "ts", "open", "high", "low", "close"]

STALE_SIGNAL_SECONDS = 150          # never act on an entry from a bar older than this (catch-up after a stall)
# Backup stop: Alpaca's paper stop leg once stayed "held" while SPY traded through it for hours
# (2026-10-22). If a closed bar's close is beyond the stop by this fraction, close at market ourselves.
BACKUP_STOP_TOLERANCE = 0.0005
BACKUP_STOP_MAX_BAR_AGE = 180       # seconds: only judge on a recent bar
QUIET_LOGGERS = ("quarterly.signals", "quarterly.triad", "quarterly.nyopen", "quarterly.silverbullet")


@contextmanager
def quiet():
    """Silence level logs while replaying a day that was already logged once."""
    loggers = [logging.getLogger(n) for n in QUIET_LOGGERS]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for lg, lv in zip(loggers, levels):
            lg.setLevel(lv)


class QuarterlyBot:
    def __init__(self, cfg: QuarterlyConfig, credentials: Dict, client=None, bars: Optional[MinuteBars] = None,
                 now_fn=None):
        self.cfg = cfg
        self.creds = credentials
        self.client = client
        self.bars = bars
        self.now_fn = now_fn or (lambda: pd.Timestamp(datetime.now(NY)))
        self.engine = make_engine(cfg)
        self.executor: Optional[Executor] = None
        self.stop_event = threading.Event()
        self.last_bar: Optional[pd.Timestamp] = None
        self.flattened_day = None
        self.positions: Dict[str, float] = {}
        self.day_bars = empty_frame()               # today's bars as last fetched (all feeds merged)
        self.last_fetch: Optional[pd.Timestamp] = None
        self.replays = 0
        self.stops: Dict[str, float] = {}           # symbol -> bracket stop price of the open position

    # ------------------------------------------------------------- startup
    def connect(self) -> None:
        from broker.alpaca_client import AlpacaClient

        if self.client is None:
            # Paper unless LIVE_TRADING=true; live still asks for the typed confirmation
            settings = {"broker": {"paper_trading": not self.cfg.live_trading, "max_retries": 3}}
            self.client = AlpacaClient(settings, self.creds)
        info = self.client.health_check()
        self.bars = self.bars or MinuteBars(self.cfg, credentials=self.creds)
        from quarterly.multi import sub_config

        # one executor per strategy: its own order prefix, so trade counts and stops survive a restart
        self.executors = {s: Executor(sub_config(self.cfg, s), self.client) for s in self.cfg.strategies}
        self.executor = self.executors[self.cfg.strategies[0]]
        mode = "DRY RUN" if self.cfg.dry_run else "PAPER" if self.client.paper else "LIVE"
        log.warning("quarterly strategy starting", extra={**info, "mode": mode, "strategy": self.cfg.strategy,
                                                          "strategies": list(self.cfg.strategies),
                                                          "symbols": self.cfg.symbols,
                                                          "session_start_hour": self.cfg.session_start_hour})
        foreign = sorted(set(self.executor.positions()) - set(self.cfg.symbols))
        if foreign:
            log.warning("account holds positions outside this strategy; is the other bot still running?",
                        extra={"symbols": foreign})

    def rebuild(self) -> None:
        """Replay today's closed bars so levels, pivots and SMT flags match an uninterrupted run."""
        now = self.now_fn()
        t = session_times(trading_day(now), self.cfg.session_start_hour)
        self.day_bars = self.bars.closed_only(self.bars.fetch(self.cfg.symbols, t.day_start, now), now)
        self.last_fetch = now
        self._replay(self.day_bars)
        if self.engine.day is None:
            self.engine._new_day(trading_day(now))
        self._load_trade_counts(t.day_start)
        self.positions = self.executor.positions()
        held = {s for s in self.positions if s in self.cfg.symbols}
        self.stops = {}
        for strat, ex in self.executors.items():
            for sym, p in ex.protective_stops(t.day_start - pd.Timedelta(days=4)).items():
                if sym in held:
                    self.stops[sym] = p
                    if hasattr(self.engine, "owner"):
                        self.engine.owner[sym] = strat
        if held - set(self.stops):
            log.warning("open position without a known stop: backup stop inactive for it",
                        extra={"symbols": sorted(held - set(self.stops))})
        log.info("state rebuilt from history", extra={"bars": len(self.day_bars), "last_bar": str(self.last_bar),
                                                      **self.engine.state_summary()})

    def _sub(self, strategy: Optional[str]):
        """The engine of one strategy (the engine itself when only one runs)."""
        engines = getattr(self.engine, "engines", None)
        return engines[strategy] if engines is not None and strategy in engines else self.engine

    def _load_trade_counts(self, since: pd.Timestamp) -> None:
        for strat, ex in self.executors.items():
            for sym, n in ex.trades_today(self.engine.day, since).items():
                self._sub(strat).set_trades(sym, n)

    def _counts(self) -> Dict[str, Dict[str, int]]:
        return {s: {sym: d.trades for sym, d in self._sub(s).days.items()} for s in self.cfg.strategies}

    def _replay(self, df: pd.DataFrame) -> None:
        """Fresh engine fed `df` (no execution); keeps today's trade counts and trade owners."""
        trades = self._counts() if self.engine.day is not None else {}
        owner = dict(getattr(self.engine, "owner", {}))
        day = self.engine.day
        self.engine = make_engine(self.cfg, verbose=False)
        with quiet():
            for ts, minute in iter_minutes(df):
                self.engine.on_minute(ts, minute)
                self.last_bar = ts
        self.engine.verbose = True
        if hasattr(self.engine, "owner"):
            self.engine.owner.update(owner)
        if self.engine.day == day:
            for strat, counts in trades.items():
                for sym, n in counts.items():
                    self._sub(strat).set_trades(sym, n)

    # ---------------------------------------------------------------- step
    def step(self) -> int:
        """Process every newly closed minute. Returns the number of minutes processed."""
        now = self.now_fn()
        start = (self.last_bar + pd.Timedelta(minutes=1)) if self.last_bar is not None else now - pd.Timedelta(minutes=5)
        if self.last_fetch is not None:     # delayed feeds: re-request what they could not serve last time
            start = min(start, self.last_fetch - pd.Timedelta(minutes=self.cfg.recent_delay_minutes + 1))
        fresh = self.bars.closed_only(self.bars.fetch(self.cfg.symbols, start, now), now)
        self.last_fetch = now
        today = session_times(trading_day(now), self.cfg.session_start_hour).day_start
        old = self.day_bars[self.day_bars["ts"] >= today] if not self.day_bars.empty else self.day_bars
        merged = (pd.concat([f for f in (old, fresh) if not f.empty]) if (len(old) or len(fresh)) else empty_frame())
        if not merged.empty:
            merged = (merged[merged["ts"] >= today].drop_duplicates(["symbol", "ts"], keep="last")
                      .sort_values(["ts", "symbol"]).reset_index(drop=True))
        self.day_bars = merged
        df = merged[merged["ts"] > self.last_bar] if self.last_bar is not None and not merged.empty else merged
        if self.last_bar is not None and not merged.empty:
            seen_new = merged[merged["ts"] <= self.last_bar].reset_index(drop=True)
            seen_old = old[old["ts"] <= self.last_bar].reset_index(drop=True) if not old.empty else old
            revised = (len(seen_new) != len(seen_old)
                       or not seen_new[COLS].reset_index(drop=True).equals(seen_old[COLS].reset_index(drop=True)))
            if revised:
                before = self.last_bar
                self._replay(seen_new)
                self.last_bar = before
                self.replays += 1
                log.debug("day replayed with delayed bars", extra={"bars": len(seen_new), "last_bar": str(before)})
        processed = 0
        if not df.empty:
            self.positions = self.executor.positions()
        for ts, minute in iter_minutes(df):
            day_before = self.engine.day
            decisions = self.engine.on_minute(ts, minute, has_position=lambda s: s in self.positions)
            if self.engine.day != day_before:
                for ex in self.executors.values():
                    ex.new_day()
            self.last_bar = ts
            processed += 1
            for dec in decisions:
                if dec.action is None:
                    continue
                age = (now - (ts + pd.Timedelta(minutes=1))).total_seconds()
                if age > STALE_SIGNAL_SECONDS:
                    log.warning("entry signal too old: not executed", extra={"symbol": dec.symbol, "bar": str(ts),
                                                                            "age_s": round(age)})
                    continue
                strat = dec.strategy if dec.strategy in self.executors else self.cfg.strategy
                n = self._sub(strat).days[dec.symbol].trades + 1
                outcome = self.executors[strat].execute(dec, self.engine.day, n)
                if outcome.counts_as_trade():
                    if hasattr(self.engine, "owner"):
                        self.engine.record_trade(dec.symbol, strat)
                    else:
                        self.engine.record_trade(dec.symbol)
                    self.positions[dec.symbol] = -outcome.qty if dec.action == SHORT else outcome.qty
                    self.stops[dec.symbol] = bracket_prices(dec.action, dec.stop, dec.target)[0]
        self.enforce_stops(now)
        self.maybe_opt_out()
        self.maybe_flatten(now)
        return processed

    def maybe_opt_out(self) -> None:
        """nyopen: close a trade when another index reached its own 00:00/07:30 open first."""
        exit_signal = getattr(self.engine, "exit_signal", None)
        if exit_signal is None:
            return
        for sym, qty in list(self.positions.items()):
            if sym not in self.cfg.symbols or not qty:
                continue
            reason = exit_signal(sym, SHORT if qty < 0 else LONG)
            if reason:
                log.warning("opting out of trade", extra={"symbol": sym, "qty": qty, "reason": reason})
                self.executor.flatten([sym], reason=reason)
                self.positions.pop(sym, None)
                self.stops.pop(sym, None)

    def enforce_stops(self, now: pd.Timestamp) -> None:
        """Backup stop: close at market when the last closed bar finished beyond the bracket stop."""
        if not self.stops or self.day_bars.empty:
            return
        last = self.day_bars.sort_values("ts").groupby("symbol").last()
        for sym, stop in list(self.stops.items()):
            qty = self.positions.get(sym)
            if not qty or sym not in last.index:
                if not qty:
                    self.stops.pop(sym, None)
                continue
            bar = last.loc[sym]
            if (now - (bar["ts"] + pd.Timedelta(minutes=1))).total_seconds() > BACKUP_STOP_MAX_BAR_AGE:
                continue
            close, tol = float(bar["close"]), stop * BACKUP_STOP_TOLERANCE
            if (qty > 0 and close < stop - tol) or (qty < 0 and close > stop + tol):
                log.error("broker stop did not fill: closing at market (backup stop)",
                          extra={"symbol": sym, "qty": qty, "stop": stop, "close": close, "bar": str(bar["ts"])})
                self.executor.flatten([sym], reason="backup stop")
                self.positions.pop(sym, None)
                self.stops.pop(sym, None)

    def maybe_flatten(self, now: pd.Timestamp) -> None:
        flat = self.engine.flatten_at
        if flat is None:
            return
        if now >= flat and trading_day(now) == self.engine.day and self.flattened_day != self.engine.day:
            self.flattened_day = self.engine.day
            self.executor.flatten(self.cfg.symbols)

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        def request_stop(signum, frame):
            log.warning("shutdown requested", extra={"signal": signal_mod.Signals(signum).name})
            self.stop_event.set()

        signal_mod.signal(signal_mod.SIGINT, request_stop)
        signal_mod.signal(signal_mod.SIGTERM, request_stop)
        self.connect()
        self.rebuild()
        log.info("quarterly strategy online")
        while not self.stop_event.is_set():
            now = self.now_fn()
            # floor in UTC: in NY time the repeated hour of the autumn DST change is ambiguous and floor() raises
            wake = now.tz_convert("UTC").floor("1min") + pd.Timedelta(minutes=1, seconds=self.cfg.loop_delay_seconds)
            if self.stop_event.wait((wake - now).total_seconds()):
                break
            try:
                self.step()
            except Exception:
                log.exception("quarterly step failed; will retry next minute")
        log.info("quarterly strategy stopped (open positions keep their bracket orders)")
