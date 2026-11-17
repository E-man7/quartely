"""Signal engine for the Quarterly Theory NY "Q3 of Q3" model (strategy: q3, dormant by default).

Pure logic, no I/O: the live loop and the backtest both call
`StrategyEngine.on_minute(...)` with the closed 1-minute bars of every symbol
for one minute, in time order.

Per bar (all in America/New_York):
* reference levels: True Day Open (first bar at/after 00:00, else the
  configured fallback), NY Session True Open (first bar at/after Q2 start),
  day high/low since 18:00 the previous day.
* session high/low of every symbol from session start; from `smt_from` on, a
  bearish SMT is set when, for any of the symbol's partners, exactly one of
  the two makes a new session high on this bar (bullish: new low). With a
  group such as QQQ/DIA/SPY every member is checked against each of the
  others. Flags stay set until the next session start. A partner is only
  compared on minutes where both have a bar.
* swing pivots of strength `pivot_len`, confirmed only once the right-side bars
  have closed; a market structure shift is a close crossing the most recent
  pivot confirmed before this bar.
* bias: SELL if close > TDO and > session open; BUY if below both.
* entry: inside the window, bias + matching SMT flag + MSS on this bar, no
  position, under the per-day trade limit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Deque, Dict, List, Optional, Tuple

import pandas as pd

from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.timemodel import SessionTimes, clock, session_times, to_ny, trading_day

log = get_logger(__name__)

SHORT, LONG = "SHORT", "LONG"


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: pd.Timestamp          # bar open time, NY
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class PivotTracker:
    """Swing highs/lows of strength L, confirmed when the L bars to the right have closed.

    A pivot high needs a high strictly above the L highs on each side (lows
    likewise). Nothing about a bar is known before its right side has closed.
    """

    def __init__(self, length: int):
        self.length = length
        self.window: Deque[Tuple[pd.Timestamp, float, float]] = deque(maxlen=2 * length + 1)
        self.swing_high: Optional[float] = None
        self.swing_low: Optional[float] = None
        self.swing_high_ts: Optional[pd.Timestamp] = None
        self.swing_low_ts: Optional[pd.Timestamp] = None

    def update(self, bar: Bar) -> Tuple[bool, bool]:
        """Add a closed bar; returns (new swing high confirmed, new swing low confirmed)."""
        self.window.append((bar.ts, bar.high, bar.low))
        if len(self.window) < self.window.maxlen:
            return False, False
        L = self.length
        ts, h, lo = self.window[L]
        others = [w for i, w in enumerate(self.window) if i != L]
        new_high = all(h > w[1] for w in others)
        new_low = all(lo < w[2] for w in others)
        if new_high:
            self.swing_high, self.swing_high_ts = h, ts
        if new_low:
            self.swing_low, self.swing_low_ts = lo, ts
        return new_high, new_low


def update_tdo(d, bar: Bar, t: SessionTimes, cfg: QuarterlyConfig) -> None:
    """True Day Open: the 00:00 open. The first bar 00:00-00:<tdo_search_minutes>, else the configured fallback:
    last_close_before_midnight (the price at 00:00 is the last trade before it, if one came after 20:00),
    first_premarket_bar, or skip_day.

    `d` is any per-day state with tdo / tdo_source / skip_reason attributes; bars before midnight are passed
    in too, so the last close before 00:00 is known.
    """
    if bar.ts < t.midnight:
        if bar.ts >= t.midnight - timedelta(hours=4):
            d.pre_midnight_close = bar.close
        return
    if d.tdo is not None or d.skip_reason is not None:
        return
    pre = getattr(d, "pre_midnight_close", None)
    if bar.ts < t.midnight + timedelta(minutes=cfg.tdo_search_minutes):
        d.tdo, d.tdo_source = bar.open, "midnight"
    elif cfg.tdo_fallback == "last_close_before_midnight" and pre is not None:
        d.tdo, d.tdo_source = pre, "last_close_before_midnight"
    elif cfg.tdo_fallback in ("skip_day", "last_close_before_midnight"):
        d.skip_reason = f"no bar 00:00-00:{cfg.tdo_search_minutes:02d} NY for the True Day Open (skip_day)"
    elif bar.ts >= clock(t.day, cfg.premarket_start):
        d.tdo, d.tdo_source = bar.open, "first_premarket_bar"
    if d.tdo is not None or d.skip_reason:
        log.info("true day open", extra={"symbol": bar.symbol, "tdo": d.tdo, "source": d.tdo_source,
                                         "bar": str(bar.ts), "skip": d.skip_reason})


@dataclass
class SymbolDay:
    """One symbol's state for one trading day (from 18:00 the previous day)."""
    tdo: Optional[float] = None
    tdo_source: Optional[str] = None
    skip_reason: Optional[str] = None
    pre_midnight_close: Optional[float] = None
    session_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    prev_close: Optional[float] = None
    trades: int = 0
    pivots: Optional[PivotTracker] = None


@dataclass
class SymbolSession:
    """Session high/low and SMT flags; reset at each session start."""
    high: Optional[float] = None
    low: Optional[float] = None
    smt_bear: bool = False
    smt_bull: bool = False
    smt_bear_ts: Optional[pd.Timestamp] = None
    smt_bull_ts: Optional[pd.Timestamp] = None
    smt_bear_partner: Optional[str] = None       # the partner that diverged first
    smt_bull_partner: Optional[str] = None


@dataclass
class Decision:
    symbol: str
    ts: pd.Timestamp
    action: Optional[str]                  # SHORT | LONG | None
    reasons: List[str] = field(default_factory=list)
    bias: Optional[str] = None             # SELL | BUY | None
    in_window: bool = False
    smt_bear: bool = False
    smt_bull: bool = False
    mss_bear: bool = False
    mss_bull: bool = False
    levels: Dict[str, Optional[float]] = field(default_factory=dict)
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    risk_per_share: Optional[float] = None
    strategy: Optional[str] = None         # set when several strategies run together

    def as_dict(self) -> Dict:
        return {"symbol": self.symbol, "strategy": self.strategy, "bar": str(self.ts), "action": self.action, "bias": self.bias,
                "in_window": self.in_window, "smt_bear": self.smt_bear, "smt_bull": self.smt_bull,
                "mss_bear": self.mss_bear, "mss_bull": self.mss_bull,
                "levels": {k: (round(v, 4) if v is not None else None) for k, v in self.levels.items()},
                "entry": self.entry, "stop": self.stop, "target": self.target, "risk_per_share": self.risk_per_share,
                "reasons": self.reasons}


class StrategyEngine:
    def __init__(self, cfg: QuarterlyConfig, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.partners = cfg.partners
        self.day: Optional[date] = None
        self.times: Optional[SessionTimes] = None
        self.days: Dict[str, SymbolDay] = {}
        self.sessions: Dict[str, SymbolSession] = {s: SymbolSession() for s in cfg.symbols}
        self.session_day: Optional[date] = None
        self.last_ts: Optional[pd.Timestamp] = None

    # --------------------------------------------------------------- clock
    def smt_from_time(self) -> pd.Timestamp:
        t = self.times
        return {"session_start": t.session_start, "session_q3": t.q3_start, "window_start": t.window_start}[
            self.cfg.smt_from]

    def _new_day(self, day: date) -> None:
        self.day = day
        self.times = session_times(day, self.cfg.session_start_hour)
        self.days = {s: SymbolDay(pivots=PivotTracker(self.cfg.pivot_len)) for s in self.cfg.symbols}
        log.info("new trading day", extra={"day": str(day), "window_start": str(self.times.window_start),
                                           "window_end": str(self.times.window_end)})

    def set_trades(self, symbol: str, n: int) -> None:
        """Trades already placed today (e.g. found in the broker's order history after a restart)."""
        if symbol in self.days:
            self.days[symbol].trades = n

    def record_trade(self, symbol: str) -> None:
        self.days[symbol].trades += 1

    @property
    def flatten_at(self) -> Optional[pd.Timestamp]:
        """Close open positions from this time on (None: hold, the bracket legs manage the exit)."""
        return self.times.session_end if self.cfg.flat_at_session_end and self.times is not None else None

    def trade_context(self, dec: "Decision") -> Dict:
        """Levels at an entry signal, for the backtest's trade records and the chart."""
        d, s = self.days[dec.symbol], self.sessions[dec.symbol]
        short = dec.action == SHORT
        return dict(partner=s.smt_bear_partner if short else s.smt_bull_partner,
                    tdo=d.tdo, tdo_source=d.tdo_source, session_open=d.session_open,
                    day_high=d.day_high, day_low=d.day_low, smt_ts=s.smt_bear_ts if short else s.smt_bull_ts,
                    pivot_ts=d.pivots.swing_low_ts if short else d.pivots.swing_high_ts,
                    pivot_price=d.pivots.swing_low if short else d.pivots.swing_high)

    def state_summary(self) -> Dict:
        return {"levels": {s: {"tdo": d.tdo, "tdo_source": d.tdo_source, "session_open": d.session_open,
                               "day_high": d.day_high, "day_low": d.day_low, "trades": d.trades,
                               "skip": d.skip_reason} for s, d in self.days.items()},
                "smt": {s: {"bear": v.smt_bear, "bull": v.smt_bull} for s, v in self.sessions.items()}}

    # -------------------------------------------------------------- levels
    def _update_levels(self, bar: Bar) -> Tuple[bool, bool, bool, bool]:
        """Update one symbol's levels; returns (mss_bear, mss_bull, new_session_high, new_session_low)."""
        d, s, t = self.days[bar.symbol], self.sessions[bar.symbol], self.times
        update_tdo(d, bar, t, self.cfg)
        # NY Session True Open
        if d.session_open is None and bar.ts >= t.q2_start:
            d.session_open = bar.open
            log.info("session true open", extra={"symbol": bar.symbol, "open": bar.open, "bar": str(bar.ts)})
        # Day range since 18:00 the previous day
        d.day_high = bar.high if d.day_high is None else max(d.day_high, bar.high)
        d.day_low = bar.low if d.day_low is None else min(d.day_low, bar.low)
        # Session range (prior values decide whether this bar made a new one)
        new_high = new_low = False
        if t.in_session(bar.ts):
            new_high = s.high is not None and bar.high > s.high
            new_low = s.low is not None and bar.low < s.low
            s.high = bar.high if s.high is None else max(s.high, bar.high)
            s.low = bar.low if s.low is None else min(s.low, bar.low)
        # MSS against pivots confirmed before this bar; then let this bar confirm new ones
        p = d.pivots
        mss_bear = (p.swing_low is not None and d.prev_close is not None
                    and d.prev_close >= p.swing_low and bar.close < p.swing_low)
        mss_bull = (p.swing_high is not None and d.prev_close is not None
                    and d.prev_close <= p.swing_high and bar.close > p.swing_high)
        p.update(bar)
        d.prev_close = bar.close
        return mss_bear, mss_bull, new_high, new_low

    def _update_smt(self, ts: pd.Timestamp, symbol: str, extremes: Dict[str, Tuple[bool, bool]]) -> Optional[str]:
        """Set SMT flags for `symbol` against each partner. Returns a note about partners without a bar, or None."""
        t = self.times
        if not (self.smt_from_time() <= ts < t.session_end):
            return None
        sess = self.sessions[symbol]
        s_high, s_low = extremes[symbol]
        missing = []
        for pair in self.partners[symbol]:
            if pair.partner not in extremes:
                missing.append(pair.partner)
                continue
            p_high, p_low = extremes[pair.partner]
            if pair.inverse:            # inversely correlated partner: its lows mirror our highs
                p_high, p_low = p_low, p_high
            if s_high != p_high and not sess.smt_bear:
                sess.smt_bear, sess.smt_bear_ts, sess.smt_bear_partner = True, ts, pair.partner
                log.info("bearish SMT", extra={"symbol": symbol, "partner": pair.partner, "bar": str(ts),
                                               "new_high_by": symbol if s_high else pair.partner, "inverse": pair.inverse})
            if s_low != p_low and not sess.smt_bull:
                sess.smt_bull, sess.smt_bull_ts, sess.smt_bull_partner = True, ts, pair.partner
                log.info("bullish SMT", extra={"symbol": symbol, "partner": pair.partner, "bar": str(ts),
                                               "new_low_by": symbol if s_low else pair.partner, "inverse": pair.inverse})
        if missing:
            evaluated = len(self.partners[symbol]) - len(missing)
            return (f"no {', '.join(missing)} bar this minute: SMT "
                    f"{'not evaluated' if not evaluated else 'checked against the other partners only'}")
        return None

    # ---------------------------------------------------------------- step
    def on_minute(self, ts, bars: Dict[str, Bar], has_position: Callable[[str], bool] = lambda s: False
                  ) -> List[Decision]:
        """Process the closed bars of one minute (any subset of symbols). Returns one decision per bar."""
        ts = to_ny(ts)
        if self.last_ts is not None and ts <= self.last_ts:
            return []                                    # already processed: never act twice on a bar
        self.last_ts = ts
        day = trading_day(ts)
        if day != self.day:
            self._new_day(day)
        if ts >= self.times.session_start and self.session_day != day:
            self.sessions = {s: SymbolSession() for s in self.cfg.symbols}   # SMT flags last until here
            self.session_day = day

        bars = {s: b for s, b in bars.items() if s in self.days}
        mss: Dict[str, Tuple[bool, bool]] = {}
        extremes: Dict[str, Tuple[bool, bool]] = {}
        for sym, bar in bars.items():
            mb, mu, nh, nl = self._update_levels(bar)
            mss[sym], extremes[sym] = (mb, mu), (nh, nl)
        smt_notes = {sym: self._update_smt(ts, sym, extremes) for sym in bars}
        return [self._decide(sym, bar, mss[sym], smt_notes[sym], has_position(sym)) for sym, bar in bars.items()]

    def _decide(self, sym: str, bar: Bar, mss: Tuple[bool, bool], smt_note: Optional[str], in_position: bool
                ) -> Decision:
        d, s, t = self.days[sym], self.sessions[sym], self.times
        p = d.pivots
        dec = Decision(sym, bar.ts, None, in_window=t.in_window(bar.ts), smt_bear=s.smt_bear, smt_bull=s.smt_bull,
                       mss_bear=mss[0], mss_bull=mss[1],
                       levels={"close": bar.close, "tdo": d.tdo, "session_open": d.session_open,
                               "day_high": d.day_high, "day_low": d.day_low, "session_high": s.high,
                               "session_low": s.low, "swing_high": p.swing_high, "swing_low": p.swing_low})
        notes: List[str] = [smt_note] if smt_note else []     # informational
        r: List[str] = []                                       # conditions blocking an entry
        if d.tdo is not None and d.session_open is not None:
            if bar.close > d.tdo and bar.close > d.session_open:
                dec.bias = "SELL"
            elif bar.close < d.tdo and bar.close < d.session_open:
                dec.bias = "BUY"

        if d.skip_reason:
            r.append(d.skip_reason)
        elif d.tdo is None:
            r.append("True Day Open not set yet")
        elif d.session_open is None:
            r.append("NY Session True Open not set yet (before Q2)")
        elif dec.bias is None:
            r.append(f"no bias: close {bar.close} between TDO {d.tdo} and session open {d.session_open}")
        if not dec.in_window:
            r.append("outside entry window")
        side = None
        if dec.bias == "SELL":
            side = SHORT
            if not s.smt_bear:
                r.append("sell bias but no bearish SMT")
            if not mss[0]:
                r.append("sell bias but no bearish MSS on this bar")
        elif dec.bias == "BUY":
            side = LONG
            if not s.smt_bull:
                r.append("buy bias but no bullish SMT")
            if not mss[1]:
                r.append("buy bias but no bullish MSS on this bar")
        if in_position:
            r.append("already in a position")
        if d.trades >= self.cfg.max_trades_per_symbol_per_day:
            r.append(f"trade limit reached ({d.trades}/{self.cfg.max_trades_per_symbol_per_day} today)")

        if side is not None and not r:
            entry = bar.close
            if side == SHORT:
                stop = d.day_high + self.cfg.stop_buffer
                risk = stop - entry
                target = entry - self.cfg.rr * risk
            else:
                stop = d.day_low - self.cfg.stop_buffer
                risk = entry - stop
                target = entry + self.cfg.rr * risk
            if risk <= 0:
                r.append(f"risk per share {risk:.4f} <= 0")
            else:
                dec.action, dec.entry, dec.stop, dec.target, dec.risk_per_share = side, entry, stop, target, risk
                notes.append(f"{side}: {dec.bias} bias, {'bearish' if side == SHORT else 'bullish'} SMT and MSS "
                             f"inside the entry window")
        dec.reasons = notes + r

        if self.verbose and (dec.in_window or dec.action):
            log.info("decision", extra=dec.as_dict())
        else:
            log.debug("decision", extra=dec.as_dict())
        return dec
