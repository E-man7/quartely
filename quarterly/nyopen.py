"""Signal engine for the NY open model (strategy: nyopen), the Python port of
"Quarterly Theory NY Open v2" on QQQ / DIA / SPY (Nasdaq 100, Dow 30, S&P 500).

Per index (America/New_York, so daylight saving is handled by the clock):
* the 00:00 open (True Day Open, per `tdo_fallback`) and the 07:30 open
  (first bar at/after 07:30);
* day high/low since 18:00 the previous day;
* high/low since `smt_from` (07:30 by default).

SMT is one flag for the triad, reset at `smt_from`: bearish when, on a minute
all three indices trade, some but not all of them make a new high since
`smt_from` (bullish: new low).

Entries are judged once per entry time (09:30 and/or 09:45), on the bar that
closes at that time (opening a minute earlier; if an index has no bar then,
the first bar of the next two minutes):
* SELL bias: close above both opens; BUY bias: below both;
* the matching SMT flag;
* no other index on the same side has taken out its own open first: for a
  sell, another index that traded above both its opens and has since come
  back down to the higher one (buys: below both, back up to the lower one).
  An index on the other side of its opens, or between them, doesn't count;
* momentum: the index's % change over the last `momentum_bars` closes is the
  strongest of the three in the trade direction, so at most one index trades;
* optionally a market structure shift since `smt_from`.
Stop at the day high/low (+ buffer), target the higher open (sells) / lower
open (buys), i.e. whichever open price reaches first. One trade per day.

Opt out: while in a trade, if another index on the same side takes out its own
open after the entry, the trade is closed at market (`exit_signal`).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Deque, Dict, List, Optional

import pandas as pd

from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.signals import LONG, SHORT, Bar, Decision, PivotTracker, update_tdo
from quarterly.timemodel import SessionTimes, clock, session_times, to_ny, trading_day

log = get_logger(__name__)


@dataclass
class IndexDay:
    tdo: Optional[float] = None
    tdo_source: Optional[str] = None
    skip_reason: Optional[str] = None
    pre_midnight_close: Optional[float] = None
    o0730: Optional[float] = None
    h0730: Optional[float] = None             # the 07:30 one-minute candle (first bar at/after 07:30)
    l0730: Optional[float] = None
    h0830: Optional[float] = None             # the 08:30 one-minute candle (first bar at/after 08:30)
    l0830: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    run_high: Optional[float] = None          # since smt_from
    run_low: Optional[float] = None
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    last_close: Optional[float] = None
    closes: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    ranges: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    last_ts: Optional[pd.Timestamp] = None
    prev_close: Optional[float] = None
    pivots: Optional[PivotTracker] = None
    mss_bear_seen: bool = False
    mss_bull_seen: bool = False
    was_above: bool = False                   # since smt_from: a bar entirely above both opens
    was_below: bool = False
    took_sell_ts: Optional[pd.Timestamp] = None   # then came back down to the higher open
    took_buy_ts: Optional[pd.Timestamp] = None    # (was below both, back up to the lower open)
    trades: int = 0

    def ready(self) -> bool:
        return self.tdo is not None and self.o0730 is not None

    def sell_level(self) -> Optional[float]:
        return max(self.tdo, self.o0730) if self.ready() else None

    def buy_level(self) -> Optional[float]:
        return min(self.tdo, self.o0730) if self.ready() else None

    def took_sell_level(self, rule: str = "crossed", after: Optional[pd.Timestamp] = None) -> bool:
        if rule == "beyond":
            return self.ready() and self.last_low is not None and self.last_low <= self.sell_level()
        if rule == "after_entry" and after is None:
            return False
        return self.took_sell_ts is not None and (after is None or self.took_sell_ts > after)

    def took_buy_level(self, rule: str = "crossed", after: Optional[pd.Timestamp] = None) -> bool:
        if rule == "beyond":
            return self.ready() and self.last_high is not None and self.last_high >= self.buy_level()
        if rule == "after_entry" and after is None:
            return False
        return self.took_buy_ts is not None and (after is None or self.took_buy_ts > after)

    def track_opens(self, bar: Bar) -> None:
        """Was this index beyond its opens, and has it come back to them since? (prior bars decide)"""
        if not self.ready():
            return
        hi_open, lo_open = self.sell_level(), self.buy_level()
        if self.was_above and self.took_sell_ts is None and bar.low <= hi_open:
            self.took_sell_ts = bar.ts
        if self.was_below and self.took_buy_ts is None and bar.high >= lo_open:
            self.took_buy_ts = bar.ts
        self.was_above = self.was_above or bar.low > hi_open
        self.was_below = self.was_below or bar.high < lo_open

    def momentum(self, n: int) -> Optional[float]:
        if len(self.closes) <= n:
            return None
        return (self.closes[-1] / self.closes[-1 - n] - 1.0) * 100.0


class NyOpenEngine:
    def __init__(self, cfg: QuarterlyConfig, verbose: bool = True):
        self.cfg = cfg
        self.nc = cfg.nyopen
        self.verbose = verbose
        self.symbols = cfg.symbols
        self.day: Optional[date] = None
        self.times: Optional[SessionTimes] = None
        self.days: Dict[str, IndexDay] = {}
        self.last_ts: Optional[pd.Timestamp] = None
        self.smt_bear = self.smt_bull = False
        self.smt_bear_ts: Optional[pd.Timestamp] = None
        self.smt_bull_ts: Optional[pd.Timestamp] = None
        self.judged: set = set()                      # entry times already judged today
        self.context: Dict[str, Dict] = {}

    # --------------------------------------------------------------- clock
    def _new_day(self, day: date) -> None:
        self.day = day
        self.times = session_times(day, self.cfg.session_start_hour)
        self.t0730 = clock(day, (7, 30))
        self.t0830 = clock(day, (8, 30))
        self.smt_start = clock(day, self.nc.smt_from)
        self.entries = [clock(day, t) for t in self.nc.entry_times]
        self.days = {s: IndexDay(pivots=PivotTracker(self.cfg.pivot_len)) for s in self.symbols}
        self.smt_bear = self.smt_bull = False
        self.smt_bear_ts = self.smt_bull_ts = None
        self.judged, self.context = set(), {}
        log.info("new trading day", extra={"day": str(day), "strategy": "nyopen",
                                           "entries": [str(t) for t in self.entries]})

    @property
    def flatten_at(self) -> Optional[pd.Timestamp]:
        return clock(self.day, self.nc.flat_at) if self.day is not None and self.nc.flat_at else None

    def set_trades(self, symbol: str, n: int) -> None:
        if symbol in self.days:
            self.days[symbol].trades = n

    def record_trade(self, symbol: str) -> None:
        self.days[symbol].trades += 1

    def _slot(self, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        """The entry time whose judging bar this is: the bar closing at it, or one of the two after."""
        for t in self.entries:
            if t - pd.Timedelta(minutes=1) <= ts < t + pd.Timedelta(minutes=2) and t not in self.judged:
                return t
        return None

    # -------------------------------------------------------------- levels
    def _update(self, bar: Bar) -> "tuple[bool, bool]":
        d = self.days[bar.symbol]
        update_tdo(d, bar, self.times, self.cfg)
        if d.o0730 is None and bar.ts >= self.t0730:
            d.o0730, d.h0730, d.l0730 = bar.open, bar.high, bar.low
        if d.h0830 is None and bar.ts >= self.t0830:
            d.h0830, d.l0830 = bar.high, bar.low
        d.day_high = bar.high if d.day_high is None else max(d.day_high, bar.high)
        d.day_low = bar.low if d.day_low is None else min(d.day_low, bar.low)
        new_high = new_low = False
        if bar.ts >= self.smt_start:
            if d.run_high is None:
                d.run_high, d.run_low = bar.high, bar.low
            else:
                new_high, new_low = bar.high > d.run_high, bar.low < d.run_low
                d.run_high, d.run_low = max(d.run_high, bar.high), min(d.run_low, bar.low)
        if bar.ts >= self.smt_start:
            d.track_opens(bar)
        p = d.pivots
        if bar.ts >= self.smt_start and d.prev_close is not None:
            if p.swing_low is not None and d.prev_close >= p.swing_low > bar.close:
                d.mss_bear_seen = True
            if p.swing_high is not None and d.prev_close <= p.swing_high < bar.close:
                d.mss_bull_seen = True
        p.update(bar)
        d.prev_close = d.last_close = bar.close
        d.last_high, d.last_low, d.last_ts = bar.high, bar.low, bar.ts
        d.closes.append(bar.close)
        d.ranges.append(bar.high - bar.low)
        return new_high, new_low

    # ---------------------------------------------------------------- step
    def on_minute(self, ts, bars: Dict[str, Bar], has_position: Callable[[str], bool] = lambda s: False
                  ) -> List[Decision]:
        ts = to_ny(ts)
        if self.last_ts is not None and ts <= self.last_ts:
            return []
        self.last_ts = ts
        day = trading_day(ts)
        if day != self.day:
            self._new_day(day)
        bars = {s: b for s, b in bars.items() if s in self.days}
        ext = {s: self._update(b) for s, b in bars.items()}
        if ts >= self.smt_start and len(ext) == len(self.symbols):
            highs = sum(h for h, _ in ext.values())
            lows = sum(lo for _, lo in ext.values())
            if 0 < highs < len(self.symbols) and not self.smt_bear:
                self.smt_bear, self.smt_bear_ts = True, ts
                log.info("bearish SMT", extra={"bar": str(ts), "new_high_by": [s for s, (h, _) in ext.items() if h]})
            if 0 < lows < len(self.symbols) and not self.smt_bull:
                self.smt_bull, self.smt_bull_ts = True, ts
                log.info("bullish SMT", extra={"bar": str(ts), "new_low_by": [s for s, (_, lo) in ext.items() if lo]})
        slot = self._slot(ts)
        if slot is None:
            return []
        # wait_for_all_bars: judge once every index has a bar, or at the last minute of the slot;
        # otherwise judge on the bar closing at the entry time (exactly 09:30), latest prices for the rest
        if self.nc.wait_for_all_bars and len(bars) < len(self.symbols) and ts < slot + pd.Timedelta(minutes=1):
            return []
        self.judged.add(slot)
        decs = [self._decide(sym, ts, slot, has_position(sym)) for sym in self.symbols]
        if self.nc.other_open_rule == "first_to_target":
            return self._race(decs, ts)
        live = [d for d in decs if d.action]
        if len(live) > 1:                     # a sell and a buy at once: one trade a day, the stronger move
            best = max(live, key=lambda d: abs(d.levels.get("momentum_pct") or 0))
            for d in live:
                if d is not best:
                    d.reasons.append(f"{best.symbol}'s {best.action} has the stronger momentum "
                                     f"({best.levels['momentum_pct']:+.3f}% vs {d.levels['momentum_pct']:+.3f}%)")
                    d.action = d.entry = d.stop = d.target = d.risk_per_share = None
                    self.context.pop(d.symbol, None)
        return decs

    def _decide(self, sym: str, ts: pd.Timestamp, slot: pd.Timestamp, in_position: bool) -> Decision:
        d, nc = self.days[sym], self.nc
        others = [self.days[s] for s in self.symbols if s != sym]
        n = nc.momentum_bars
        mom = {s: self.days[s].momentum(n) for s in self.symbols}
        dec = Decision(sym, ts, None, in_window=True, smt_bear=self.smt_bear, smt_bull=self.smt_bull,
                       levels={"close": d.last_close, "open_0000": d.tdo, "open_0730": d.o0730,
                               "high_0730": d.h0730, "low_0730": d.l0730, "high_0830": d.h0830, "low_0830": d.l0830,
                               "day_high": d.day_high, "day_low": d.day_low, "momentum_pct": mom[sym]})
        r: List[str] = []
        ready = d.ready() and all(o.ready() for o in others)
        if d.skip_reason:
            r.append(d.skip_reason)
        if not ready:
            r.append("00:00 / 07:30 opens not known for every index")
        if d.last_close is None:
            r.append("no bar yet")
        side = None
        if ready and d.last_close is not None:
            if d.last_close > d.sell_level():
                dec.bias, side = "SELL", SHORT
            elif d.last_close < d.buy_level():
                dec.bias, side = "BUY", LONG
            else:
                r.append(f"no bias: close {d.last_close} between the 00:00 open {d.tdo} and the 07:30 open {d.o0730}")
        if side == SHORT:
            if not self.smt_bear:
                r.append("sell bias but no bearish SMT since " + "{:02d}:{:02d}".format(*nc.smt_from))
            took = [s for s in self.symbols if s != sym and self.days[s].took_sell_level(nc.other_open_rule)]
            if took and nc.other_open_rule != "first_to_target":
                r.append(f"{', '.join(took)} already at/through its 00:00/07:30 open ({nc.other_open_rule} rule)")
            if nc.require_mss and not d.mss_bear_seen:
                r.append("no bearish market structure shift since the SMT start")
        elif side == LONG:
            if not self.smt_bull:
                r.append("buy bias but no bullish SMT since " + "{:02d}:{:02d}".format(*nc.smt_from))
            took = [s for s in self.symbols if s != sym and self.days[s].took_buy_level(nc.other_open_rule)]
            if took and nc.other_open_rule != "first_to_target":
                r.append(f"{', '.join(took)} already at/through its 00:00/07:30 open ({nc.other_open_rule} rule)")
            if nc.require_mss and not d.mss_bull_seen:
                r.append("no bullish market structure shift since the SMT start")
        if side is not None and nc.other_open_rule != "first_to_target":
            vals = [m for m in mom.values() if m is not None]
            if mom[sym] is None or len(vals) < len(self.symbols):
                r.append(f"momentum over {n} bars not known for every index")
            elif (side == SHORT and mom[sym] > min(vals)) or (side == LONG and mom[sym] < max(vals)):
                best = (min if side == SHORT else max)(mom, key=lambda s: mom[s] if mom[s] is not None else 0)
                r.append(f"{best} has the stronger momentum ({mom[best]:+.3f}% vs {mom[sym]:+.3f}%)")
        if in_position:
            r.append("already in a position")
        if any(x.trades for x in self.days.values()):
            r.append("one trade per day: already traded today")

        notes: List[str] = []
        if side is not None and not r:
            entry = d.last_close
            target = self.target_of(d, side)
        if side is not None and not r and target is None:
            r.append(f"no {nc.target} level yet for {sym}")
        if side is not None and not r:
            if side == SHORT:
                stop = d.day_high + nc.stop_buffer
                risk, reward = stop - entry, entry - target
            else:
                stop = d.day_low - nc.stop_buffer
                risk, reward = entry - stop, target - entry
            if risk <= 0 or reward <= 0:
                r.append(f"risk {risk:.4f} / reward {reward:.4f} not both positive")
            elif nc.min_rr and reward / risk < nc.min_rr:
                r.append(f"reward:risk {reward / risk:.2f} below the minimum {nc.min_rr}")
            else:
                if nc.target_rr:   # fixed reward:risk target in place of the 07:30 level
                    target = round(entry - nc.target_rr * risk if side == SHORT else entry + nc.target_rr * risk, 4)
                    reward = nc.target_rr * risk
                dec.action, dec.entry, dec.stop, dec.target, dec.risk_per_share = side, entry, stop, target, risk
                self.context[sym] = {"slot": slot, "ts": ts, "rr": reward / risk}
                notes.append(f"{side} at {slot:%H:%M}: {dec.bias} bias, SMT, momentum {mom[sym] or 0:+.3f}%, "
                             f"target {target} (reward:risk {reward / risk:.2f})")
        dec.reasons = notes + r
        (log.info if self.verbose else log.debug)("decision", extra={**dec.as_dict(), "entry_time": f"{slot:%H:%M}"})
        return dec

    # ------------------------------------------------------- first to target
    def target_of(self, d: IndexDay, side: str) -> Optional[float]:
        if self.nc.target == "0830_candle":
            return d.l0830 if side == SHORT else d.h0830
        if self.nc.target == "0730_candle":
            return d.l0730 if side == SHORT else d.h0730
        return d.sell_level() if side == SHORT else d.buy_level()

    def _score(self, dec: Decision) -> float:
        """Lower is better: how hard it looks for this setup to reach its target first."""
        d, pick = self.days[dec.symbol], self.nc.pick
        reward = abs(dec.entry - dec.target)
        if pick == "momentum":
            m = d.momentum(self.nc.momentum_bars) or 0.0
            return m if dec.action == SHORT else -m
        if pick == "nearest_target":
            return reward / dec.entry
        if pick == "vol_units":
            atr = sum(d.ranges) / len(d.ranges) if d.ranges else 0.0
            return reward / atr if atr > 0 else float("inf")
        return -dec.risk_per_share / (dec.risk_per_share + reward)          # rw_probability, as a cost

    def _race(self, decs: List[Decision], ts: pd.Timestamp) -> List[Decision]:
        """Trade the setup most likely to reach its target first; the other setups are its rivals."""
        live = [d for d in decs if d.action]
        if not live:
            return decs
        best = min(live, key=self._score)
        for d in live:
            if d is not best:
                d.reasons.append(f"not picked: {best.symbol} looks likelier to reach its target first "
                                 f"({self.nc.pick} {self._score(best):.4f} vs {self._score(d):.4f})")
                d.action = d.entry = d.stop = d.target = d.risk_per_share = None
                self.context.pop(d.symbol, None)
        rivals = {}
        for s in self.symbols:                          # every other index with a side at the entry
            x = self.days[s]
            if s == best.symbol or not x.ready() or x.last_close is None:
                continue
            if x.last_close > x.sell_level():
                rivals[s] = (SHORT, self.target_of(x, SHORT))
            elif x.last_close < x.buy_level():
                rivals[s] = (LONG, self.target_of(x, LONG))
        self.context[best.symbol]["rivals"] = rivals
        best.reasons.append(f"picked by {self.nc.pick}; rivals {', '.join(rivals) or 'none'}")
        return decs

    # ------------------------------------------------------------- opt out
    def exit_signal(self, sym: str, side: str) -> Optional[str]:
        """Reason to close an open trade now: another index traded to its own target level first."""
        if not self.nc.opt_out or sym not in self.days:
            return None
        rule = self.nc.other_open_rule
        if rule == "first_to_target":
            ctx = self.context.get(sym, {})
            since = ctx.get("ts")
            won = [s for s, (rside, target) in ctx.get("rivals", {}).items()
                   if since is not None and target is not None and self.days[s].last_ts is not None and self.days[s].last_ts > since
                   and (self.days[s].last_low <= target if rside == SHORT else self.days[s].last_high >= target)]
            return f"opt out: {', '.join(won)} reached its own target first" if won else None
        since = None if rule == "beyond" else self.context.get(sym, {}).get("ts")
        took = [s for s in self.symbols if s != sym
                and (self.days[s].took_sell_level(rule, since) if side == SHORT
                     else self.days[s].took_buy_level(rule, since))]
        return f"opt out: {', '.join(took)} took out its 00:00/07:30 open first" if took else None

    # ------------------------------------------------------------- reports
    def trade_context(self, dec: Decision) -> Dict:
        d = self.days[dec.symbol]
        short = dec.action == SHORT
        return dict(tdo=d.tdo, tdo_source=d.tdo_source, session_open=d.o0730, day_high=d.day_high, day_low=d.day_low,
                    smt_ts=self.smt_bear_ts if short else self.smt_bull_ts)

    def state_summary(self) -> Dict:
        return {"levels": {s: {"open_0000": d.tdo, "open_0730": d.o0730, "day_high": d.day_high, "day_low": d.day_low,
                               "trades": d.trades, "skip": d.skip_reason} for s, d in self.days.items()},
                "smt": {"bear": self.smt_bear, "bull": self.smt_bull}}
