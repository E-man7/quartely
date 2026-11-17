"""Signal engine for the index triad 9:30 manipulation (strategy: triad930).

Port of the Pine Script "QT Triad 9:30 Manipulation" onto QQQ / DIA / SPY
(Nasdaq 100, Dow 30, S&P 500 proxies). Pure logic, same interface as
`signals.StrategyEngine`, so the live loop and the backtest drive it unchanged.

Per symbol and day (America/New_York):
* True Day Open: the 00:00 bar's open (fallback per `tdo_fallback`);
  day high/low from 00:00.
* 8am range: high/low of the bars opening 08:00-08:59. The range is done
  from the first bar after it; from then on (or only inside the window with
  `sweep_in_window`) a high above it "takes the 8am high", a low below it
  "takes the 8am low". Both flags stay set for the rest of the day.
* window: bars opening 09:30-09:32; window high/low (for stop_mode: window).

A leader is any of the other members of the symbol's group:
* bearish leader: took its 8am high, not its 8am low, and its latest bar
  traded above its TDO;
* bullish leader: the mirror (8am low, below its TDO).
A leader that took both sides is no leader.

SELL the symbol, on a bar inside the window, when some leader is bearish and
the symbol lags: it has taken neither side of its own 8am range (strict_lag),
or at least not its 8am high. BUY is the mirror. Optionally the symbol must
also close above (sells) / below (buys) its own TDO. One trade per symbol per
day, stop at the day high/low (or window high/low) plus a buffer, target at
rr x risk, open trades closed at `flat_at` (15:55).

With `pick` tightest_stop (default) / widest_stop, when several symbols signal
on the same minute only the one with the tightest / widest stop as a percent of
its price is traded, and the triad takes at most one trade per day.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional

import pandas as pd

from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.signals import LONG, SHORT, Bar, Decision, update_tdo
from quarterly.timemodel import SessionTimes, clock, session_times, to_ny, trading_day

log = get_logger(__name__)


@dataclass
class AssetDay:
    """One symbol's state for one day (levels count from 00:00 NY)."""
    tdo: Optional[float] = None
    tdo_source: Optional[str] = None
    skip_reason: Optional[str] = None
    pre_midnight_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_done: bool = False
    took_high: bool = False
    took_low: bool = False
    took_high_ts: Optional[pd.Timestamp] = None
    took_low_ts: Optional[pd.Timestamp] = None
    win_high: Optional[float] = None
    win_low: Optional[float] = None
    last_high: Optional[float] = None          # latest bar, carried forward on minutes without one
    last_low: Optional[float] = None
    trades: int = 0

    def bear_leader(self) -> bool:
        return (self.took_high and not self.took_low and self.tdo is not None
                and self.last_high is not None and self.last_high > self.tdo)

    def bull_leader(self) -> bool:
        return (self.took_low and not self.took_high and self.tdo is not None
                and self.last_low is not None and self.last_low < self.tdo)


class TriadEngine:
    def __init__(self, cfg: QuarterlyConfig, verbose: bool = True):
        self.cfg = cfg
        self.tc = cfg.triad
        self.verbose = verbose
        self.leaders = {s: [p.partner for p in ps] for s, ps in cfg.partners.items()}
        self.day: Optional[date] = None
        self.times: Optional[SessionTimes] = None
        self.days: Dict[str, AssetDay] = {}
        self.last_ts: Optional[pd.Timestamp] = None
        self.leader_of: Dict[str, Optional[str]] = {}     # symbol -> leader behind today's signal

    # --------------------------------------------------------------- clock
    def _new_day(self, day: date) -> None:
        self.day = day
        self.times = session_times(day, self.cfg.session_start_hour)     # midnight / day_start only
        self.range_start, self.range_end = clock(day, self.tc.range_start), clock(day, self.tc.range_end)
        self.window_start, self.window_end = clock(day, self.tc.window_start), clock(day, self.tc.window_end)
        self.days = {s: AssetDay() for s in self.cfg.symbols}
        self.leader_of = {}
        log.info("new trading day", extra={"day": str(day), "strategy": "triad930",
                                           "window_start": str(self.window_start), "window_end": str(self.window_end)})

    def in_window(self, ts: pd.Timestamp) -> bool:
        return self.window_start <= ts < self.window_end

    @property
    def flatten_at(self) -> Optional[pd.Timestamp]:
        return clock(self.day, self.tc.flat_at) if self.day is not None else None

    def set_trades(self, symbol: str, n: int) -> None:
        if symbol in self.days:
            self.days[symbol].trades = n

    def record_trade(self, symbol: str) -> None:
        self.days[symbol].trades += 1

    # -------------------------------------------------------------- levels
    def _update(self, bar: Bar) -> None:
        d, tc = self.days[bar.symbol], self.tc
        update_tdo(d, bar, self.times, self.cfg)         # also notes the last close before 00:00
        if bar.ts < self.times.midnight:
            return                                       # 18:00-24:00 belongs to no other level of this strategy
        d.day_high = bar.high if d.day_high is None else max(d.day_high, bar.high)
        d.day_low = bar.low if d.day_low is None else min(d.day_low, bar.low)
        if self.range_start <= bar.ts < self.range_end:
            d.range_high = bar.high if d.range_high is None else max(d.range_high, bar.high)
            d.range_low = bar.low if d.range_low is None else min(d.range_low, bar.low)
        elif bar.ts >= self.range_end and d.range_high is not None and not d.range_done:
            d.range_done = True
            log.info("8am range", extra={"symbol": bar.symbol, "high": d.range_high, "low": d.range_low})
        if d.range_done and (self.in_window(bar.ts) or not tc.sweep_in_window):
            if bar.high > d.range_high and not d.took_high:
                d.took_high, d.took_high_ts = True, bar.ts
                log.info("took 8am high", extra={"symbol": bar.symbol, "bar": str(bar.ts), "high": bar.high,
                                                 "range_high": d.range_high})
            if bar.low < d.range_low and not d.took_low:
                d.took_low, d.took_low_ts = True, bar.ts
                log.info("took 8am low", extra={"symbol": bar.symbol, "bar": str(bar.ts), "low": bar.low,
                                                "range_low": d.range_low})
        if self.in_window(bar.ts):
            d.win_high = bar.high if d.win_high is None else max(d.win_high, bar.high)
            d.win_low = bar.low if d.win_low is None else min(d.win_low, bar.low)
        d.last_high, d.last_low = bar.high, bar.low

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
        for bar in bars.values():                        # every asset first: leaders are judged on this bar too
            self._update(bar)
        return self._pick([self._decide(sym, bar, has_position(sym)) for sym, bar in bars.items()])

    def _pick(self, decs: List[Decision]) -> List[Decision]:
        """pick: keep one entry of the minute, by stop distance as a percent of price."""
        live = [d for d in decs if d.action]
        if self.tc.pick == "all" or len(live) < 2:
            return decs
        pct = lambda d: d.risk_per_share / d.entry                    # noqa: E731
        best = (min if self.tc.pick == "tightest_stop" else max)(live, key=pct)
        for d in live:
            if d is not best:
                d.reasons.append(f"not picked: {best.symbol} has the {self.tc.pick.split('_')[0]} stop "
                                 f"({pct(best):.3%} vs {pct(d):.3%})")
                d.action = d.entry = d.stop = d.target = d.risk_per_share = None
                self.leader_of.pop(d.symbol, None)
        log.info("triad pick", extra={"picked": best.symbol, "rule": self.tc.pick,
                                      "candidates": {d.symbol: d.levels.get("close") for d in live}})
        return decs

    def _first(self, names: List[str], attr: str) -> Optional[str]:
        """The leader whose sweep came first."""
        return min(names, key=lambda n: getattr(self.days[n], attr)) if names else None

    def _decide(self, sym: str, bar: Bar, in_position: bool) -> Decision:
        d, tc = self.days[sym], self.tc
        bear = [n for n in self.leaders[sym] if self.days[n].bear_leader()]
        bull = [n for n in self.leaders[sym] if self.days[n].bull_leader()]
        dec = Decision(sym, bar.ts, None, in_window=self.in_window(bar.ts), smt_bear=bool(bear), smt_bull=bool(bull),
                       levels={"close": bar.close, "tdo": d.tdo, "range_high": d.range_high, "range_low": d.range_low,
                               "day_high": d.day_high, "day_low": d.day_low, "window_high": d.win_high,
                               "window_low": d.win_low})
        lag_sell = (not d.took_high and not d.took_low) if tc.strict_lag else not d.took_high
        lag_buy = (not d.took_high and not d.took_low) if tc.strict_lag else not d.took_low
        sell = tc.allow_shorts and bool(bear) and lag_sell and (not tc.need_premium or (d.tdo is not None and bar.close > d.tdo))
        buy = tc.allow_longs and bool(bull) and lag_buy and (not tc.need_premium or (d.tdo is not None and bar.close < d.tdo))
        if sell:
            dec.bias = "SELL"
        elif buy:
            dec.bias = "BUY"

        r: List[str] = []
        if not dec.in_window:
            r.append("outside entry window")
        if not d.range_done:
            r.append("8am range not complete (no 08:00-08:59 bar)" if d.range_high is None else "8am range not complete")
        if sell and buy:
            r.append(f"conflicting leaders: bearish {bear}, bullish {bull}")
        elif not (sell or buy):
            if bear or bull:
                r.append(f"leader {'bearish ' + ','.join(bear) if bear else 'bullish ' + ','.join(bull)} but "
                         f"{sym} does not lag (took 8am high={d.took_high}, low={d.took_low})"
                         + (" or fails the TDO premium filter" if tc.need_premium else ""))
            else:
                r.append("no leader: no partner took exactly one side of its 8am range beyond its TDO")
        if in_position:
            r.append("already in a position")
        if d.trades >= self.cfg.max_trades_per_symbol_per_day:
            r.append(f"trade limit reached ({d.trades}/{self.cfg.max_trades_per_symbol_per_day} today)")
        elif tc.pick != "all" and any(x.trades for x in self.days.values()):
            r.append("triad trade already taken today (pick: one trade per day)")

        notes: List[str] = []
        if not r and (sell or buy):
            entry = bar.close
            if sell:
                level = d.day_high if tc.stop_mode == "day" else d.win_high
                side, stop = SHORT, level + tc.stop_buffer
                risk = stop - entry
                target = entry - self.cfg.rr * risk
            else:
                level = d.day_low if tc.stop_mode == "day" else d.win_low
                side, stop = LONG, level - tc.stop_buffer
                risk = entry - stop
                target = entry + self.cfg.rr * risk
            if risk <= 0:
                r.append(f"risk per share {risk:.4f} <= 0")
            else:
                leader = self._first(bear, "took_high_ts") if sell else self._first(bull, "took_low_ts")
                self.leader_of[sym] = leader
                dec.action, dec.entry, dec.stop, dec.target, dec.risk_per_share = side, entry, stop, target, risk
                notes.append(f"{side}: {leader} took its 8am {'high above' if sell else 'low below'} its TDO, "
                             f"{sym} lags")
        dec.reasons = notes + r

        if self.verbose and (dec.in_window or dec.action):
            log.info("decision", extra=dec.as_dict())
        else:
            log.debug("decision", extra=dec.as_dict())
        return dec

    # ------------------------------------------------------------- reports
    def trade_context(self, dec: Decision) -> Dict:
        d = self.days[dec.symbol]
        leader = self.leader_of.get(dec.symbol)
        ld = self.days.get(leader) if leader else None
        short = dec.action == SHORT
        return dict(partner=leader, tdo=d.tdo, tdo_source=d.tdo_source, day_high=d.day_high, day_low=d.day_low,
                    range_high=d.range_high, range_low=d.range_low,
                    smt_ts=(ld.took_high_ts if short else ld.took_low_ts) if ld else None)

    def state_summary(self) -> Dict:
        return {"levels": {s: {"tdo": d.tdo, "tdo_source": d.tdo_source, "skip": d.skip_reason,
                               "range_high": d.range_high, "range_low": d.range_low, "took_high": d.took_high,
                               "took_low": d.took_low, "day_high": d.day_high, "day_low": d.day_low,
                               "trades": d.trades} for s, d in self.days.items()}}


def make_engine(cfg: QuarterlyConfig, verbose: bool = True):
    """The engine of the configured strategy (the others stay dormant), or of several together."""
    if len(cfg.strategies) > 1:
        from quarterly.multi import MultiEngine

        return MultiEngine(cfg, verbose=verbose)
    if cfg.strategy == "q3":
        from quarterly.signals import StrategyEngine

        return StrategyEngine(cfg, verbose=verbose)
    if cfg.strategy == "nyopen":
        from quarterly.nyopen import NyOpenEngine

        return NyOpenEngine(cfg, verbose=verbose)
    if cfg.strategy == "silverbullet":
        from quarterly.silverbullet import SilverBulletEngine

        return SilverBulletEngine(cfg, verbose=verbose)
    return TriadEngine(cfg, verbose=verbose)


def strategy_label(cfg: QuarterlyConfig) -> str:
    if len(cfg.strategies) > 1:
        names = {"q3": "Q3-of-Q3", "nyopen": "NY open v2", "silverbullet": "Silver bullet", "triad930": "Triad 9:30"}
        return " + ".join(names[s] for s in cfg.strategies)
    return {"q3": "Q3-of-Q3", "nyopen": "NY open v2", "silverbullet": "Silver bullet"}.get(cfg.strategy, "Triad 9:30 manipulation")


def chart_spans(cfg: QuarterlyConfig, day: date) -> "tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]":
    """(entry window, session) shown shaded / as the session zoom on the trade chart."""
    if cfg.strategy == "q3":
        st = session_times(day, cfg.session_start_hour)
        return (st.window_start, st.window_end), (st.session_start, st.session_end)
    if cfg.strategy == "nyopen":
        nc = cfg.nyopen
        first, last = clock(day, nc.entry_times[0]), clock(day, nc.entry_times[-1])
        return ((first - pd.Timedelta(minutes=1), last + pd.Timedelta(minutes=1)),
                (clock(day, nc.smt_from), clock(day, nc.flat_at or (16, 0))))
    if cfg.strategy == "silverbullet":
        sc = cfg.silverbullet
        first = clock(day, sc.sessions[0])
        last = clock(day, sc.sessions[-1]) + pd.Timedelta(minutes=max(b for _, b in sc.macros))
        return (first, last), (first - pd.Timedelta(hours=2), clock(day, sc.flat_at or (16, 0)))
    tc = cfg.triad
    return (clock(day, tc.window_start), clock(day, tc.window_end)), (clock(day, tc.range_start), clock(day, tc.flat_at))
