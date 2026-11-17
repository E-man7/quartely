"""Signal engine for the silver bullet models (strategy: silverbullet) on QQQ / DIA / SPY.

Times are America/New_York, so daylight saving is handled by the clock. A
silver bullet hour S is 03:00-04:00 (London; backtest only on the free data
plan), 10:00-11:00 (NY AM) or 14:00-15:00 (NY PM). Hour candles are built
from the 1-minute bars.

Entries only happen in the macro windows: S+20..S+40 and S+50..S+70 minutes
(10:20-10:40 and 10:50-11:10). The entry trigger is a 1-minute CISD (change
in state of delivery): for a short, a close below the open of the run of
up-close candles that delivered the high (for a long, a close above the open
of the down-close run that delivered the low). A CISD outside a macro window
is spent; a new extreme arms a new one.

Reversal model (SBRM), C1 = the hour before S (09:00):
* during S, price runs C1's high (buy-side liquidity) and a CISD follows -> short;
* SMT: some but not all indices have taken their own C1 high since S;
* stop above the sweep high, target C1's low (sell-side liquidity).
Longs mirror it at C1's low.

Continuation model (SBCM), C1 = S-2h (08:00), C2 = S-1h (09:00):
* H1 bias: C2 takes C1's high and closes back below it -> bearish
  (SMT: not every index's C2 took its C1 high);
* PDA: the last bearish fair value gap on `fvg_minutes` candles inside C2;
* during S, price retraces into the FVG, does not close above its top, and a
  CISD follows -> short;
* stop above the retrace high, target the lower of the C1 / C2 lows.
Longs mirror it.

At most `max_trades_per_session` trades per silver bullet hour and
`max_trades_per_day` a day across the three indices; if
several setups trigger on the same minute, the best reward:risk is taken.
Not modelled: discretionary higher-timeframe key levels and narrative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional

import pandas as pd

from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.signals import LONG, SHORT, Bar, Decision
from quarterly.timemodel import clock, to_ny, trading_day

log = get_logger(__name__)

HOUR = pd.Timedelta(hours=1)


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float

    def add(self, bar: Bar) -> None:
        self.high, self.low, self.close = max(self.high, bar.high), min(self.low, bar.low), bar.close


@dataclass
class Leg:
    """One armed setup: a model and a side on one index during one silver bullet hour."""
    model: str                         # reversal | continuation
    side: str                          # SHORT | LONG
    liquidity: float                   # reversal: C1 high/low to run; continuation: FVG edge to reach
    target: float
    limit: Optional[float] = None      # continuation: FVG top (shorts) / bottom (longs); a close beyond kills it
    extreme: Optional[float] = None    # sweep / retrace high (shorts) or low (longs) since the liquidity was taken
    cisd: Optional[float] = None       # open of the run that delivered the extreme
    dead: Optional[str] = None


@dataclass
class SbDay:
    hours: Dict[pd.Timestamp, Candle] = field(default_factory=dict)
    minutes: Dict[pd.Timestamp, Bar] = field(default_factory=dict)       # only the hours a session needs
    up_open: Optional[float] = None    # open of the latest run of up-close candles
    down_open: Optional[float] = None
    prev_dir: int = 0
    last_close: Optional[float] = None
    last_ts: Optional[pd.Timestamp] = None
    sess_high: Optional[float] = None  # since the silver bullet hour began
    sess_low: Optional[float] = None
    legs: List[Leg] = field(default_factory=list)
    trades: int = 0


def hour_of(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.replace(minute=0, second=0, microsecond=0, nanosecond=0)


def fvgs(candles: List[Candle]) -> "tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]":
    """(last bearish gap, last bullish gap) as (bottom, top) over consecutive candles a, b, c."""
    bear = bull = None
    for a, _, c in zip(candles, candles[1:], candles[2:]):
        if a.low > c.high:
            bear = (c.high, a.low)
        if a.high < c.low:
            bull = (a.high, c.low)
    return bear, bull


class SilverBulletEngine:
    def __init__(self, cfg: QuarterlyConfig, verbose: bool = True):
        self.cfg = cfg
        self.sc = cfg.silverbullet
        self.verbose = verbose
        self.symbols = cfg.symbols
        self.day: Optional[date] = None
        self.days: Dict[str, SbDay] = {}
        self.last_ts: Optional[pd.Timestamp] = None
        self.session: Optional[pd.Timestamp] = None        # start of the active silver bullet hour
        self.session_trades = 0
        self.context: Dict[str, Dict] = {}

    # --------------------------------------------------------------- clock
    def _new_day(self, day: date) -> None:
        self.day = day
        self.starts = sorted(clock(day, s) for s in self.sc.sessions)
        self.days = {s: SbDay() for s in self.symbols}
        self.session, self.context = None, {}
        log.info("new trading day", extra={"day": str(day), "strategy": "silverbullet",
                                           "sessions": [str(t) for t in self.starts], "models": list(self.sc.models)})

    @property
    def flatten_at(self) -> Optional[pd.Timestamp]:
        return clock(self.day, self.sc.flat_at) if self.day is not None and self.sc.flat_at else None

    def set_trades(self, symbol: str, n: int) -> None:
        if symbol in self.days:
            self.days[symbol].trades = n

    def record_trade(self, symbol: str) -> None:
        self.days[symbol].trades += 1
        self.session_trades += 1

    def room(self) -> int:
        """Trades still allowed now: within the day's and this silver bullet hour's limits."""
        day_left = self.sc.max_trades_per_day - sum(x.trades for x in self.days.values())
        return max(0, min(day_left, self.sc.max_trades_per_session - self.session_trades))

    def _session_of(self, ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        end = max(b for _, b in self.sc.macros)
        for s in self.starts:
            if s <= ts < s + pd.Timedelta(minutes=end):
                return s
        return None

    def in_macro(self, ts: pd.Timestamp) -> bool:
        """The bar closing at ts + 1 minute closes inside a macro window."""
        s = self._session_of(ts)
        if s is None:
            return False
        close_min = (ts + pd.Timedelta(minutes=1) - s).total_seconds() / 60
        return any(a < close_min <= b for a, b in self.sc.macros)

    # ------------------------------------------------------------- arming
    def _arm(self, s: pd.Timestamp) -> None:
        """Start of a silver bullet hour: read C1 / C2 and the C2 FVG, arm the legs of every index."""
        self.session, self.session_trades = s, 0
        for sym, d in self.days.items():
            d.legs, d.sess_high, d.sess_low = [], None, None
            c1r, c1c, c2 = d.hours.get(s - HOUR), d.hours.get(s - 2 * HOUR), d.hours.get(s - HOUR)
            if "reversal" in self.sc.models and c1r is not None:
                d.legs += [Leg("reversal", SHORT, c1r.high, c1r.low), Leg("reversal", LONG, c1r.low, c1r.high)]
            if "continuation" in self.sc.models and c1c is not None and c2 is not None:
                bear, bull = fvgs(self._c2_candles(d, s - HOUR))
                if c2.high > c1c.high and c2.close < c1c.high and bear is not None:
                    d.legs.append(Leg("continuation", SHORT, bear[0], min(c1c.low, c2.low), limit=bear[1]))
                if c2.low < c1c.low and c2.close > c1c.low and bull is not None:
                    d.legs.append(Leg("continuation", LONG, bull[1], max(c1c.high, c2.high), limit=bull[0]))
            (log.info if self.verbose else log.debug)("silver bullet armed", extra={
                "symbol": sym, "session": str(s),
                "legs": [f"{x.model} {x.side} liq {x.liquidity:.2f} tgt {x.target:.2f}"
                         + (f" fvg-limit {x.limit:.2f}" if x.limit is not None else "") for x in d.legs]})
        for d in self.days.values():                    # minutes are only needed to build the C2 FVG
            d.minutes.clear()

    def _c2_candles(self, d: SbDay, c2_start: pd.Timestamp) -> List[Candle]:
        n = self.sc.fvg_minutes
        out: Dict[pd.Timestamp, Candle] = {}
        for ts in sorted(t for t in d.minutes if c2_start <= t < c2_start + HOUR):
            b = d.minutes[ts]
            k = c2_start + pd.Timedelta(minutes=(ts - c2_start).total_seconds() // 60 // n * n)
            if k in out:
                out[k].add(b)
            else:
                out[k] = Candle(b.open, b.high, b.low, b.close)
        return [out[k] for k in sorted(out)]

    # ---------------------------------------------------------------- step
    def _update(self, bar: Bar) -> None:
        d = self.days[bar.symbol]
        h = hour_of(bar.ts)
        if h in d.hours:
            d.hours[h].add(bar)
        else:
            d.hours[h] = Candle(bar.open, bar.high, bar.low, bar.close)
        if any(s - HOUR <= bar.ts < s for s in self.starts):
            d.minutes[bar.ts] = bar
        direction = 1 if bar.close > bar.open else -1 if bar.close < bar.open else 0
        if direction > 0:
            d.up_open = bar.open if d.prev_dir <= 0 or d.up_open is None else d.up_open
        elif direction < 0:
            d.down_open = bar.open if d.prev_dir >= 0 or d.down_open is None else d.down_open
        d.prev_dir = direction
        d.last_close, d.last_ts = bar.close, bar.ts

    def on_minute(self, ts, bars: Dict[str, Bar], has_position: Callable[[str], bool] = lambda s: False
                  ) -> List[Decision]:
        ts = to_ny(ts)
        if self.last_ts is not None and ts <= self.last_ts:
            return []
        self.last_ts = ts
        day = trading_day(ts)
        if day != self.day:
            self._new_day(day)
        s = self._session_of(ts)
        if s is not None and s != self.session:
            self._arm(s)
        bars = {k: b for k, b in bars.items() if k in self.days}
        for b in bars.values():
            self._update(b)
        if s is None:
            return []
        for b in bars.values():
            d = self.days[b.symbol]
            d.sess_high = b.high if d.sess_high is None else max(d.sess_high, b.high)
            d.sess_low = b.low if d.sess_low is None else min(d.sess_low, b.low)
        decs = []
        for sym, b in bars.items():
            for leg in self.days[sym].legs:
                dec = self._step_leg(sym, leg, b, ts, s, has_position(sym))
                if dec is not None:
                    decs.append(dec)
        return self._pick(decs)

    def _step_leg(self, sym: str, leg: Leg, b: Bar, ts: pd.Timestamp, s: pd.Timestamp, in_position: bool
                  ) -> Optional[Decision]:
        if leg.dead:
            return None
        d, short = self.days[sym], leg.side == SHORT
        if leg.extreme is None and (b.low <= leg.target if short else b.high >= leg.target):
            leg.dead = "target reached before the setup"
            return None
        if leg.limit is not None and (b.close > leg.limit if short else b.close < leg.limit):
            leg.dead = "closed through the FVG: PDA not respected"
            return None
        if leg.model == "continuation":                 # retraced into the FVG
            took = b.high >= leg.liquidity if short else b.low <= leg.liquidity
        else:                                           # ran C1's high / low
            took = b.high > leg.liquidity if short else b.low < leg.liquidity
        if took and (leg.extreme is None or (b.high >= leg.extreme if short else b.low <= leg.extreme)):
            leg.extreme = b.high if short else b.low
            leg.cisd = d.up_open if short else d.down_open
        if leg.extreme is None or leg.cisd is None:
            return None
        if not (b.close < leg.cisd if short else b.close > leg.cisd):
            return None
        level, leg.cisd = leg.cisd, None                # spent; a new extreme arms a new one
        return self._decide(sym, leg, b, ts, s, level, in_position)

    def smt(self, leg: Leg, s: pd.Timestamp) -> "tuple[bool, List[str]]":
        """Some but not all indices took the liquidity: (ok, the indices that took it)."""
        short = leg.side == SHORT
        took = []
        for sym, d in self.days.items():
            if leg.model == "reversal":
                c1 = d.hours.get(s - HOUR)
                hit = c1 is not None and (d.sess_high is not None and d.sess_high > c1.high if short
                                          else d.sess_low is not None and d.sess_low < c1.low)
            else:
                c1, c2 = d.hours.get(s - 2 * HOUR), d.hours.get(s - HOUR)
                hit = c1 is not None and c2 is not None and (c2.high > c1.high if short else c2.low < c1.low)
            if hit:
                took.append(sym)
        return 0 < len(took) < len(self.days), took

    def _decide(self, sym: str, leg: Leg, b: Bar, ts: pd.Timestamp, s: pd.Timestamp, level: float,
                in_position: bool) -> Decision:
        sc, short = self.sc, leg.side == SHORT
        smt_ok, took = self.smt(leg, s)
        dec = Decision(sym, ts, None, bias="SELL" if short else "BUY", in_window=self.in_macro(ts),
                       smt_bear=short and smt_ok, smt_bull=(not short) and smt_ok,
                       levels={"close": b.close, "cisd": level, "liquidity": leg.liquidity, "extreme": leg.extreme,
                               "target": leg.target, "fvg_limit": leg.limit})
        entry = b.close
        stop = leg.extreme + sc.stop_buffer if short else leg.extreme - sc.stop_buffer
        risk, reward = (stop - entry, entry - leg.target) if short else (entry - stop, leg.target - entry)
        r: List[str] = []
        if not dec.in_window:
            r.append("CISD outside the macro windows")
        if sc.require_smt and not smt_ok:
            r.append(f"no SMT: {'all' if took else 'no'} indices took the liquidity ({', '.join(took) or 'none'})")
        if risk <= 0 or reward <= 0:
            r.append(f"risk {risk:.4f} / reward {reward:.4f} not both positive")
        elif reward / risk < sc.min_rr:
            r.append(f"reward:risk {reward / risk:.2f} below the minimum {sc.min_rr}")
        if in_position:
            r.append("already in a position")
        if self.room() <= 0:
            r.append("trade limit reached for this silver bullet hour / day")
        if not r:
            target = leg.target
            if sc.target_rr:   # fixed reward:risk target in place of the C1/C2 extreme (the setup still needs min_rr)
                target = round(entry - sc.target_rr * risk if short else entry + sc.target_rr * risk, 4)
                reward = sc.target_rr * risk
            dec.action, dec.entry, dec.stop, dec.target, dec.risk_per_share = leg.side, entry, stop, target, risk
            self.context[sym] = {"model": leg.model, "rr": reward / risk, "ts": ts, "smt_by": took,
                                 "liquidity": leg.liquidity}
            dec.reasons = [f"{leg.model} {leg.side} at {ts + pd.Timedelta(minutes=1):%H:%M}: CISD through {level:.2f} "
                           f"after taking {leg.liquidity:.2f}, SMT by {', '.join(took)}, target {target:.2f} "
                           f"(reward:risk {reward / risk:.2f})"]
        else:
            dec.reasons = [f"{leg.model} {leg.side} CISD at {ts + pd.Timedelta(minutes=1):%H:%M} not taken"] + r
        (log.info if self.verbose else log.debug)("decision", extra={**dec.as_dict(), "model": leg.model})
        return dec

    def _pick(self, decs: List[Decision]) -> List[Decision]:
        """Several setups on one minute: the best reward:risk, within the day's trade allowance."""
        rr = lambda d: abs(d.target - d.entry) / d.risk_per_share          # noqa: E731
        room = self.room()
        taken: set = set()
        for d in sorted((d for d in decs if d.action), key=rr, reverse=True):
            if len(taken) < room and d.symbol not in taken:
                taken.add(d.symbol)
                self.context[d.symbol]["rr"] = rr(d)
                continue
            d.reasons.append("not picked: another setup on this minute has the better reward:risk")
            d.action = d.entry = d.stop = d.target = d.risk_per_share = None
        return decs

    # ------------------------------------------------------------- reports
    def trade_context(self, dec: Decision) -> Dict:
        ctx = self.context.get(dec.symbol, {})
        c1 = self.days[dec.symbol].hours.get(self.session - HOUR) if self.session is not None else None
        return dict(model=ctx.get("model"), range_high=c1.high if c1 else None, range_low=c1.low if c1 else None,
                    smt_ts=ctx.get("ts"))

    def state_summary(self) -> Dict:
        return {"levels": {s: {"session": str(self.session) if self.session is not None else None,
                               "legs": [f"{x.model} {x.side}{' (' + x.dead + ')' if x.dead else ''}" for x in d.legs],
                               "trades": d.trades} for s, d in self.days.items()}}
