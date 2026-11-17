"""Settings for the Quarterly Theory strategy, from config/quarterly.yaml plus environment flags."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config import CONFIG_DIR, PROJECT_ROOT

SMT_FROM = ("session_start", "session_q3", "window_start")
TDO_FALLBACKS = ("last_close_before_midnight", "skip_day", "first_premarket_bar")
FEEDS = ("iex", "sip", "boats")
STRATEGIES = ("nyopen", "triad930", "q3", "silverbullet")   # nyopen: NY open v2; triad930: 9:30 manipulation;
                                                             # q3: Q3-of-Q3; silverbullet: silver bullet models
STOP_MODES = ("day", "window")
PICKS = ("tightest_stop", "widest_stop", "all")


def _truthy(value: Optional[str]) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _hhmm(value: str) -> "tuple[int, int]":
    h, m = str(value).split(":")
    return int(h), int(m)


@dataclass(frozen=True)
class Pair:
    symbol: str
    partner: str
    inverse: bool = False
    enabled: bool = True


@dataclass(frozen=True)
class TriadConfig:
    """Index triad 9:30 manipulation (strategy: triad930). Times are NY wall clock (hour, minute)."""
    range_start: "tuple[int, int]" = (8, 0)       # 8am range: bars opening 08:00 ... 08:59
    range_end: "tuple[int, int]" = (9, 0)
    window_start: "tuple[int, int]" = (9, 30)     # signals on bars opening 09:30 ... 09:32
    window_end: "tuple[int, int]" = (9, 33)
    sweep_in_window: bool = False                 # leader's sweep must happen inside the window
    strict_lag: bool = True                       # traded asset must have taken NEITHER side of its 8am range
    need_premium: bool = False                    # traded asset also above its TDO for sells / below for buys
    allow_shorts: bool = True
    allow_longs: bool = True
    stop_mode: str = "day"                        # day: day high/low since 00:00 | window: window high/low
    stop_buffer: float = 0.05                     # $ beyond the stop level (5 ticks of $0.01)
    flat_at: "tuple[int, int]" = (15, 55)         # close open trades from this bar on
    pick: str = "tightest_stop"                   # several signals: trade only the one with the tightest
                                                  # (or widest) stop as % of price, one triad trade per day;
                                                  # all: every signal, one trade per symbol per day

    def __post_init__(self) -> None:
        if self.stop_mode not in STOP_MODES:
            raise ValueError(f"triad stop_mode must be one of {STOP_MODES}, got {self.stop_mode!r}")
        if not (self.range_start < self.range_end <= self.window_start < self.window_end <= self.flat_at):
            raise ValueError("triad times must satisfy range_start < range_end <= window_start < window_end <= flat_at")
        if self.stop_buffer < 0:
            raise ValueError("triad stop_buffer must be >= 0")
        if self.pick not in PICKS:
            raise ValueError(f"triad pick must be one of {PICKS}, got {self.pick!r}")


@dataclass(frozen=True)
class NyOpenConfig:
    """NY open model (strategy: nyopen). Times are NY wall clock (hour, minute)."""
    entry_times: "tuple[tuple[int, int], ...]" = ((9, 30), (9, 45))
    smt_from: "tuple[int, int]" = (7, 30)          # SMT and the market structure shift count from here
    momentum_bars: int = 3                         # % change over this many 1-minute closes
    require_mss: bool = False
    stop_buffer: float = 0.0                       # $ beyond the day high/low
    min_rr: float = 0.0                            # minimum reward:risk (0 = off)
    target_rr: float = 0.0                         # >0: target = entry +/- target_rr x risk instead of the 07:30 level
    opt_out: bool = True                           # close when another index reaches its own open first
    # When another index counts as having taken out its own open (blocks an entry, triggers the opt out):
    #   beyond:       its latest bar is at/through its target open, whichever side it is on (the Pine v2 rule;
    #                 in effect all three indices must be on the same side of their opens to trade)
    #   crossed:      only an index on the same side that came back to its open, any time since smt_from
    #   after_entry:  only an index on the same side that came back to its open after the entry (opt out only)
    #   first_to_target: a race from the entry: no index blocks the entry; of the indices with a setup, trade the
    #                 one most likely to reach its target first (`pick`); opt out if any other index with a
    #                 setup at the entry (either side) reaches its own target first
    other_open_rule: str = "beyond"
    # first_to_target: how "most likely to reach its target first" is judged
    #   momentum:       strongest % move over momentum_bars in the trade direction
    #   nearest_target: smallest distance to the target, in % of price
    #   vol_units:      smallest distance to the target in average 1-minute ranges (last 15 bars)
    #   rw_probability: highest stop / (stop + target) distance: random-walk odds of target before stop
    pick: str = "momentum"
    # target: nearer_open = the nearer of the 00:00 / 07:30 opens (sells the higher, buys the lower);
    #         0730_candle = the 07:30 one-minute candle's low (sells) / high (buys)
    #         0830_candle = the 08:30 one-minute candle's low (sells) / high (buys)
    target: str = "nearer_open"
    # Judge an entry time on the first bar that closes at it even if an index has no bar then (its latest
    # price is used): entries at exactly 09:30. True: wait up to two minutes for every index to have a bar.
    wait_for_all_bars: bool = False
    flat_at: "Optional[tuple[int, int]]" = (15, 55)

    def __post_init__(self) -> None:
        if not self.entry_times or any(not ((9, 0) <= t <= (15, 0)) for t in self.entry_times):
            raise ValueError("nyopen entry_times must be NY times between 09:00 and 15:00")
        if self.smt_from not in ((0, 0), (7, 30)):
            raise ValueError("nyopen smt_from must be 00:00 or 07:30")
        if self.other_open_rule not in ("beyond", "crossed", "after_entry", "first_to_target"):
            raise ValueError("nyopen other_open_rule must be beyond, crossed, after_entry or first_to_target")
        if self.target not in ("nearer_open", "0730_candle", "0830_candle"):
            raise ValueError("nyopen target must be nearer_open, 0730_candle or 0830_candle")
        if self.pick not in ("momentum", "nearest_target", "vol_units", "rw_probability"):
            raise ValueError("nyopen pick must be momentum, nearest_target, vol_units or rw_probability")
        if self.momentum_bars < 1 or self.stop_buffer < 0 or self.min_rr < 0 or self.target_rr < 0:
            raise ValueError("nyopen momentum_bars >= 1, stop_buffer >= 0 and min_rr >= 0 are required")


@dataclass(frozen=True)
class SilverBulletConfig:
    """Silver bullet reversal (SBRM) and continuation (SBCM) models; see quarterly/silverbullet.py."""
    models: "tuple[str, ...]" = ("reversal", "continuation")
    # Silver bullet hours (NY): 03:00 London, 10:00 NY AM, 14:00 NY PM (C1 = the hour before; continuation
    # C1/C2 = the two hours before). 03:00 is backtest only on the free plan: overnight bars arrive 16 minutes
    # late, so its live signals are always stale and are logged, not traded.
    sessions: "tuple[tuple[int, int], ...]" = ((10, 0),)
    # Macro windows, minutes past the hour: xx:20-xx:40 and xx:50-xx:10 (the next hour's :10)
    macros: "tuple[tuple[int, int], ...]" = ((20, 40), (50, 70))
    require_smt: bool = True        # some but not all indices took the liquidity (C1 high/low)
    fvg_minutes: int = 5            # continuation PDA: the FVG timeframe inside C2 (5 or 15)
    stop_buffer: float = 0.0
    min_rr: float = 1.0                            # checked on the setup's own target (C1/C2 extreme)
    target_rr: float = 0.0                         # >0: order target = entry +/- target_rr x risk instead
    max_trades_per_day: int = 3
    max_trades_per_session: int = 1
    flat_at: "Optional[tuple[int, int]]" = (15, 55)

    def __post_init__(self) -> None:
        if self.max_trades_per_session < 1:
            raise ValueError("silverbullet max_trades_per_session must be >= 1")
        if not self.models or any(m not in ("reversal", "continuation") for m in self.models):
            raise ValueError("silverbullet models must be reversal and/or continuation")
        if not self.sessions or any(s not in ((3, 0), (10, 0), (14, 0)) for s in self.sessions):
            raise ValueError("silverbullet sessions must be among 03:00, 10:00 and 14:00")
        if any(not (0 <= a < b <= 70) for a, b in self.macros):
            raise ValueError("silverbullet macros are (start, end) minutes past the hour, 0..70")
        if self.fvg_minutes not in (5, 15) or self.stop_buffer < 0 or self.min_rr < 0 or self.target_rr < 0 \
                or self.max_trades_per_day < 1:
            raise ValueError("silverbullet fvg_minutes 5/15, stop_buffer >= 0, min_rr >= 0, max_trades_per_day >= 1")


@dataclass
class QuarterlyConfig:
    strategy: str = "triad930"
    # Strategies run together by one bot (first = `strategy`); empty means just `strategy`
    strategies: "tuple[str, ...]" = ()
    triad: TriadConfig = field(default_factory=TriadConfig)
    nyopen: NyOpenConfig = field(default_factory=NyOpenConfig)
    silverbullet: SilverBulletConfig = field(default_factory=SilverBulletConfig)
    session_start_hour: int = 7
    pairs: List[Pair] = field(default_factory=list)
    tdo_fallback: str = "skip_day"
    tdo_search_minutes: int = 30
    premarket_start: "tuple[int, int]" = (4, 0)
    smt_from: str = "session_q3"
    pivot_len: int = 2
    rr: float = 3.0
    risk_pct: float = 1.0
    stop_buffer: float = 0.0
    max_leverage: float = 4.0                        # position value cap, x equity (and never above buying power)
    max_trades_per_symbol_per_day: int = 1
    flat_at_session_end: bool = False
    time_in_force: str = "gtc"
    overnight_feed: str = "boats"
    day_feed: str = "iex"
    overnight_start: "tuple[int, int]" = (20, 0)
    overnight_end: "tuple[int, int]" = (4, 0)
    premarket_feed: Optional[str] = None             # used overnight_end-premarket_end; None: the day feed
    premarket_end: "tuple[int, int]" = (9, 30)
    backtest_day_feed: str = "iex"
    delayed_feeds: "tuple[str, ...]" = ("boats", "sip")   # free plan: no data from the last few minutes
    recent_delay_minutes: int = 16
    max_wick_pct: float = 0.75                       # clip bar highs/lows further than this % beyond open/close
    loop_delay_seconds: float = 5.0
    dry_run: bool = False
    live_trading: bool = False
    results_dir: Path = PROJECT_ROOT / "results" / "quarterly"

    def __post_init__(self) -> None:
        if self.strategy not in self.strategies:            # e.g. replace(cfg, strategy=...): run just that one
            self.strategies = (self.strategy,)
        self.strategies = tuple(self.strategies)
        if any(x not in STRATEGIES for x in self.strategies):
            raise ValueError(f"strategy / strategies must be among {STRATEGIES}, got {self.strategies!r}")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("strategies must not repeat")
        if self.session_start_hour not in (6, 7):
            raise ValueError(f"session_start_hour must be 6 or 7, got {self.session_start_hour}")
        if self.smt_from not in SMT_FROM:
            raise ValueError(f"smt_from must be one of {SMT_FROM}, got {self.smt_from!r}")
        if self.tdo_fallback not in TDO_FALLBACKS:
            raise ValueError(f"tdo_fallback must be one of {TDO_FALLBACKS}, got {self.tdo_fallback!r}")
        if self.premarket_feed is not None and self.premarket_feed not in FEEDS:
            raise ValueError(f"premarket_feed must be one of {FEEDS}, got {self.premarket_feed!r}")
        for name in ("overnight_feed", "day_feed", "backtest_day_feed"):
            if getattr(self, name) not in FEEDS:
                raise ValueError(f"{name} must be one of {FEEDS}, got {getattr(self, name)!r}")
        if self.pivot_len < 1 or self.rr <= 0 or self.risk_pct <= 0 or self.stop_buffer < 0:
            raise ValueError("pivot_len >= 1, rr > 0, risk_pct > 0 and stop_buffer >= 0 are required")
        if self.time_in_force not in ("day", "gtc"):
            raise ValueError("time_in_force must be day or gtc")

    @property
    def active_pairs(self) -> List[Pair]:
        return [p for p in self.pairs if p.enabled]

    @property
    def partners(self) -> Dict[str, List[Pair]]:
        """symbol -> its SMT partners, for every member of every enabled pair/group (each trades on its own signals)."""
        out: Dict[str, List[Pair]] = {}

        def link(a: str, b: str, inverse: bool) -> None:
            lst = out.setdefault(a, [])
            if all(p.partner != b for p in lst):
                lst.append(Pair(a, b, inverse, True))

        for p in self.active_pairs:
            link(p.symbol, p.partner, p.inverse)
            link(p.partner, p.symbol, p.inverse)
        return out

    @property
    def symbols(self) -> List[str]:
        return sorted(self.partners)


def load_config(path: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> QuarterlyConfig:
    env = os.environ if env is None else env
    raw: Dict[str, Any] = yaml.safe_load((path or CONFIG_DIR / "quarterly.yaml").read_text()) or {}
    data = raw.pop("data", {}) or {}
    pairs = [Pair(p["symbol"], p["partner"], bool(p.get("inverse", False)), bool(p.get("enabled", True)))
             for p in raw.pop("pairs", None) or []]
    # A group (e.g. the index triad) means every member is the SMT partner of every other member
    for g in raw.pop("groups", None) or []:
        members = list(g["symbols"])
        pairs += [Pair(a, b, False, bool(g.get("enabled", True)))
                  for i, a in enumerate(members) for b in members[i + 1:]]
    tr = raw.pop("triad930", None) or {}
    triad = TriadConfig(**{k: (_hhmm(v) if k in ("range_start", "range_end", "window_start", "window_end", "flat_at")
                                 else v) for k, v in tr.items()})
    ny = dict(raw.pop("nyopen", None) or {})
    if "entry_times" in ny:
        ny["entry_times"] = tuple(_hhmm(t) for t in ny["entry_times"])
    for k in ("smt_from", "flat_at"):
        if ny.get(k) is not None:
            ny[k] = _hhmm(ny[k])
    sb = dict(raw.pop("silverbullet", None) or {})
    if "models" in sb:
        sb["models"] = tuple(sb["models"])
    if "sessions" in sb:
        sb["sessions"] = tuple(_hhmm(t) for t in sb["sessions"])
    if "macros" in sb:
        sb["macros"] = tuple((int(a), int(b)) for a, b in sb["macros"])
    if sb.get("flat_at") is not None:
        sb["flat_at"] = _hhmm(sb["flat_at"])
    strategies = tuple(raw.get("strategies") or [raw.get("strategy", "triad930")])
    cfg = QuarterlyConfig(
        strategy=strategies[0],
        strategies=strategies,
        silverbullet=SilverBulletConfig(**sb),
        triad=triad,
        nyopen=NyOpenConfig(**ny),
        session_start_hour=int(raw.get("session_start_hour", 7)),
        pairs=pairs,
        tdo_fallback=raw.get("tdo_fallback", "skip_day"),
        tdo_search_minutes=int(raw.get("tdo_search_minutes", 30)),
        premarket_start=_hhmm(raw.get("premarket_start", "04:00")),
        smt_from=raw.get("smt_from", "session_q3"),
        pivot_len=int(raw.get("pivot_len", 2)),
        rr=float(raw.get("rr", 3.0)),
        risk_pct=float(raw.get("risk_pct", 1.0)),
        stop_buffer=float(raw.get("stop_buffer", 0.0)),
        max_leverage=float(raw.get("max_leverage", 4.0)),
        max_trades_per_symbol_per_day=int(raw.get("max_trades_per_symbol_per_day", 1)),
        flat_at_session_end=bool(raw.get("flat_at_session_end", False)),
        time_in_force=str(raw.get("time_in_force", "gtc")).lower(),
        overnight_feed=data.get("overnight_feed", "boats"),
        day_feed=data.get("day_feed", "iex"),
        overnight_start=_hhmm(data.get("overnight_start", "20:00")),
        overnight_end=_hhmm(data.get("overnight_end", "04:00")),
        premarket_feed=data.get("premarket_feed") or None,
        premarket_end=_hhmm(data.get("premarket_end", "09:30")),
        backtest_day_feed=data.get("backtest_day_feed", data.get("day_feed", "iex")),
        delayed_feeds=tuple(data.get("delayed_feeds", ["boats", "sip"])),
        recent_delay_minutes=int(data.get("recent_delay_minutes", 16)),
        max_wick_pct=float(data.get("max_wick_pct", 0.75)),
        loop_delay_seconds=float(raw.get("loop_delay_seconds", 5)),
        dry_run=bool(raw.get("dry_run", False)) or _truthy(env.get("DRY_RUN")),
        live_trading=_truthy(env.get("LIVE_TRADING")),
        results_dir=PROJECT_ROOT / raw.get("results_dir", "results/quarterly"),
    )
    return cfg
