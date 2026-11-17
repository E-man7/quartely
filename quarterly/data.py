"""1-minute bars from Alpaca's StockHistoricalDataClient, merged across feeds.

The overnight feed (`boats`) covers 20:00-04:00 NY, which is where the True
Day Open (00:00) lives; the optional premarket feed (`sip`) covers 04:00-09:30,
where IEX has almost no bars (the 8am range); the day feed (`iex`) covers the
rest. On the free plan boats and SIP are only served once 15 minutes old: for
the most recent minutes the day feed's bars stand in, and the delayed bars
replace them when they arrive (the live loop then replays the day). Bars are timestamped at their open, converted to NY time, and
only *closed* bars (open + 1 min <= now) are ever returned for live use.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from config import PROJECT_ROOT
from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.signals import Bar
from quarterly.timemodel import NY, clock, is_overnight, session_times, trading_day

log = get_logger(__name__)

COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume"]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "quarterly"


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def clip_spikes(df: pd.DataFrame, max_wick_pct: float) -> pd.DataFrame:
    """Clip bad prints: a high (low) more than max_wick_pct % above (below) the bar's open/close is cut back
    to that limit. A single stray trade otherwise sets a false 8am high / day high (e.g. QQQ 2026-10-22 08:09,
    high 724.66 on a 713.5-713.7 bar). Index ETF minute bars rarely have genuine wicks that long."""
    if df.empty or not max_wick_pct or max_wick_pct <= 0:
        return df
    k = max_wick_pct / 100.0
    top = df[["open", "close"]].max(axis=1)
    bottom = df[["open", "close"]].min(axis=1)
    hi_cap, lo_cap = top * (1 + k), bottom * (1 - k)
    bad = (df["high"] > hi_cap) | (df["low"] < lo_cap)
    if not bad.any():
        return df
    df = df.copy()
    for r in df[bad].itertuples(index=False):
        log.warning("bad print clipped", extra={"symbol": r.symbol, "bar": str(r.ts), "high": r.high, "low": r.low,
                                                "open": r.open, "close": r.close, "max_wick_pct": max_wick_pct})
    df["high"] = df["high"].where(df["high"] <= hi_cap, hi_cap)
    df["low"] = df["low"].where(df["low"] >= lo_cap, lo_cap)
    return df


class MinuteBars:
    def __init__(self, cfg: QuarterlyConfig, client=None, credentials: Optional[Dict] = None, sleep=time.sleep,
                 now_fn=None):
        self.cfg = cfg
        self.sleep = sleep
        self.now_fn = now_fn or (lambda: pd.Timestamp.now(tz=NY))
        if client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            keys = (credentials or {}).get("alpaca", {})
            client = StockHistoricalDataClient(keys.get("api_key"), keys.get("secret_key"))
        self.client = client
        self._unavailable: set = set()

    # ------------------------------------------------------------- fetching
    def _request(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp, feed: str) -> pd.DataFrame:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        if feed in self._unavailable:
            return empty_frame()
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame.Minute,
                               start=start.to_pydatetime(), end=end.to_pydatetime(), feed=DataFeed(feed))
        for attempt in range(4):
            try:
                df = self.client.get_stock_bars(req).df
                break
            except Exception as exc:
                msg = str(exc)
                if "invalid feed" in msg:
                    log.error("data feed not available on this account", extra={"feed": feed, "error": msg[:200]})
                    self._unavailable.add(feed)
                    return empty_frame()
                if "not permit" in msg:     # recent data on a restricted feed: try again on the next call
                    log.warning("recent data not permitted on this feed", extra={"feed": feed, "error": msg[:200]})
                    return empty_frame()
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                log.warning("bar request failed, retrying", extra={"feed": feed, "error": msg[:200], "wait_s": wait})
                self.sleep(wait)
        if df is None or df.empty:
            return empty_frame()
        df = df.reset_index()
        df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
        return df[COLUMNS].copy()

    def _capped(self, feed: str, end: pd.Timestamp) -> pd.Timestamp:
        """The free plan refuses the most recent minutes of the boats and SIP feeds; stop short of them."""
        if feed in self.cfg.delayed_feeds:
            return min(end, self.now_fn() - pd.Timedelta(minutes=self.cfg.recent_delay_minutes))
        return end

    def fetch(self, symbols: Sequence[str], start, end, day_feed: Optional[str] = None) -> pd.DataFrame:
        """Bars with open time in [start, end): overnight feed overnight, day feed otherwise."""
        start, end = pd.Timestamp(start).tz_convert(NY), pd.Timestamp(end).tz_convert(NY)
        day_feed = day_feed or self.cfg.day_feed
        on, off = self.cfg.overnight_start, self.cfg.overnight_end
        def req(feed: str) -> pd.DataFrame:
            stop = self._capped(feed, end)
            return self._request(symbols, start, stop, feed) if stop > start else empty_frame()

        if day_feed == self.cfg.overnight_feed:
            df = req(day_feed)
        else:
            night = req(self.cfg.overnight_feed)
            day = req(day_feed)
            night = night[night["ts"].map(lambda t: is_overnight(t, on, off))] if not night.empty else night
            day = day[~day["ts"].map(lambda t: is_overnight(t, on, off))] if not day.empty else day
            pre = empty_frame()
            pf = self.cfg.premarket_feed
            if pf and pf != day_feed:
                pre = req(pf)
                pre = pre[pre["ts"].map(lambda t: self.in_premarket(t))] if not pre.empty else pre
            # later frames win on the same (symbol, minute): premarket feed over the day feed
            frames = [f for f in (night, day, pre) if not f.empty]
            df = pd.concat(frames) if frames else empty_frame()
        if df.empty:
            return empty_frame()
        df = df[(df["ts"] >= start) & (df["ts"] < end)]
        df = df.drop_duplicates(["symbol", "ts"], keep="last").sort_values(["ts", "symbol"]).reset_index(drop=True)
        return clip_spikes(df, self.cfg.max_wick_pct)

    def in_premarket(self, ts) -> bool:
        t = pd.Timestamp(ts).tz_convert(NY)
        hm = (t.hour, t.minute)
        return self.cfg.overnight_end <= hm < self.cfg.premarket_end

    @staticmethod
    def closed_only(df: pd.DataFrame, now) -> pd.DataFrame:
        """Drop any bar that has not closed yet (open + 1 minute > now)."""
        if df.empty:
            return df
        return df[df["ts"] + pd.Timedelta(minutes=1) <= pd.Timestamp(now).tz_convert(NY)]

    # -------------------------------------------------------------- history
    def history(self, symbols: Sequence[str], start: date, end: date, day_feed: Optional[str] = None,
                use_cache: bool = True) -> pd.DataFrame:
        """All bars of the trading days start..end (each from 18:00 the day before), cached per day."""
        day_feed = day_feed or self.cfg.backtest_day_feed
        frames: List[pd.DataFrame] = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                frames.append(self._day(symbols, d, day_feed, use_cache))
            d += timedelta(days=1)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames).sort_values(["ts", "symbol"]).reset_index(drop=True) if frames else empty_frame()

    def _day(self, symbols: Sequence[str], day: date, day_feed: str, use_cache: bool) -> pd.DataFrame:
        # One file per day and symbol set: a symbol with no bars that day is still "cached"
        feeds = [self.cfg.overnight_feed] + ([self.cfg.premarket_feed] if self.cfg.premarket_feed
                                             and self.cfg.premarket_feed != day_feed else []) + [day_feed]
        path = CACHE_DIR / "+".join(feeds) / "-".join(sorted(symbols)) / f"{day}.csv"
        if use_cache and path.exists():
            df = pd.read_csv(path)
            if df.empty:
                return empty_frame()
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
            return clip_spikes(df, self.cfg.max_wick_pct)      # files cached before the filter existed
        t = session_times(day)
        df = self.fetch(symbols, t.day_start, t.day_start + pd.Timedelta(days=1), day_feed)
        if use_cache and pd.Timestamp.now(tz=NY) > t.day_start + pd.Timedelta(days=1, minutes=20):   # day complete
            path.parent.mkdir(parents=True, exist_ok=True)
            (df if not df.empty else empty_frame()).to_csv(path, index=False)
        return df


def iter_minutes(df: pd.DataFrame) -> Iterator[Tuple[pd.Timestamp, Dict[str, Bar]]]:
    """Group bars by minute, in time order: (minute, {symbol: Bar})."""
    if df.empty:
        return
    for ts, group in df.groupby("ts", sort=True):
        yield ts, {r.symbol: Bar(r.symbol, ts, float(r.open), float(r.high), float(r.low), float(r.close),
                                 float(r.volume)) for r in group.itertuples(index=False)}


def coverage(df: pd.DataFrame, cfg: QuarterlyConfig) -> pd.DataFrame:
    """Per symbol and day: bars inside the entry window, bars in the triad's 8am range and whether a
    midnight TDO bar exists."""
    from quarterly.triad import chart_spans

    if df.empty:
        return pd.DataFrame()
    rows = []
    for (sym, day), g in df.assign(day=df["ts"].map(trading_day)).groupby(["symbol", "day"]):
        t = session_times(day, cfg.session_start_hour)
        (w0, w1), _ = chart_spans(cfg, day)
        r0, r1 = clock(day, cfg.triad.range_start), clock(day, cfg.triad.range_end)
        rows.append({"symbol": sym, "day": day,
                     "window_bars": int(((g["ts"] >= w0) & (g["ts"] < w1)).sum()),
                     "range_bars": int(((g["ts"] >= r0) & (g["ts"] < r1)).sum()),
                     "midnight_bar": bool(((g["ts"] >= t.midnight) &
                                           (g["ts"] < t.midnight + pd.Timedelta(minutes=cfg.tdo_search_minutes))).any())})
    return pd.DataFrame(rows)
