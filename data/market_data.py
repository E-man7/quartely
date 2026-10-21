"""Market data from Alpaca, yfinance or local CSV: history, latest bar/quote/snapshot, live streams.

Gaps: weekends and holidays simply have no bars. A requested range that falls
entirely inside a market closure returns an empty frame instead of raising;
unusually long gaps between sessions (halts, missing data) are logged, never
filled with invented prices.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from config import PROJECT_ROOT
from monitoring.logger import get_logger

log = get_logger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]
MAX_SESSION_GAP_DAYS = 4          # longest normal closure: a holiday next to a weekend
_YF_INTERVALS = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "30Min": "30m", "1Hour": "60m", "1Day": "1d",
                 "1Week": "1wk"}


def parse_timeframe(tf: str):
    """'15Min' / '1Hour' / '1Day' / '1Week' -> alpaca TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    units = {"min": TimeFrameUnit.Minute, "hour": TimeFrameUnit.Hour, "day": TimeFrameUnit.Day,
             "week": TimeFrameUnit.Week, "month": TimeFrameUnit.Month}
    digits = "".join(c for c in tf if c.isdigit()) or "1"
    unit = tf[len(digits):].lower() if tf[:len(digits)] == digits else tf.lower()
    if unit not in units:
        raise ValueError(f"Unsupported timeframe {tf!r}; use e.g. 1Min, 15Min, 1Hour, 1Day")
    return TimeFrame(int(digits), units[unit])


def find_gaps(df: pd.DataFrame, max_days: int = MAX_SESSION_GAP_DAYS) -> pd.DataFrame:
    """Consecutive daily bars further apart than a normal market closure."""
    if len(df) < 2:
        return pd.DataFrame(columns=["after", "before", "days"])
    idx = pd.DatetimeIndex(df.index)
    delta = (idx[1:] - idx[:-1]).days
    mask = delta > max_days
    return pd.DataFrame({"after": idx[:-1][mask], "before": idx[1:][mask], "days": delta[mask]})


class MarketData:
    def __init__(self, settings: Dict, credentials: Optional[Dict] = None):
        self.cfg = settings.get("data", {})
        self.source = self.cfg.get("source", "alpaca")
        self.cache_dir = PROJECT_ROOT / self.cfg.get("csv_dir", "data/cache")
        self.credentials = credentials or {}
        self._alpaca = None
        self.quote_max_age = self.cfg.get("quote_max_age_seconds", 5)
        self._quotes: Dict[str, Tuple[float, float, pd.Timestamp]] = {}
        self._subscriptions: List[Tuple[str, Any, List[str]]] = []
        self._supervisor = None
        self._stream_lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def history(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Daily OHLCV bars, index = tz-naive dates, columns = OHLCV."""
        use_cache = self.cfg.get("cache", True) if use_cache is None else use_cache
        if start is None:
            years = self.cfg.get("history_years", 15)
            start = (datetime.now() - timedelta(days=int(365.25 * years))).strftime("%Y-%m-%d")

        cache_file = self.cache_dir / f"{symbol}.csv"
        if self.source == "csv":
            df = self._load_csv(cache_file)
        else:
            fetch = self._fetch_alpaca if self.source == "alpaca" else self._fetch_yfinance
            try:
                df = fetch(symbol, start, end)
                if df.empty:
                    # yfinance reports failures by returning nothing; never cache that over good data
                    raise ValueError(f"empty download for {symbol}")
                if use_cache:
                    self._save_csv(df, cache_file)
            except Exception as exc:
                if use_cache and cache_file.exists():
                    log.warning("fetch failed, using cache", extra={"symbol": symbol, "error": str(exc)})
                    df = self._load_csv(cache_file)
                else:
                    raise

        df = df.loc[start:end] if end else df.loc[start:]
        if df.empty:
            log.info("no bars in range (market closed?)", extra={"symbol": symbol, "start": start, "end": end})
            return df
        gaps = find_gaps(df)
        if not gaps.empty:
            log.warning("unusual gaps between sessions", extra={
                "symbol": symbol, "count": len(gaps),
                "longest": f"{gaps['after'].iloc[gaps['days'].argmax()].date()} +{int(gaps['days'].max())}d"})
        return df

    def get_historical_bars(self, symbol: str, timeframe: str = "1Day", start: Optional[str] = None,
                            end: Optional[str] = None) -> pd.DataFrame:
        """OHLCV bars at any timeframe. Daily bars use `history` (cached); intraday bars are
        timestamped in US/Eastern and never cached."""
        if str(parse_timeframe(timeframe)) == "1Day":
            return self.history(symbol, start=start, end=end)
        start = start or (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        if self.source == "alpaca":
            df = self._fetch_alpaca(symbol, start, end, timeframe)
        elif self.source == "yfinance":
            import yfinance as yf

            if timeframe not in _YF_INTERVALS:
                raise ValueError(f"yfinance does not support timeframe {timeframe!r}")
            raw = yf.download(symbol, start=start, end=end, interval=_YF_INTERVALS[timeframe], auto_adjust=True,
                              progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = self._normalize(raw, intraday=True)
        else:
            raise ValueError("Intraday bars need data.source alpaca or yfinance")
        if df.empty:
            log.info("no intraday bars in range (market closed?)", extra={"symbol": symbol, "start": start})
        return df

    # ----------------------------------------------------------- latest data
    def get_latest_bar(self, symbol: str) -> Dict[str, Any]:
        """Most recent bar with its age; on weekends/holidays this is the last session's bar."""
        if self.source == "alpaca":
            from alpaca.data.requests import StockLatestBarRequest

            b = self._alpaca_client().get_stock_latest_bar(
                StockLatestBarRequest(symbol_or_symbols=symbol, feed=self._live_feed()))[symbol]
            bar = {"timestamp": pd.Timestamp(b.timestamp), "open": float(b.open), "high": float(b.high),
                   "low": float(b.low), "close": float(b.close), "volume": float(b.volume)}
        else:
            df = self.history(symbol, use_cache=False)
            last = df.iloc[-1]
            bar = {"timestamp": pd.Timestamp(df.index[-1]).tz_localize("America/New_York"),
                   **{c: float(last[c]) for c in OHLCV if c in last}}
        bar["symbol"] = symbol
        bar["age_seconds"] = (pd.Timestamp.now(tz="UTC") - bar["timestamp"].tz_convert("UTC")).total_seconds()
        return bar

    def get_latest_quote(self, symbol: str) -> Optional[Tuple[float, float]]:
        """(bid, ask): a fresh streamed quote if subscribed, else Alpaca REST; None for other sources."""
        cached = self._quotes.get(symbol)
        if cached and (pd.Timestamp.now(tz="UTC") - cached[2]).total_seconds() <= self.quote_max_age:
            return cached[0], cached[1]
        if self.source != "alpaca":
            return None
        from alpaca.data.requests import StockLatestQuoteRequest

        quote = self._alpaca_client().get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._live_feed()))[symbol]
        return float(quote.bid_price), float(quote.ask_price)

    latest_quote = get_latest_quote

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Latest trade, quote, minute bar, today's and the previous daily bar."""
        if self.source != "alpaca":
            df = self.history(symbol, use_cache=False)
            prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            return {"symbol": symbol, "latest_trade": float(df["close"].iloc[-1]), "latest_quote": None,
                    "minute_bar": None, "daily_bar": df.iloc[-1].to_dict(), "previous_daily_bar": prev.to_dict()}
        from alpaca.data.requests import StockSnapshotRequest

        snap = self._alpaca_client().get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=symbol, feed=self._live_feed()))[symbol]

        def bar(b):
            return None if b is None else {"timestamp": b.timestamp, "open": b.open, "high": b.high, "low": b.low,
                                           "close": b.close, "volume": b.volume}

        q = snap.latest_quote
        return {"symbol": symbol,
                "latest_trade": float(snap.latest_trade.price) if snap.latest_trade else None,
                "latest_quote": (float(q.bid_price), float(q.ask_price)) if q else None,
                "minute_bar": bar(snap.minute_bar), "daily_bar": bar(snap.daily_bar),
                "previous_daily_bar": bar(snap.previous_daily_bar)}

    # --------------------------------------------------------------- streams
    def subscribe_bars(self, symbols: List[str], timeframe: str, callback: Callable[[Dict[str, Any]], None]):
        """Live bars: '1Min' (minute bars) or '1Day' (daily bars). callback gets a dict per bar."""
        kind = {"1min": "bars", "1day": "daily_bars"}.get(timeframe.lower())
        if kind is None:
            raise ValueError("Streaming supports 1Min and 1Day bars")

        async def handler(b):
            callback({"symbol": b.symbol, "timestamp": pd.Timestamp(b.timestamp), "open": float(b.open),
                      "high": float(b.high), "low": float(b.low), "close": float(b.close), "volume": float(b.volume)})

        return self._subscribe(kind, handler, symbols)

    def subscribe_quotes(self, symbols: List[str], callback: Optional[Callable[[str, float, float, Any], None]] = None):
        """Live quotes; also keeps the latest (bid, ask) per symbol for get_latest_quote / spread checks."""

        async def handler(q):
            ts = pd.Timestamp(q.timestamp)
            self._quotes[q.symbol] = (float(q.bid_price), float(q.ask_price), ts.tz_convert("UTC") if ts.tz else ts.tz_localize("UTC"))
            if callback:
                callback(q.symbol, float(q.bid_price), float(q.ask_price), ts)

        return self._subscribe("quotes", handler, symbols)

    def _subscribe(self, kind: str, handler, symbols: List[str]):
        """All subscriptions share one websocket (Alpaca allows one data connection per account)."""
        if self.source != "alpaca":
            raise ValueError("Live streams need data.source: alpaca")
        from broker.alpaca_client import StreamSupervisor

        with self._stream_lock:
            self._subscriptions.append((kind, handler, list(symbols)))
            if self._supervisor is None:
                self._supervisor = StreamSupervisor("market_data", self._make_stream).start()
            elif self._supervisor.stream is not None:
                getattr(self._supervisor.stream, f"subscribe_{kind}")(handler, *symbols)
        return self._supervisor

    def _make_stream(self):
        from alpaca.data.live import StockDataStream

        creds = self.credentials.get("alpaca", {})
        stream = StockDataStream(creds.get("api_key"), creds.get("secret_key"), feed=self._live_feed())
        for kind, handler, symbols in self._subscriptions:   # replayed on every reconnect
            getattr(stream, f"subscribe_{kind}")(handler, *symbols)
        return stream

    def stop_streams(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop()

    def _live_feed(self):
        from alpaca.data.enums import DataFeed

        return DataFeed(self.cfg.get("live_feed", "iex"))

    def latest_price(self, symbol: str) -> float:
        if self.source == "alpaca":
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockLatestTradeRequest

            # Latest trade must come from IEX on free plans (SIP is delayed)
            trade = self._alpaca_client().get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed(self.cfg.get("live_feed", "iex")))
            )
            return float(trade[symbol].price)
        return float(self.history(symbol, use_cache=False)["close"].iloc[-1])

    # --------------------------------------------------------------- backends
    def _alpaca_client(self):
        if self._alpaca is None:
            from alpaca.data.historical import StockHistoricalDataClient

            creds = self.credentials.get("alpaca", {})
            self._alpaca = StockHistoricalDataClient(creds.get("api_key"), creds.get("secret_key"))
        return self._alpaca

    def _fetch_alpaca(self, symbol: str, start: str, end: Optional[str], timeframe: str = "1Day") -> pd.DataFrame:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest

        # Free plans can't query SIP data from the last 15 minutes
        end_ts = pd.Timestamp(end, tz=timezone.utc) if end else pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(minutes=16)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=parse_timeframe(timeframe),
            start=pd.Timestamp(start, tz=timezone.utc).to_pydatetime(),
            end=end_ts.to_pydatetime(),
            adjustment=Adjustment.ALL,
            feed=DataFeed(self.cfg.get("feed", "sip")),
        )
        bars = self._alpaca_client().get_stock_bars(req).df
        if bars.empty:
            return pd.DataFrame(columns=OHLCV)
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")
        return self._normalize(bars, intraday=str(parse_timeframe(timeframe)) != "1Day")

    def _fetch_yfinance(self, symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return self._normalize(raw)

    @staticmethod
    def _normalize(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
        """Lower-case OHLCV columns; daily bars on tz-naive dates, intraday bars in US/Eastern."""
        if df.empty:
            return pd.DataFrame(columns=OHLCV)
        df = df.rename(columns=str.lower)
        idx = pd.DatetimeIndex(df.index)
        if intraday:
            idx = idx.tz_localize("UTC") if idx.tz is None else idx
            df.index = idx.tz_convert("America/New_York")
        else:
            if idx.tz is not None:
                idx = idx.tz_convert("America/New_York").tz_localize(None)
            df.index = idx.normalize()
        df.index.name = "date"
        df = df[[c for c in OHLCV if c in df.columns]].sort_index()
        return df[~df.index.duplicated(keep="last")].dropna(subset=["close"])

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"No cached data at {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
