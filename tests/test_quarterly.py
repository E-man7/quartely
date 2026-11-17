"""Quarterly Theory NY Q3-of-Q3 strategy: time model, levels, SMT, pivots, entries, sizing, execution,
live loop and backtest. Offline: fake broker and fake data client."""
from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from quarterly.backtest import Backtester, Trade, report
from quarterly.config import Pair, QuarterlyConfig, load_config
from quarterly.data import MinuteBars, iter_minutes
from quarterly.execution import Executor, bracket_prices, order_id, position_size
from quarterly.live import QuarterlyBot
from quarterly.signals import LONG, SHORT, Bar, PivotTracker, StrategyEngine
from quarterly.timemodel import NY, is_overnight, session_times, trading_day

DAY = date(2026, 7, 15)            # Wednesday, EDT


def ny(day: date, hhmm: str) -> pd.Timestamp:
    h, m = map(int, hhmm.split(":"))
    return pd.Timestamp(day.year, day.month, day.day, h, m, tz=NY)


def cfg(**kw) -> QuarterlyConfig:
    base = dict(strategy="q3", pairs=[Pair("AAA", "BBB")], smt_from="session_q3", pivot_len=2, rr=3.0, risk_pct=1.0)
    base.update(kw)
    return QuarterlyConfig(**base)


def bar(sym, day, hhmm, o, h, l, c):
    return Bar(sym, ny(day, hhmm), o, h, l, c, 1000)


# AAA makes a new session high at 10:00 while BBB doesn't (bearish SMT); a swing low at 10:46 (102.0)
# is confirmed after 10:48 closes; 10:49 closes below it (bearish MSS) above TDO 100 and session open 101.
AAA = [("00:00", 100, 100.2, 99.8, 100), ("07:00", 101, 101.2, 100.8, 101), ("08:30", 101, 101.1, 100.9, 101),
       ("10:00", 102, 103, 101.9, 102.8), ("10:44", 102.6, 102.8, 102.4, 102.5), ("10:45", 102.5, 102.6, 102.2, 102.3),
       ("10:46", 102.3, 102.4, 102.0, 102.2), ("10:47", 102.2, 102.5, 102.1, 102.4),
       ("10:48", 102.4, 102.6, 102.3, 102.5), ("10:49", 102.5, 102.5, 101.8, 101.9)]
BBB = [("00:00", 50, 50.1, 49.9, 50), ("07:00", 50.5, 51.2, 50.4, 50.6), ("08:30", 50.6, 50.7, 50.5, 50.6)] + \
      [(t, 50.6, 50.8, 50.5, 50.7) for t in ("10:00", "10:44", "10:45", "10:46", "10:47", "10:48", "10:49")]


def scenario(day=DAY, extra_aaa=(), extra_bbb=()):
    rows = [bar("AAA", day, *r) for r in list(AAA) + list(extra_aaa)] + [bar("BBB", day, *r) for r in list(BBB) + list(extra_bbb)]
    return pd.DataFrame([dataclasses.asdict(b) for b in rows]).sort_values(["ts", "symbol"]).reset_index(drop=True)


def run_engine(engine, df, has_position=lambda s: False):
    out = []
    for ts, minute in iter_minutes(df):
        out += engine.on_minute(ts, minute, has_position)
    return out


# ------------------------------------------------------------ time model
@pytest.mark.parametrize("day, start, window, utc_offset", [
    (date(2026, 7, 15), 7, ("10:45:00", "11:07:30"), -4),     # EDT
    (date(2026, 1, 14), 7, ("10:45:00", "11:07:30"), -5),     # EST
    (date(2026, 7, 15), 6, ("09:45:00", "10:07:30"), -4),
    (date(2026, 1, 14), 6, ("09:45:00", "10:07:30"), -5),
    (date(2026, 3, 9), 7, ("10:45:00", "11:07:30"), -4),      # Monday after the March 8 spring-forward
    (date(2026, 11, 2), 6, ("09:45:00", "10:07:30"), -5),     # Monday after the November 1 fall-back
])
def test_window_times(day, start, window, utc_offset):
    t = session_times(day, start)
    assert t.window_start.strftime("%H:%M:%S") == window[0] and t.window_end.strftime("%H:%M:%S") == window[1]
    assert t.window_start.utcoffset() == timedelta(hours=utc_offset)
    assert t.session_start.hour == start and t.q2_start - t.session_start == timedelta(minutes=90)
    assert t.q3_start - t.session_start == timedelta(minutes=180)
    assert t.session_end - t.session_start == timedelta(hours=6)
    assert t.window_end - t.window_start == timedelta(minutes=22, seconds=30)
    assert t.day_start.date() == day - timedelta(days=1) and t.day_start.strftime("%H:%M") == "18:00"


def test_window_membership_and_trading_day():
    t = session_times(DAY, 7)
    assert not t.in_window(ny(DAY, "10:44")) and t.in_window(ny(DAY, "10:45")) and t.in_window(ny(DAY, "11:07"))
    assert not t.in_window(ny(DAY, "11:08"))
    assert trading_day(ny(DAY, "17:59")) == DAY and trading_day(ny(DAY, "18:00")) == DAY + timedelta(days=1)
    assert trading_day(pd.Timestamp("2026-07-15 22:30", tz="UTC")) == DAY + timedelta(days=1)   # 18:30 NY
    assert is_overnight(ny(DAY, "20:00")) and is_overnight(ny(DAY, "03:59")) and not is_overnight(ny(DAY, "04:00"))


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        cfg(session_start_hour=8)
    with pytest.raises(ValueError):
        cfg(smt_from="lunch")
    c = load_config(env={"LIVE_TRADING": "true", "DRY_RUN": "1"})
    assert c.live_trading and c.dry_run and c.symbols == ["DIA", "QQQ", "SPY"]   # index triad; metals/FX disabled
    assert {k: sorted(p.partner for p in v) for k, v in c.partners.items()} == {
        "QQQ": ["DIA", "SPY"], "DIA": ["QQQ", "SPY"], "SPY": ["DIA", "QQQ"]}
    assert load_config(env={}).live_trading is False


# ------------------------------------------------------------------ TDO
def test_tdo_from_midnight_bar():
    e = StrategyEngine(cfg())
    run_engine(e, scenario())
    d = e.days["AAA"]
    assert (d.tdo, d.tdo_source, d.session_open) == (100, "midnight", 101)
    assert (d.day_high, d.day_low) == (103, 99.8)


def _late_open(fallback):
    e = StrategyEngine(cfg(tdo_fallback=fallback))
    rows = [("23:30", 99, 105, 98.5, 99), ("00:42", 98, 98.5, 97.5, 98), ("03:59", 97, 97, 97, 97),
            ("04:00", 96, 96.5, 95.5, 96), ("08:30", 101, 101, 101, 101)]
    df = pd.DataFrame([dataclasses.asdict(bar("AAA", DAY - timedelta(days=1) if r[0] == "23:30" else DAY, *r))
                       for r in rows])
    decisions = run_engine(e, df)
    return e.days["AAA"], decisions


def test_tdo_fallback_skip_day():
    d, decisions = _late_open("skip_day")
    assert d.tdo is None and "skip_day" in d.skip_reason
    assert "skip_day" in " ".join(decisions[-1].reasons) and decisions[-1].action is None
    assert d.day_high == 105                                     # 23:30 the previous evening counts for the day range


def test_tdo_fallback_first_premarket_bar():
    d, _ = _late_open("first_premarket_bar")
    assert (d.tdo, d.tdo_source) == (96, "first_premarket_bar")  # 00:42 and 03:59 ignored


# ------------------------------------------------------------------ SMT
def _smt_engine(**kw):
    return StrategyEngine(cfg(smt_from="session_start", **kw))


def _minute(e, hhmm, a=None, b=None, day=DAY):
    bars = {}
    if a:
        bars["AAA"] = bar("AAA", day, hhmm, *a)
    if b:
        bars["BBB"] = bar("BBB", day, hhmm, *b)
    return e.on_minute(ny(day, hhmm), bars)


def test_smt_exactly_one_new_high():
    e = _smt_engine()
    _minute(e, "07:00", (10, 11, 9, 10), (20, 21, 19, 20))
    _minute(e, "07:01", (10, 12, 9.5, 11), (20, 22, 19.5, 21))       # both new highs: no SMT
    assert not e.sessions["AAA"].smt_bear
    _minute(e, "07:02", (11, 13, 10, 12), (21, 21.5, 20, 21))        # only AAA: bearish SMT for both
    assert e.sessions["AAA"].smt_bear and e.sessions["BBB"].smt_bear
    assert not e.sessions["AAA"].smt_bull
    _minute(e, "07:03", (12, 12.5, 8, 9), (21, 21.2, 20.5, 21))      # only AAA new low: bullish SMT
    assert e.sessions["AAA"].smt_bull
    _minute(e, "07:04", (9, 9.5, 8.5, 9), (21, 21.2, 20.8, 21))
    assert e.sessions["AAA"].smt_bear and e.sessions["AAA"].smt_bull   # flags persist
    nxt = DAY + timedelta(days=1)
    _minute(e, "07:00", (9, 9.5, 8.5, 9), (21, 21.2, 20.8, 21), day=nxt)
    assert not e.sessions["AAA"].smt_bear and not e.sessions["AAA"].smt_bull   # reset at next session start


def test_smt_needs_both_bars_and_respects_smt_from():
    e = _smt_engine()
    _minute(e, "07:00", (10, 11, 9, 10), (20, 21, 19, 20))
    decisions = _minute(e, "07:01", a=(10, 12, 9.5, 11))              # partner bar missing
    assert not e.sessions["AAA"].smt_bear and "SMT not evaluated" in " ".join(decisions[0].reasons)
    q3 = StrategyEngine(cfg(smt_from="session_q3"))
    _minute(q3, "07:00", (10, 11, 9, 10), (20, 21, 19, 20))
    _minute(q3, "07:01", (10, 12, 9.5, 11), (20, 20.5, 19.5, 20))     # before Q3 (10:00): ignored
    assert not q3.sessions["AAA"].smt_bear
    _minute(q3, "10:00", (11, 13, 10, 12), (20, 20.5, 19.6, 20))
    assert q3.sessions["AAA"].smt_bear


def test_smt_inverse_pair():
    e = StrategyEngine(cfg(smt_from="session_start", pairs=[Pair("AAA", "BBB", inverse=True)]))
    _minute(e, "07:00", (10, 11, 9, 10), (20, 21, 19, 20))
    _minute(e, "07:01", (10, 12, 9.5, 11), (20, 20.5, 18, 19))       # AAA high + BBB low: in step, no SMT
    assert not e.sessions["AAA"].smt_bear
    _minute(e, "07:02", (11, 13, 10, 12), (19, 20, 18.5, 19))        # AAA high, BBB no new low: SMT
    assert e.sessions["AAA"].smt_bear


def _triad():
    return StrategyEngine(cfg(smt_from="session_start", pairs=[Pair("QQQ", "DIA"), Pair("QQQ", "SPY"), Pair("DIA", "SPY")]))


def _tri_minute(e, hhmm, **bars):
    return e.on_minute(ny(DAY, hhmm), {s: bar(s, DAY, hhmm, *v) for s, v in bars.items()})


def test_triad_smt_any_partner_diverges():
    e = _triad()
    _tri_minute(e, "07:00", QQQ=(10, 11, 9, 10), DIA=(20, 21, 19, 20), SPY=(30, 31, 29, 30))
    _tri_minute(e, "07:01", QQQ=(10, 12, 9.5, 11), DIA=(20, 22, 19.5, 21), SPY=(30, 32, 29.5, 31))   # all agree
    assert not any(e.sessions[s].smt_bear for s in ("QQQ", "DIA", "SPY"))
    _tri_minute(e, "07:02", QQQ=(11, 13, 10, 12), DIA=(21, 21.5, 20, 21), SPY=(31, 33, 30, 32))      # DIA lags
    q, d, sp = (e.sessions[s] for s in ("QQQ", "DIA", "SPY"))
    assert q.smt_bear and q.smt_bear_partner == "DIA"          # QQQ vs SPY agrees, vs DIA diverges: enough
    assert sp.smt_bear and sp.smt_bear_partner == "DIA"
    assert d.smt_bear and d.smt_bear_partner in ("QQQ", "SPY")
    assert not q.smt_bull


def test_triad_partner_missing_checks_the_others():
    e = _triad()
    _tri_minute(e, "07:00", QQQ=(10, 11, 9, 10), DIA=(20, 21, 19, 20), SPY=(30, 31, 29, 30))
    out = _tri_minute(e, "07:01", QQQ=(10, 12, 9.5, 11), SPY=(30, 30.5, 29.5, 30))                 # no DIA bar
    assert e.sessions["QQQ"].smt_bear and e.sessions["QQQ"].smt_bear_partner == "SPY"
    note = [d for d in out if d.symbol == "QQQ"][0].reasons[0]
    assert "no DIA bar" in note and "other partners" in note


def test_backtest_records_diverging_partner():
    bt = Backtester(cfg())
    (t,) = bt.run(scenario(extra_aaa=[("10:50", 101.9, 102.0, 101.0, 101.2), ("10:51", 101.2, 101.3, 98.0, 98.2)]))
    assert t.partner == "BBB"


# --------------------------------------------------------------- pivots
def test_pivots_confirm_only_after_right_bars():
    p = PivotTracker(2)
    highs = [1, 2, 5, 3, 2, 4, 1]
    seen = []
    for i, h in enumerate(highs):
        p.update(Bar("X", ny(DAY, f"09:{i:02d}"), h, h, h - 1, h))
        seen.append(p.swing_high)
    assert seen == [None, None, None, None, 5, 5, 5]               # pivot at bar 2 known at bar 4's close


def test_pivots_have_no_lookahead():
    highs = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]

    def states(n):
        p, out = PivotTracker(2), []
        for i, h in enumerate(highs[:n]):
            p.update(Bar("X", ny(DAY, f"09:{i:02d}"), h, h + 0.5, h - 0.5, h))
            out.append((p.swing_high, p.swing_low))
        return out

    full = states(len(highs))
    for n in range(1, len(highs)):
        assert states(n) == full[:n]


# --------------------------------------------------------------- entries
def test_short_entry_in_window():
    e = StrategyEngine(cfg())
    decisions = run_engine(e, scenario())
    entries = [d for d in decisions if d.action]
    assert len(entries) == 1
    d = entries[0]
    assert (d.symbol, d.action, d.ts) == ("AAA", SHORT, ny(DAY, "10:49"))
    assert d.bias == "SELL" and d.smt_bear and d.mss_bear and d.in_window
    assert (d.entry, d.stop) == (101.9, 103) and d.risk_per_share == pytest.approx(1.1)
    assert d.target == pytest.approx(101.9 - 3 * 1.1)
    bbb = [x for x in decisions if x.symbol == "BBB" and x.ts == ny(DAY, "10:49")][0]
    assert bbb.action is None and "no bearish MSS" in " ".join(bbb.reasons)


def test_no_entry_outside_window_or_in_position_or_over_limit():
    early = [d for d in run_engine(StrategyEngine(cfg(session_start_hour=6)), scenario()) if d.action]
    assert not early                                               # window is 09:45-10:07:30 with start 6
    held = [d for d in run_engine(StrategyEngine(cfg()), scenario(), has_position=lambda s: s == "AAA") if d.action]
    assert not held
    e = StrategyEngine(cfg())
    e._new_day(DAY)
    e.set_trades("AAA", 1)                                         # already traded today (e.g. before a restart)
    run_engine(e, scenario())
    assert e.days["AAA"].trades == 1


def test_long_entry_mirror():
    flip = lambda rows, k: [(t, k - o, k - l, k - h, k - c) for t, o, h, l, c in rows]   # noqa: E731
    rows = [bar("AAA", DAY, *r) for r in flip(AAA, 200)] + [bar("BBB", DAY, *r) for r in flip(BBB, 100)]
    df = pd.DataFrame([dataclasses.asdict(b) for b in rows]).sort_values(["ts", "symbol"])
    entries = [d for d in run_engine(StrategyEngine(cfg()), df) if d.action]
    assert [(d.symbol, d.action) for d in entries] == [("AAA", LONG)]
    assert entries[0].stop == pytest.approx(97) and entries[0].bias == "BUY"


# ---------------------------------------------------------------- sizing
@pytest.mark.parametrize("equity, risk_pct, rps, expected", [
    (100_000, 1.0, 2.37, 421), (100_000, 1.0, 1.1, 909), (50_000, 0.5, 300.0, 0), (100_000, 1.0, 0.0, 0),
    (100_000, 1.0, -1.0, 0), (0, 1.0, 1.0, 0), (100_000, 2.0, 0.5, 4000),
])
def test_position_size(equity, risk_pct, rps, expected):
    assert position_size(equity, risk_pct, rps) == expected


def test_bracket_price_rounding():
    assert bracket_prices(SHORT, 103.001, 98.6049) == (103.01, 98.6)   # short stop rounds up, never tighter
    assert bracket_prices(LONG, 96.999, 105.0) == (96.99, 105.0)


# ------------------------------------------------------------- execution
@pytest.fixture
def fake_client(broker_settings):
    from broker.alpaca_client import AlpacaClient
    from test_broker import CREDS, FakeTradingClient

    fake = FakeTradingClient(price=101.9)
    fake.get_asset = lambda s: SimpleNamespace(tradable=True, shortable=s != "FXE", easy_to_borrow=s != "FXE",
                                               fractionable=True)
    return AlpacaClient(broker_settings, CREDS, client_factory=lambda: fake), fake


@pytest.fixture
def broker_settings(settings):
    import copy

    s = copy.deepcopy(settings)
    s["broker"].update(paper_trading=True)
    return s


def _short_decision(sym="AAA"):
    e = StrategyEngine(cfg())
    d = [x for x in run_engine(e, scenario()) if x.action][0]
    return dataclasses.replace(d, symbol=sym)


def test_execute_bracket_order(fake_client):
    client, fake = fake_client
    out = Executor(cfg(), client).execute(_short_decision(), DAY, 1)
    req = fake.submitted[0]
    assert out.status == "submitted" and out.qty == 909 and req.qty == 909          # 1% of 100k / $1.10
    assert str(getattr(req.side, "value", req.side)) == "sell" and str(req.order_class.value) == "bracket"
    assert str(req.time_in_force.value) == "gtc"
    assert (req.stop_loss.stop_price, req.take_profit.limit_price) == (103.0, 98.6)
    assert req.client_order_id == order_id("AAA", DAY, 1) == "qt3-AAA-20260715-1"


def test_execute_skips_unshortable_and_dry_run(fake_client):
    client, fake = fake_client
    assert Executor(cfg(), client).execute(_short_decision("FXE"), DAY, 1).status == "skipped"
    dry = Executor(cfg(dry_run=True), client).execute(_short_decision(), DAY, 1)
    assert dry.status == "dry_run" and dry.counts_as_trade() and not fake.submitted


def test_trades_today_from_order_history(fake_client):
    client, fake = fake_client
    ex = Executor(cfg(), client)
    ex.execute(_short_decision(), DAY, 1)
    fake._new(symbol="QQQ", qty=1, side="buy", type="market", client_order_id="something-else", order_class="simple")
    assert ex.trades_today(DAY, session_times(DAY).day_start) == {"AAA": 1}


# ------------------------------------------------------------------ data
class FakeDataClient:
    def __init__(self, frames):
        self.frames = frames                       # feed -> DataFrame(symbol, timestamp, ohlcv)
        self.requests = []

    def get_stock_bars(self, req):
        feed = str(req.feed.value)
        self.requests.append(feed)
        if feed not in self.frames:
            raise Exception('{"message":"invalid feed: ' + feed + '"}')
        df = self.frames[feed]
        utc = lambda x: pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x)  # noqa: E731
        start, end = utc(req.start), utc(req.end)
        df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        return SimpleNamespace(df=df.set_index(["symbol", "timestamp"]))


def _feed(rows):
    return pd.DataFrame([{"symbol": s, "timestamp": ny(d, t).tz_convert("UTC"), "open": p, "high": p, "low": p,
                          "close": p, "volume": 1} for s, d, t, p in rows])


def test_feed_merge_and_closed_bars():
    boats = _feed([("AAA", DAY - timedelta(days=1), "23:00", 1), ("AAA", DAY, "05:00", 2)])
    iex = _feed([("AAA", DAY, "03:00", 3), ("AAA", DAY, "09:00", 4), ("AAA", DAY, "09:01", 5)])
    mb = MinuteBars(cfg(), client=FakeDataClient({"boats": boats, "iex": iex}))
    df = mb.fetch(["AAA"], ny(DAY - timedelta(days=1), "18:00"), ny(DAY, "12:00"))
    assert df["open"].tolist() == [1, 4, 5]                        # boats overnight, iex in the day
    closed = mb.closed_only(df, ny(DAY, "09:01") + pd.Timedelta(seconds=30))
    assert closed["open"].tolist() == [1, 4]                       # the 09:01 bar hasn't closed yet


def test_delayed_feed_request_stops_before_recent_minutes():
    boats = _feed([("AAA", DAY - timedelta(days=1), "23:00", 1)])
    iex = _feed([("AAA", DAY, "10:30", 4)])
    client = FakeDataClient({"boats": boats, "iex": iex})
    seen = []
    orig = client.get_stock_bars

    def spy(req):
        seen.append((str(req.feed.value), pd.Timestamp(req.end)))
        if str(req.feed.value) == "boats" and pd.Timestamp(req.end).tz_localize("UTC") > ny(DAY, "10:35"):
            raise Exception('{"message":"subscription does not permit querying recent BOATS data"}')
        return orig(req)

    client.get_stock_bars = spy
    now = ny(DAY, "10:50")
    mb = MinuteBars(cfg(), client=client, now_fn=lambda: now)
    df = mb.fetch(["AAA"], ny(DAY - timedelta(days=1), "18:00"), now)
    assert df["open"].tolist() == [1, 4]                         # overnight TDO bar still arrives
    boats_end = [e for f, e in seen if f == "boats"][0].tz_localize("UTC")
    assert boats_end == now - pd.Timedelta(minutes=16) and "boats" not in mb._unavailable


def test_unavailable_feed_is_skipped():
    iex = _feed([("AAA", DAY, "09:00", 4)])
    mb = MinuteBars(cfg(), client=FakeDataClient({"iex": iex}), sleep=lambda s: None)
    assert mb.fetch(["AAA"], ny(DAY, "00:00"), ny(DAY, "12:00"))["open"].tolist() == [4]
    assert "boats" in mb._unavailable


# ------------------------------------------------------------------ live
class StubBars:
    def __init__(self, df):
        self.df = df

    def fetch(self, symbols, start, end, day_feed=None):
        return self.df[(self.df["ts"] >= pd.Timestamp(start)) & (self.df["ts"] < pd.Timestamp(end))]

    closed_only = staticmethod(MinuteBars.closed_only)


def test_live_rebuild_then_trade_on_closed_bar(fake_client):
    client, fake = fake_client
    now = [ny(DAY, "10:49") + pd.Timedelta(seconds=30)]               # 10:49 bar not closed yet
    bot = QuarterlyBot(cfg(), {}, client=client, bars=StubBars(scenario()), now_fn=lambda: now[0])
    bot.connect()
    bot.rebuild()
    assert bot.last_bar == ny(DAY, "10:48") and bot.engine.sessions["AAA"].smt_bear
    assert bot.engine.days["AAA"].tdo == 100 and not fake.submitted
    now[0] = ny(DAY, "10:50") + pd.Timedelta(seconds=5)
    assert bot.step() == 1 and len(fake.submitted) == 1               # the 10:49 bar, now closed
    assert bot.step() == 0 and len(fake.submitted) == 1               # never twice


def test_live_ignores_stale_signal(fake_client):
    client, fake = fake_client
    now = [ny(DAY, "10:48") + pd.Timedelta(seconds=30)]
    bot = QuarterlyBot(cfg(), {}, client=client, bars=StubBars(scenario()), now_fn=lambda: now[0])
    bot.connect()
    bot.rebuild()
    now[0] = ny(DAY, "10:58")                                          # loop stalled for 8 minutes
    bot.step()
    assert not fake.submitted


# -------------------------------------------------------------- backtest
def test_backtest_target_and_report():
    target_hit = [("10:50", 101.9, 102.0, 101.0, 101.2), ("10:51", 101.2, 101.3, 98.0, 98.2)]
    bt = Backtester(cfg())
    trades = bt.run(scenario(extra_aaa=target_hit))
    (t,) = trades
    assert (t.side, t.entry, t.stop, t.exit_reason, t.exit) == (SHORT, 101.9, 103.0, "target", 98.6)
    assert t.qty == 909 and t.r == pytest.approx((101.9 - 98.6) / 1.1)
    rep = report(trades)
    assert rep.loc["AAA", "trades"] == 1 and rep.loc["AAA", "win_rate"] == 1.0
    assert rep.loc["ALL", "avg_r"] == pytest.approx(3.0)


def test_backtest_stop_wins_ties_and_gaps():
    bt = Backtester(cfg())
    t = Trade("AAA", SHORT, ny(DAY, "10:49"), ny(DAY, "10:50"), 100.0, stop=101.0, target=97.0, qty=10, risk_per_share=1.0)
    bt.open["AAA"] = t
    bt._exits(Bar("AAA", ny(DAY, "10:51"), 99.0, 101.5, 96.0, 98.0))   # both touched in one bar
    assert (t.exit_reason, t.exit, t.r) == ("stop", 101.0, -1.0)
    g = Trade("AAA", LONG, ny(DAY, "10:49"), ny(DAY, "10:50"), 100.0, stop=99.0, target=103.0, qty=10, risk_per_share=1.0)
    bt.open["AAA"] = g
    bt._exits(Bar("AAA", ny(DAY, "10:51"), 98.0, 98.5, 97.5, 98.0))    # gapped through the stop
    assert (g.exit, g.r) == (98.0, -2.0)


def test_backtest_and_live_share_the_engine():
    """The backtest's entries are exactly the engine's own decisions."""
    df = scenario()
    engine_entries = [(d.symbol, d.ts, d.action) for d in run_engine(StrategyEngine(cfg()), df) if d.action]
    bt = Backtester(cfg())
    bt.run(df)
    assert [(d.symbol, d.ts, d.action) for d in bt.decisions] == engine_entries


# ----------------------------------------------------------------- chart
def test_trade_chart(tmp_path):
    import json as _json

    from quarterly.chart import build_chart_data, chart_time, write_chart

    df = scenario(extra_aaa=[("10:50", 101.9, 102.0, 101.0, 101.2), ("10:51", 101.2, 101.3, 98.0, 98.2)])
    bt = Backtester(cfg())
    trades = pd.DataFrame([dataclasses.asdict(t) for t in bt.run(df)])
    data = build_chart_data(trades, df, cfg(), report(bt.trades))
    (t,) = data["trades"]
    assert (t["symbol"], t["partner"], t["side"], t["exit_reason"]) == ("AAA", "BBB", SHORT, "target")
    assert (t["tdo"], t["session_open"], t["pivot_price"]) == (100, 101, 102.0)
    assert t["signal"] == chart_time(ny(DAY, "10:49")) and t["smt"] == chart_time(ny(DAY, "10:00"))
    assert t["window"] == [chart_time(ny(DAY, "10:45")), chart_time(ny(DAY, "10:45") + pd.Timedelta(seconds=1350))]
    times = {b[0] for b in data["bars"]["AAA|2026-07-15"]}
    assert {t["signal"], t["fill"], t["exit"], t["smt"], t["pivot"]} <= times      # every marker sits on a bar
    assert "BBB|2026-07-15" in data["bars"] and len(data["bars"]) == 2           # only the days/symbols needed
    assert chart_time(ny(DAY, "10:49")) % 86400 == (10 * 60 + 49) * 60            # NY wall clock on the chart

    trades.loc[0, "tdo_source"] = "</script><script>alert(1)</script>"
    html = write_chart(tmp_path / "chart.html", trades, df, cfg()).read_text()
    assert "/*__DATA__*/null" not in html and "</script><script>alert" not in html
    payload = html.split("const DATA = ", 1)[1].split(";\nconst dark", 1)[0]
    assert _json.loads(payload)["trades"][0]["symbol"] == "AAA"


# ========================================================= triad 9:30 manipulation
from quarterly.config import TriadConfig  # noqa: E402
from quarterly.triad import TriadEngine, make_engine  # noqa: E402


def tcfg(triad=None, **kw) -> QuarterlyConfig:
    base = dict(strategy="triad930", pairs=[Pair("AAA", "BBB"), Pair("AAA", "CCC"), Pair("BBB", "CCC")],
                rr=3.0, risk_pct=1.0, triad=TriadConfig(**{"pick": "all", **(triad or {})}))
    base.update(kw)
    return QuarterlyConfig(**base)


# AAA and CCC stay inside their 8am ranges; BBB (the leader) takes its 8am high at 09:30, above its TDO 50.
T_AAA = [("00:00", 100, 100.5, 99.5, 100), ("08:00", 101, 102, 100.5, 101.5), ("08:30", 101.5, 101.8, 101, 101.2),
         ("09:00", 101.2, 101.5, 101, 101.3), ("09:30", 101.3, 101.6, 101.1, 101.2)]
T_BBB = [("00:00", 50, 50.2, 49.8, 50), ("08:00", 51, 51.5, 50.8, 51.2), ("08:30", 51.2, 51.3, 51, 51.1),
         ("09:00", 51.1, 51.2, 51, 51.1), ("09:30", 51.1, 51.8, 51.0, 51.6)]
T_CCC = [("00:00", 200, 200.5, 199.5, 200), ("08:00", 201, 203, 200, 202), ("08:30", 202, 202.5, 201.5, 202),
         ("09:00", 202, 202.2, 201.8, 202), ("09:30", 202, 202.5, 201.5, 202)]


def triad_df(aaa=T_AAA, bbb=T_BBB, ccc=T_CCC, day=DAY):
    rows = [bar(s, day, *r) for s, rs in (("AAA", aaa), ("BBB", bbb), ("CCC", ccc)) for r in rs]
    return pd.DataFrame([dataclasses.asdict(b) for b in rows]).sort_values(["ts", "symbol"]).reset_index(drop=True)


def replace_row(rows, hhmm, new):
    return [new if r[0] == hhmm else r for r in rows] + ([] if any(r[0] == hhmm for r in rows) else [new])


def entries(engine, df):
    return {(d.symbol, d.action) for d in run_engine(engine, df) if d.action}


def test_triad_is_the_default_strategy_and_q3_is_dormant():
    c = load_config(env={})
    assert c.strategy == "nyopen" and isinstance(make_engine(dataclasses.replace(c, strategy="triad930")), TriadEngine)
    assert isinstance(make_engine(cfg()), StrategyEngine)                    # q3 still available
    with pytest.raises(ValueError):
        tcfg(strategy="other")
    with pytest.raises(ValueError):
        TriadConfig(stop_mode="pivot")
    with pytest.raises(ValueError):
        TriadConfig(window_start=(8, 30))                                    # window inside the 8am range


def test_triad_sells_the_laggers_when_leader_takes_8am_high_above_tdo():
    e = TriadEngine(tcfg())
    decs = [d for d in run_engine(e, triad_df()) if d.action]
    assert {(d.symbol, d.action) for d in decs} == {("AAA", SHORT), ("CCC", SHORT)}   # BBB itself has no leader
    a = next(d for d in decs if d.symbol == "AAA")
    assert a.ts == ny(DAY, "09:30") and a.entry == 101.2
    assert a.stop == pytest.approx(102.05) and a.risk_per_share == pytest.approx(0.85)  # day high + 5 ticks
    assert a.target == pytest.approx(101.2 - 3 * 0.85)
    assert e.days["AAA"].range_high == 102 and e.days["AAA"].range_low == 100.5
    assert e.leader_of["AAA"] == "BBB" and "BBB took its 8am high" in a.reasons[0]


def test_triad_leader_taking_both_sides_means_no_trade():
    bbb = replace_row(T_BBB, "09:10", ("09:10", 51.0, 51.1, 50.6, 50.9))   # took the 8am low first
    assert entries(TriadEngine(tcfg()), triad_df(bbb=sorted(bbb))) == set()


def test_triad_leader_must_trade_beyond_its_tdo():
    bbb = replace_row(T_BBB, "00:00", ("00:00", 52, 52.2, 51.8, 52))      # TDO 52 above the 51.8 sweep
    assert entries(TriadEngine(tcfg()), triad_df(bbb=bbb)) == set()


def test_triad_buy_mirror():
    bbb = replace_row(T_BBB, "00:00", ("00:00", 52, 52.2, 51.8, 52))
    bbb = replace_row(bbb, "09:30", ("09:30", 51.0, 51.1, 50.5, 50.6))     # takes the 8am low, below TDO 52
    e = TriadEngine(tcfg())
    decs = [d for d in run_engine(e, triad_df(bbb=bbb)) if d.action]
    assert {(d.symbol, d.action) for d in decs} == {("AAA", LONG), ("CCC", LONG)}
    a = next(d for d in decs if d.symbol == "AAA")
    assert a.stop == pytest.approx(99.45) and a.target == pytest.approx(101.2 + 3 * (101.2 - 99.45))
    assert entries(TriadEngine(tcfg(triad={"allow_longs": False})), triad_df(bbb=bbb)) == set()


def test_triad_strict_lag():
    aaa = sorted(replace_row(T_AAA, "09:10", ("09:10", 101, 101.1, 100.4, 100.9)))   # AAA took its 8am low
    df = triad_df(aaa=aaa)
    assert entries(TriadEngine(tcfg()), df) == {("CCC", SHORT)}
    assert entries(TriadEngine(tcfg(triad={"strict_lag": False})), df) == {("AAA", SHORT), ("CCC", SHORT)}


def test_triad_window_and_sweep_timing():
    late = replace_row(T_BBB, "09:30", ("09:30", 51.1, 51.3, 51.0, 51.2))            # no sweep in the window
    late = late + [("09:40", 51.2, 51.9, 51.1, 51.8)]
    aaa, ccc = T_AAA + [("09:40", 101.2, 101.4, 101.1, 101.2)], T_CCC + [("09:40", 202, 202.4, 201.6, 202)]
    assert entries(TriadEngine(tcfg()), triad_df(aaa, late, ccc)) == set()          # 09:40 is outside 09:30-09:33
    early = sorted(replace_row(replace_row(T_BBB, "09:30", ("09:30", 51.1, 51.3, 51.0, 51.2)),
                               "09:05", ("09:05", 51.1, 51.7, 51.0, 51.5)))         # swept at 09:05 only
    assert entries(TriadEngine(tcfg()), triad_df(bbb=early)) == {("AAA", SHORT), ("CCC", SHORT)}
    assert entries(TriadEngine(tcfg(triad={"sweep_in_window": True})), triad_df(bbb=early)) == set()


def test_triad_needs_complete_range_and_premium_filter():
    no_range = [r for r in T_AAA if not r[0].startswith("08")]
    assert entries(TriadEngine(tcfg()), triad_df(aaa=no_range)) == {("CCC", SHORT)}
    aaa = replace_row(T_AAA, "00:00", ("00:00", 102, 102, 101.9, 102))              # AAA below its TDO
    assert entries(TriadEngine(tcfg(triad={"need_premium": True})), triad_df(aaa=aaa)) == {("CCC", SHORT)}


def test_triad_window_stop_mode():
    e = TriadEngine(tcfg(triad={"stop_mode": "window", "stop_buffer": 0.0}))
    a = next(d for d in run_engine(e, triad_df()) if d.symbol == "AAA" and d.action)
    assert a.stop == 101.6                                                           # the 09:30 bar's high


def test_triad_backtest_fills_next_open_and_flattens_at_1555():
    tail = lambda rows, px: rows + [("09:31", px, px + 0.1, px - 0.1, px), ("15:55", px - 0.3, px, px - 0.4, px - 0.2)]
    df = triad_df(tail(T_AAA, 101.1), tail(T_BBB, 51.5), tail(T_CCC, 202))
    bt = Backtester(tcfg())
    trades = {t.symbol: t for t in bt.run(df)}
    a = trades["AAA"]
    assert (a.entry_ts, a.entry) == (ny(DAY, "09:31"), 101.1)
    assert (a.exit_ts, a.exit, a.exit_reason) == (ny(DAY, "15:55"), pytest.approx(100.8), "flatten")
    assert (a.partner, a.range_high, a.range_low, a.tdo) == ("BBB", 102, 100.5, 100)
    assert a.smt_ts == ny(DAY, "09:30")


def test_triad_chart_and_order_prefix(tmp_path, fake_client):
    from quarterly.chart import build_chart_data, chart_time

    df = triad_df(T_AAA + [("09:31", 101.1, 101.2, 98, 98.5)], T_BBB, T_CCC)
    bt = Backtester(tcfg())
    trades = pd.DataFrame([dataclasses.asdict(t) for t in bt.run(df)])
    data = build_chart_data(trades, df, tcfg(), report(bt.trades))
    t = next(x for x in data["trades"] if x["symbol"] == "AAA")
    assert data["strategy"] == "triad930" and t["partner"] == "BBB" and t["exit_reason"] == "target"
    assert t["window"] == [chart_time(ny(DAY, "09:30")), chart_time(ny(DAY, "09:33"))]
    assert t["range"] == [chart_time(ny(DAY, "08:00")), chart_time(ny(DAY, "09:00"))] and t["range_high"] == 102

    client, fake = fake_client
    dec = next(d for d in run_engine(TriadEngine(tcfg()), triad_df()) if d.symbol == "AAA" and d.action)
    ex = Executor(tcfg(), client)
    assert ex.execute(dec, DAY, 1).order_id == "qt930-AAA-20260715-1"
    Executor(cfg(), client).execute(_short_decision(), DAY, 1)                      # a q3 order: not counted
    assert ex.trades_today(DAY, session_times(DAY).day_start) == {"AAA": 1}


def test_triad_live_rebuild_and_trade(fake_client):
    client, fake = fake_client
    now = [ny(DAY, "09:30") + pd.Timedelta(seconds=30)]
    bot = QuarterlyBot(tcfg(), {}, client=client, bars=StubBars(triad_df()), now_fn=lambda: now[0])
    bot.connect()
    bot.rebuild()
    assert bot.engine.days["AAA"].range_done and not fake.submitted
    now[0] = ny(DAY, "09:31") + pd.Timedelta(seconds=5)
    assert bot.step() == 1 and {r.symbol for r in fake.submitted} == {"AAA", "CCC"}
    assert all(r.client_order_id.startswith("qt930-") for r in fake.submitted)


# ------------------------------------------------- free plan: delayed SIP premarket
def test_premarket_feed_overrides_day_feed_in_premarket():
    c = tcfg(premarket_feed="sip", delayed_feeds=("boats", "sip"), recent_delay_minutes=16)
    now = ny(DAY, "09:40")
    fake = FakeDataClient({
        "boats": _feed([("AAA", DAY, "00:00", 100)]),
        "iex": _feed([("AAA", DAY, "08:00", 1), ("AAA", DAY, "09:20", 2), ("AAA", DAY, "09:30", 3)]),
        "sip": _feed([("AAA", DAY, "08:00", 10), ("AAA", DAY, "08:01", 11), ("AAA", DAY, "09:31", 99),
                      ("AAA", DAY, "09:26", 12)])})        # 09:26 is newer than now - 16 min: not requested
    df = MinuteBars(c, client=fake, now_fn=lambda: now).fetch(["AAA"], ny(DAY - timedelta(days=1), "18:00"), now)
    got = {t.strftime("%H:%M"): o for t, o in zip(df["ts"], df["open"])}
    assert got == {"00:00": 100, "08:00": 10, "08:01": 11, "09:20": 2, "09:30": 3}   # SIP only 04:00-09:30, delayed


class DayBars:
    """Bars as a free-plan feed serves them: delayed bars ('sip') only once 16 minutes old."""
    def __init__(self, iex, sip):
        self.iex, self.sip = iex, sip

    def fetch(self, symbols, start, end, day_feed=None):
        cut = pd.Timestamp(end) - pd.Timedelta(minutes=16)
        sip = self.sip[(self.sip["ts"] < cut)]
        df = pd.concat([self.iex, sip]).drop_duplicates(["symbol", "ts"], keep="last")
        return df[(df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))].sort_values(["ts", "symbol"])

    closed_only = staticmethod(MinuteBars.closed_only)


def test_live_replays_when_delayed_premarket_bars_arrive(fake_client):
    client, fake = fake_client
    # IEX: TDO bars, one stray AAA premarket bar (08:40) and the 9:30 bars; SIP: the whole premarket,
    # including an 08:35 AAA bar that raises its 8am high to 102.5 after 08:40 was already processed
    full = triad_df(aaa=sorted(T_AAA + [("08:35", 101.2, 102.5, 101.1, 101.3), ("08:40", 101.3, 101.4, 101.2, 101.3)]))
    pre = full["ts"] < ny(DAY, "09:30")
    stray = (full["symbol"] == "AAA") & (full["ts"] == ny(DAY, "08:40"))
    iex = full[~pre | (full["ts"] < ny(DAY, "01:00")) | stray]
    now = [ny(DAY, "08:50")]
    bot = QuarterlyBot(tcfg(), {}, client=client, bars=DayBars(iex, full[pre]), now_fn=lambda: now[0])
    bot.connect()
    bot.rebuild()
    assert bot.last_bar == ny(DAY, "08:40") and bot.engine.days["AAA"].range_high == 102   # 08:35 not served yet
    for hhmm in ("09:20", "09:31"):
        now[0] = ny(DAY, hhmm) + pd.Timedelta(seconds=5)
        bot.step()
    assert bot.replays == 1 and bot.engine.days["AAA"].range_high == 102.5 and bot.engine.days["AAA"].range_done
    assert {r.symbol for r in fake.submitted} == {"AAA", "CCC"}
    aaa = next(r for r in fake.submitted if r.symbol == "AAA")
    assert aaa.stop_loss.stop_price == 102.55                     # day high from the delayed bar + 5 cents


# ------------------------------------------------ pick the tightest stop, cap by buying power
def test_triad_picks_tightest_stop_and_one_trade_per_day():
    # AAA stop 0.85 on 101.2 (0.84%), CCC stop 1.05 on 202 (0.52%): CCC is picked
    e = TriadEngine(tcfg(triad={"pick": "tightest_stop"}))
    decs = run_engine(e, triad_df())
    assert {(d.symbol, d.action) for d in decs if d.action} == {("CCC", SHORT)}
    a = next(d for d in decs if d.symbol == "AAA" and d.ts == ny(DAY, "09:30"))
    assert a.action is None and a.stop is None and "not picked: CCC has the tightest stop" in a.reasons[-1]
    assert entries(TriadEngine(tcfg(triad={"pick": "widest_stop"})), triad_df()) == {("AAA", SHORT)}
    # after the pick trades, the other index is blocked on later window bars too
    e = TriadEngine(tcfg(triad={"pick": "tightest_stop"}))
    e.on_minute(ny(DAY, "00:00"), {})
    run_engine(e, triad_df()[triad_df()["ts"] < ny(DAY, "09:30")])
    e.record_trade("CCC")
    later = run_engine(e, triad_df()[triad_df()["ts"] >= ny(DAY, "09:30")])
    assert not any(d.action for d in later)
    assert any("triad trade already taken today" in r for d in later if d.symbol == "AAA" for r in d.reasons)


def test_position_capped_by_buying_power_and_leverage(fake_client):
    from quarterly.execution import capped_size

    assert capped_size(1000, 100.0, 50_000) == 500 and capped_size(10, 100.0, 50_000) == 10
    assert capped_size(10, 0, 50_000) == 0
    client, fake = fake_client                                     # fake account: equity 100k, buying power 200k
    dec = dataclasses.replace(_short_decision(), risk_per_share=0.01, stop=101.91, target=101.87)
    out = Executor(cfg(), client).execute(dec, DAY, 1)
    assert out.qty == int(0.95 * 200_000 / 101.9)                 # risk alone would size 100,000 shares
    bt = Backtester(tcfg(max_leverage=1.0))
    trades = bt.run(triad_df(T_AAA + [("09:31", 101.1, 101.2, 98, 98.5)], T_BBB, T_CCC))
    assert all(t.qty * 101.2 <= 0.95 * 100_000 + 1e-6 for t in trades if t.symbol == "AAA")


# ------------------------------------------------------- bad prints, backup stop
def test_clip_spikes_cuts_bad_prints_only():
    from quarterly.data import clip_spikes

    df = pd.DataFrame([dataclasses.asdict(b) for b in (
        Bar("QQQ", ny(DAY, "08:09"), 713.53, 724.657, 713.44, 713.731),     # the 2026-10-22 bad print
        Bar("QQQ", ny(DAY, "08:10"), 713.7, 714.2, 713.1, 714.0),           # normal bar
        Bar("SPY", ny(DAY, "09:30"), 766.0, 766.5, 750.0, 766.2))])         # bad low
    out = clip_spikes(df, 0.75)
    assert out.loc[0, "high"] == pytest.approx(713.731 * 1.0075) and out.loc[0, "low"] == 713.44
    assert out.loc[1].tolist() == df.loc[1].tolist()
    assert out.loc[2, "low"] == pytest.approx(766.0 * 0.9925)
    assert clip_spikes(df, 0).equals(df)


def test_triad_ignores_bad_print_in_8am_range():
    from quarterly.data import clip_spikes

    aaa = replace_row(T_AAA, "08:30", ("08:30", 101.5, 110.0, 101, 101.2))  # spike would set the 8am high to 110
    df = clip_spikes(triad_df(aaa=aaa), 0.75)
    e = TriadEngine(tcfg())
    run_engine(e, df)
    assert e.days["AAA"].range_high == pytest.approx(101.5 * 1.0075)        # cut back to 0.75% above the body


def _stub_order(sym, cid, stop, status="held", at=None):
    leg = SimpleNamespace(stop_price=stop, status=status, type="stop")
    return SimpleNamespace(symbol=sym, client_order_id=cid, legs=[SimpleNamespace(stop_price=None, status="new"), leg],
                           submitted_at=at or ny(DAY, "09:31"))


def test_protective_stops_from_order_history(fake_client):
    client, fake = fake_client
    orders = [_stub_order("SPY", "qt930-SPY-20261022-1", 764.5),
              _stub_order("QQQ", "qt930-QQQ-20261022-1", 720.66, status="filled"),       # already stopped out
              _stub_order("DIA", "qt3-DIA-20261022-1", 520.0)]                          # other strategy
    client.get_order_history = lambda **kw: orders
    assert Executor(tcfg(), client).protective_stops(ny(DAY, "00:00")) == {"SPY": 764.5}


def test_backup_stop_closes_when_broker_stop_does_not_fill(fake_client):
    client, fake = fake_client
    tail = [("09:31", 101.1, 101.2, 101.0, 101.1)]
    df = triad_df(T_AAA + tail, T_BBB + [("09:31", 51.5, 51.6, 51.4, 51.5)], T_CCC + [("09:31", 202, 202.1, 201.9, 202)])
    now = [ny(DAY, "09:31") + pd.Timedelta(seconds=5)]
    bars = StubBars(df)
    bot = QuarterlyBot(tcfg(), {}, client=client, bars=bars, now_fn=lambda: now[0])
    bot.connect()
    bot.rebuild()                                       # through 09:30: AAA and CCC sell signals were history
    bot.positions, bot.stops = {"AAA": -100}, {"AAA": 102.05}
    closed = []
    bot.executor.flatten = lambda syms, reason="": closed.append((tuple(syms), reason)) or list(syms)
    bot.enforce_stops(now[0])
    assert not closed                                   # 09:30 close 101.2 is below the short's stop
    bars.df = pd.concat([df, triad_df([("09:32", 102.0, 102.4, 101.9, 102.3)], [], [])])
    now[0] = ny(DAY, "09:33") + pd.Timedelta(seconds=5)
    bot.step()
    assert closed == [(("AAA",), "backup stop")] and "AAA" not in bot.stops
    # a stale bar never triggers it
    bot.positions, bot.stops, closed[:] = {"AAA": -100}, {"AAA": 102.05}, []
    bot.enforce_stops(ny(DAY, "09:50"))
    assert not closed


def test_runner_script_never_passes_live_trading():
    from pathlib import Path

    s = (Path(__file__).parents[1] / "scripts" / "run_quarterly.sh").read_text()
    assert "env -u LIVE_TRADING" in s and "caffeinate -i -w" in s


def test_loop_wake_time_survives_the_autumn_dst_hour():
    now = pd.Timestamp("2026-11-01 05:58:30", tz="UTC").tz_convert(NY)          # 01:58:30 EDT, the repeated hour
    with pytest.raises(Exception):
        now.floor("1min")
    wake = now.tz_convert("UTC").floor("1min") + pd.Timedelta(minutes=1, seconds=5)
    assert (wake - now).total_seconds() == pytest.approx(35)


# ========================================================= NY open v2 (nyopen)
from quarterly.config import NyOpenConfig  # noqa: E402
from quarterly.nyopen import NyOpenEngine  # noqa: E402


def ncfg(**ny) -> QuarterlyConfig:
    return QuarterlyConfig(strategy="nyopen", pairs=[Pair("AAA", "BBB"), Pair("AAA", "CCC"), Pair("BBB", "CCC")],
                           rr=3.0, risk_pct=1.0, nyopen=NyOpenConfig(**ny))


def nyday(aaa, bbb, ccc, day=DAY):
    return triad_df(aaa, bbb, ccc, day)


# All three above their 00:00 and 07:30 opens. AAA and BBB make new highs at 08:00, CCC doesn't (bearish SMT).
# Into 09:29 AAA falls hardest (momentum), and nobody has traded back to its opens.
def _base():
    aaa = [("00:00", 100, 100.2, 99.8, 100), ("07:30", 101, 101.2, 100.9, 101.1), ("08:00", 101.1, 103, 101, 102.8),
           ("09:26", 102.8, 102.9, 102.7, 102.8), ("09:27", 102.8, 102.8, 102.5, 102.6),
           ("09:28", 102.6, 102.6, 102.3, 102.4), ("09:29", 102.4, 102.4, 102.1, 102.2)]
    bbb = [("00:00", 50, 50.1, 49.9, 50), ("07:30", 50.5, 50.6, 50.4, 50.5), ("08:00", 50.5, 51.5, 50.5, 51.4),
           ("09:26", 51.4, 51.4, 51.3, 51.4), ("09:27", 51.4, 51.4, 51.3, 51.35),
           ("09:28", 51.35, 51.4, 51.3, 51.35), ("09:29", 51.35, 51.4, 51.3, 51.3)]
    ccc = [("00:00", 200, 200.2, 199.8, 200), ("07:30", 201, 202, 200.9, 201.5), ("08:00", 201.5, 201.9, 201.2, 201.6),
           ("09:26", 201.6, 201.8, 201.5, 201.7), ("09:27", 201.7, 201.8, 201.5, 201.7),
           ("09:28", 201.7, 201.8, 201.5, 201.7), ("09:29", 201.7, 201.8, 201.5, 201.6)]
    return aaa, bbb, ccc


def test_nyopen_is_the_default_strategy():
    c = load_config(env={})
    assert c.strategy == "nyopen" and c.strategies == ("nyopen", "silverbullet")
    assert isinstance(make_engine(dataclasses.replace(c, strategy="nyopen", strategies=("nyopen",))), NyOpenEngine)
    assert c.nyopen.entry_times == ((9, 30),) and c.nyopen.smt_from == (7, 30)              # 09:45 switched off
    with pytest.raises(ValueError):
        NyOpenConfig(smt_from=(8, 0))


def test_nyopen_sells_the_strongest_index_at_0930_targeting_the_nearer_open():
    e = NyOpenEngine(ncfg())
    decs = run_engine(e, nyday(*_base()))
    acts = [d for d in decs if d.action]
    assert [(d.symbol, d.action, d.ts) for d in acts] == [("AAA", SHORT, ny(DAY, "09:29"))]   # bar closing 09:30
    a = acts[0]
    assert e.smt_bear and a.entry == 102.2 and a.stop == 103 and a.target == 101.0          # higher of 100 / 101
    others = {d.symbol: d.reasons for d in decs if not d.action}
    assert any("stronger momentum" in r for r in others["BBB"])


def test_nyopen_skips_when_another_index_already_took_its_open():
    aaa, bbb, ccc = _base()
    bbb = replace_row(bbb, "09:29", ("09:29", 51.35, 51.4, 50.4, 50.6))      # BBB trades down to its 07:30 open 50.5
    decs = run_engine(NyOpenEngine(ncfg()), nyday(aaa, bbb, ccc))
    assert not any(d.action for d in decs if d.symbol == "AAA")
    assert any("BBB already at/through its 00:00/07:30 open" in r for d in decs if d.symbol == "AAA" for r in d.reasons)


def test_nyopen_needs_smt_and_bias():
    aaa, bbb, ccc = _base()
    ccc = replace_row(ccc, "08:00", ("08:00", 201.5, 203, 201.2, 201.6))     # all three new highs: no divergence
    assert not any(d.action for d in run_engine(NyOpenEngine(ncfg()), nyday(aaa, bbb, ccc)))
    aaa, bbb, ccc = _base()
    aaa = replace_row(aaa, "09:29", ("09:29", 102.4, 102.4, 100.4, 100.5))   # between its opens: no bias
    decs = run_engine(NyOpenEngine(ncfg()), nyday(aaa, bbb, ccc))
    assert any("no bias" in r for d in decs if d.symbol == "AAA" for r in d.reasons)


def test_nyopen_entry_times_follow_new_york_dst():
    for day, utc_hour in ((date(2026, 10, 28), 13), (date(2026, 11, 4), 14)):   # EDT, then EST after Nov 1
        e = NyOpenEngine(ncfg())
        decs = run_engine(e, nyday(*_base(), day=day))
        (a,) = [d for d in decs if d.action]
        assert a.ts.tz_convert("UTC").hour == utc_hour and a.ts.tz_convert("UTC").minute == 29


def test_nyopen_0945_only_and_one_trade_per_day():
    aaa, bbb, ccc = _base()
    later = [("09:44", 102.2, 102.2, 101.9, 102.0)]
    e = NyOpenEngine(ncfg(entry_times=((9, 45),)))
    decs = run_engine(e, nyday(aaa + later, bbb + [("09:44", 51.3, 51.3, 51.25, 51.28)],
                               ccc + [("09:44", 201.6, 201.7, 201.5, 201.6)]))
    assert [(d.symbol, d.ts) for d in decs if d.action] == [("AAA", ny(DAY, "09:44"))]
    e = NyOpenEngine(ncfg())
    e.on_minute(ny(DAY, "00:00"), {})
    run_engine(e, nyday(*_base()))
    e.record_trade("AAA")
    assert not any(d.action for d in run_engine(e, nyday(aaa + later, bbb + [("09:44", 51.3, 51.3, 51.25, 51.28)],
                                                          ccc + [("09:44", 201.6, 201.7, 201.5, 201.6)])))


def test_nyopen_backtest_opts_out_when_another_index_takes_its_open_first():
    aaa, bbb, ccc = _base()
    aaa += [("09:30", 102.2, 102.3, 101.9, 102.0), ("09:31", 102.0, 102.1, 101.8, 101.9),
            ("09:32", 101.9, 102.0, 101.7, 101.8)]
    bbb += [("09:30", 51.3, 51.3, 51.0, 51.1), ("09:31", 51.1, 51.1, 50.45, 50.5),     # BBB reaches 50.5 first
            ("09:32", 50.5, 50.6, 50.4, 50.5)]
    ccc += [("09:30", 201.6, 201.7, 201.5, 201.6), ("09:31", 201.6, 201.7, 201.5, 201.6),
            ("09:32", 201.6, 201.7, 201.5, 201.6)]
    (t,) = Backtester(ncfg()).run(nyday(aaa, bbb, ccc))
    assert (t.symbol, t.side, t.entry, t.target) == ("AAA", SHORT, 102.2, 101.0)
    assert (t.exit_reason, t.exit_ts, t.exit) == ("opt_out", ny(DAY, "09:32"), 101.9)
    assert t.r > 0


def test_nyopen_live_opts_out(fake_client):
    client, fake = fake_client
    aaa, bbb, ccc = _base()
    bot = QuarterlyBot(ncfg(other_open_rule="crossed"), {}, client=client, bars=StubBars(nyday(aaa, bbb, ccc)),
                       now_fn=lambda: ny(DAY, "09:30") + pd.Timedelta(seconds=5))
    bot.connect()
    bot.rebuild()
    closed = []
    bot.executor.flatten = lambda syms, reason="": closed.append((tuple(syms), reason)) or list(syms)
    bot.positions = {"AAA": -100}
    bot.maybe_opt_out()
    assert not closed
    bot.engine.days["BBB"].took_sell_ts = ny(DAY, "09:20")        # before the entry: no opt out
    bot.maybe_opt_out()
    assert not closed
    bot.engine.days["BBB"].took_sell_ts = ny(DAY, "09:31")        # after the entry
    bot.maybe_opt_out()
    assert closed and closed[0][0] == ("AAA",) and closed[0][1].startswith("opt out: BBB")
    assert order_id("AAA", DAY, 1, "qtny") == "qtny-AAA-20260715-1"


def test_replay_is_quiet_for_every_engine():
    from quarterly.live import QUIET_LOGGERS

    assert {"quarterly.signals", "quarterly.triad", "quarterly.nyopen"} <= set(QUIET_LOGGERS)


def test_true_day_open_is_the_price_at_midnight():
    prev = DAY - timedelta(days=1)

    def run(rows):
        e = NyOpenEngine(ncfg())
        e.cfg.tdo_fallback = "last_close_before_midnight"
        df = pd.DataFrame([dataclasses.asdict(bar("AAA", d, *r)) for d, r in rows])
        run_engine(e, df)
        return e.days["AAA"]

    d = run([(prev, ("23:17", 99, 99.5, 98.8, 99.2)), (DAY, ("00:00", 99.3, 99.4, 99.1, 99.2))])
    assert (d.tdo, d.tdo_source) == (99.3, "midnight")                       # a midnight bar: its open
    d = run([(prev, ("23:17", 99, 99.5, 98.8, 99.2)), (DAY, ("04:00", 97, 97.5, 96.5, 97))])
    assert (d.tdo, d.tdo_source) == (99.2, "last_close_before_midnight")     # last trade before 00:00
    d = run([(prev, ("18:30", 99, 99.5, 98.8, 99.2)), (DAY, ("04:00", 97, 97.5, 96.5, 97))])
    assert d.tdo is None and "skip_day" in d.skip_reason                     # nothing after 20:00: no TDO


def test_config_default_tdo_fallback():
    assert load_config(env={}).tdo_fallback == "last_close_before_midnight"


def test_nyopen_other_side_or_between_opens_does_not_block():
    """2026-11-03: QQQ and SPY above their opens (sells), DIA between its opens; DIA must not block the sell."""
    aaa, bbb, ccc = _base()
    ccc = [(t, o - 1.2, h - 1.2, l - 1.2, c - 1.2) if t >= "08:00" else (t, o, h, l, c) for t, o, h, l, c in ccc]
    decs = run_engine(NyOpenEngine(ncfg(other_open_rule="crossed")), nyday(aaa, bbb, ccc))   # CCC between its opens
    assert [(d.symbol, d.action) for d in decs if d.action] == [("AAA", SHORT)]
    assert not any(d.action for d in run_engine(NyOpenEngine(ncfg(other_open_rule="beyond")), nyday(aaa, bbb, ccc)))
    below = [(t, o - 2.5, h - 2.5, l - 2.5, c - 2.5) if t >= "08:00" else (t, o, h, l, c) for t, o, h, l, c in _base()[2]]
    decs = run_engine(NyOpenEngine(ncfg(other_open_rule="after_entry")), nyday(aaa, bbb, below))  # CCC: buy side
    assert [(d.symbol, d.action) for d in decs if d.action] == [("AAA", SHORT)]


def test_nyopen_first_to_target_race():
    """No index blocks the entry; the setup most likely to reach its target first trades; a rival reaching
    its own target first after the entry opts it out."""
    aaa, bbb, ccc = _base()
    bbb = replace_row(bbb, "09:29", ("09:29", 51.35, 51.4, 50.4, 50.6))       # BBB at its open: no block in a race
    e = NyOpenEngine(ncfg(other_open_rule="first_to_target", pick="momentum"))
    decs = run_engine(e, nyday(aaa, bbb, ccc))
    (a,) = [d for d in decs if d.action]
    assert a.symbol == "BBB"                                                  # strongest fall into 09:30
    assert set(e.context["BBB"]["rivals"]) == {"AAA", "CCC"}                  # both above their opens at entry
    assert e.exit_signal("BBB", SHORT) is None                                # nothing after the entry yet
    e.on_minute(ny(DAY, "09:30"), {"AAA": Bar("AAA", ny(DAY, "09:30"), 102.2, 102.2, 100.9, 101.0),
                                   "CCC": Bar("CCC", ny(DAY, "09:30"), 201.6, 201.7, 201.5, 201.6)})
    assert e.exit_signal("BBB", SHORT).startswith("opt out: AAA reached its own target first")
    e2 = NyOpenEngine(ncfg(other_open_rule="first_to_target", pick="nearest_target"))
    assert [d.symbol for d in run_engine(e2, nyday(aaa, bbb, ccc)) if d.action] == ["BBB"]  # 50.6 is 0.1 from 50.5
    with pytest.raises(ValueError):
        NyOpenConfig(pick="luck")


def test_nyopen_targets_the_0730_candle_and_enters_at_exactly_0930():
    aaa, bbb, ccc = _base()      # AAA's 07:30 candle: high 101.2, low 100.9; BBB's 50.6 / 50.4; CCC's 202 / 200.9
    cfg_ = ncfg(other_open_rule="first_to_target", pick="momentum", target="0730_candle")
    e = NyOpenEngine(cfg_)
    (a,) = [d for d in run_engine(e, nyday(aaa, bbb, ccc)) if d.action]
    assert (a.symbol, a.action, a.target) == ("AAA", SHORT, 100.9)            # sell targets the 07:30 low
    assert e.context["AAA"]["rivals"] == {"BBB": (SHORT, 50.4), "CCC": (SHORT, 200.9)}
    # an index with no bar closing at 09:30 doesn't delay the entry: judged on the 09:29 bar
    ccc_late = [r for r in ccc if r[0] != "09:29"] + [("09:30", 201.6, 201.7, 201.5, 201.6)]
    (a,) = [d for d in run_engine(NyOpenEngine(cfg_), nyday(aaa, bbb, ccc_late)) if d.action]
    assert a.ts == ny(DAY, "09:29")
    waits = NyOpenEngine(dataclasses.replace(cfg_, nyopen=dataclasses.replace(cfg_.nyopen, wait_for_all_bars=True)))
    full = nyday(aaa + [("09:30", 102.2, 102.2, 102.0, 102.1)], bbb + [("09:30", 51.3, 51.3, 51.25, 51.3)], ccc_late)
    assert [d.ts for d in run_engine(waits, full) if d.action] == [ny(DAY, "09:30")]       # waited a minute


def test_nyopen_buy_targets_the_0730_high():
    mirror = lambda rows: [(t, 400 - o, 400 - l, 400 - h, 400 - c) for t, o, h, l, c in rows]  # noqa: E731
    aaa, bbb, ccc = (mirror(x) for x in _base())
    e = NyOpenEngine(ncfg(other_open_rule="first_to_target", pick="momentum", target="0730_candle"))
    (a,) = [d for d in run_engine(e, nyday(aaa, bbb, ccc)) if d.action]
    assert (a.symbol, a.action, a.target) == ("AAA", LONG, 400 - 100.9)       # the 07:30 high of the mirrored bar


def test_nyopen_targets_the_0830_candle():
    aaa, bbb, ccc = _base()
    add = lambda rows, bar: sorted(rows + [bar])  # noqa: E731
    aaa = add(aaa, ("08:30", 102.0, 102.5, 101.6, 102.3))
    bbb = add(bbb, ("08:30", 51.0, 51.2, 50.9, 51.1))
    ccc = add(ccc, ("08:30", 201.5, 201.7, 201.3, 201.6))
    e = NyOpenEngine(ncfg(other_open_rule="first_to_target", pick="momentum", target="0830_candle"))
    (a,) = [d for d in run_engine(e, nyday(aaa, bbb, ccc)) if d.action]
    assert (a.symbol, a.action, a.target) == ("AAA", SHORT, 101.6)            # sell targets the 08:30 low
    assert e.context["AAA"]["rivals"] == {"BBB": (SHORT, 50.9), "CCC": (SHORT, 201.3)}
    mirror = lambda rows: [(t, 400 - o, 400 - l, 400 - h, 400 - c) for t, o, h, l, c in rows]  # noqa: E731
    e = NyOpenEngine(ncfg(other_open_rule="first_to_target", pick="momentum", target="0830_candle"))
    (a,) = [d for d in run_engine(e, nyday(*(mirror(x) for x in (aaa, bbb, ccc)))) if d.action]
    assert (a.symbol, a.action, a.target) == ("AAA", LONG, 400 - 101.6)       # buy targets the 08:30 high


# ------------------------------------------------- buying-power rejection retry
def test_refit_qty_from_broker_error():
    from quarterly.execution import refit_qty

    err = '{"buying_power":"326941.98","code":40310000,"cost_basis":"385251.2","message":"insufficient buying power"}'
    assert refit_qty(698, err) == 580                       # 698 x 326941.98 / 385251.2 x 0.98
    assert refit_qty(698, "connection reset") == 0
    assert refit_qty(698, '{"code":40310000,"message":"something else","buying_power":"1","cost_basis":"2"}') == 0


def test_execute_retries_smaller_after_buying_power_rejection(fake_client):
    client, fake = fake_client
    real = fake.submit_order
    err = '{"buying_power":"50000","code":40310000,"cost_basis":"95000","message":"insufficient buying power"}'

    def first_refused(req):
        if not fake.submitted and not getattr(fake, "refused", False):
            fake.refused = True
            raise type("APIError", (Exception,), {"status_code": 403})(err)
        return real(req)

    fake.submit_order = first_refused
    out = Executor(cfg(), client).execute(_short_decision(), DAY, 1)
    assert out.status == "submitted" and out.qty == int(909 * 50000 / 95000 * 0.98)
    assert len(fake.submitted) == 1 and fake.submitted[0].qty == out.qty


# ------------------------------------------------------------------ silver bullet
from quarterly.config import SilverBulletConfig  # noqa: E402
from quarterly.silverbullet import Candle, SilverBulletEngine, fvgs  # noqa: E402


def sbcfg(**sb) -> QuarterlyConfig:
    return QuarterlyConfig(strategy="silverbullet", pairs=[Pair("AAA", "BBB"), Pair("AAA", "CCC"), Pair("BBB", "CCC")],
                           rr=3.0, risk_pct=1.0, silverbullet=SilverBulletConfig(**sb))


# C1 = the 09:00 hour. At 10:21 AAA runs its C1 high (101) on an up-run that opened at 100.6, and 10:22 closes
# back below 100.6 (CISD) inside the 10:20-10:40 macro. BBB and CCC stay under their C1 highs: SMT.
def _sbrm(t=("10:20", "10:21", "10:22")):
    aaa = [("09:00", 100.5, 101, 99.5, 100.5), ("10:00", 100.7, 100.8, 100.4, 100.6),
           (t[0], 100.6, 100.9, 100.6, 100.85), (t[1], 100.85, 101.3, 100.8, 101.2),
           (t[2], 101.2, 101.25, 100.5, 100.55)]
    bbb = [("09:00", 50.5, 51, 50, 50.5), ("10:00", 50.5, 50.8, 50.4, 50.6)]
    ccc = [("09:00", 200.5, 201, 200, 200.5), ("10:00", 200.5, 200.8, 200.4, 200.6)]
    return aaa, bbb, ccc


def test_silverbullet_reversal_short_on_cisd_in_macro():
    e = SilverBulletEngine(sbcfg(models=("reversal",)))
    decs = [d for d in run_engine(e, nyday(*_sbrm())) if d.action]
    assert len(decs) == 1
    d = decs[0]
    assert (d.symbol, d.action, d.entry, d.stop, d.target) == ("AAA", SHORT, 100.55, 101.3, 99.5)
    assert e.context["AAA"]["model"] == "reversal" and e.context["AAA"]["smt_by"] == ["AAA"]


def test_fixed_target_rr_replaces_the_setup_target():
    # NY open: entry 102.2, stop 103 (day high), risk 0.8 -> 1:3 target 99.8 instead of the 07:30 level 101.0
    a = [d for d in run_engine(NyOpenEngine(ncfg(target_rr=3.0)), nyday(*_base())) if d.action][0]
    assert (a.entry, a.stop) == (102.2, 103) and a.target == pytest.approx(102.2 - 3 * 0.8)
    # Silver bullet: entry 100.55, stop 101.3, risk 0.75 -> 98.3; the setup still passed min_rr on C1's low (99.5)
    d = [d for d in run_engine(SilverBulletEngine(sbcfg(models=("reversal",), target_rr=3.0)), nyday(*_sbrm())) if d.action][0]
    assert (d.entry, d.stop) == (100.55, 101.3) and d.target == pytest.approx(100.55 - 3 * 0.75)
    with pytest.raises(ValueError):
        SilverBulletConfig(target_rr=-1)


def test_silverbullet_reversal_needs_smt_and_macro():
    aaa, bbb, ccc = _sbrm()
    bbb = bbb + [("10:21", 50.6, 51.2, 50.6, 51.1)]
    ccc = ccc + [("10:21", 200.6, 201.2, 200.6, 201.1)]
    assert not [d for d in run_engine(SilverBulletEngine(sbcfg()), nyday(aaa, bbb, ccc)) if d.action]   # all swept
    assert [d for d in run_engine(SilverBulletEngine(sbcfg(require_smt=False)), nyday(aaa, bbb, ccc)) if d.action]
    early = _sbrm(("10:05", "10:06", "10:07"))                                    # CISD at 10:08: no macro
    assert not [d for d in run_engine(SilverBulletEngine(sbcfg()), nyday(*early)) if d.action]


def test_fvg_detection():
    a, b, c = Candle(101, 101.4, 100.9, 101), Candle(101, 101, 100.5, 100.6), Candle(100.6, 100.7, 100.3, 100.4)
    assert fvgs([a, b, c]) == ((100.7, 100.9), None)
    assert fvgs([Candle(1, 2, 0.5, 1.9), Candle(1.9, 3, 1.8, 2.9), Candle(2.9, 3.5, 2.5, 3.4)]) == (None, (2, 2.5))


# Continuation: C1 = 08:00 (high 101), C2 = 09:00 takes 101.4 and closes 100.4 (bearish), leaving a 5m bearish FVG
# 100.7-100.9. At 10:20 price retraces into it (up-run open 100.6) and 10:21 closes 100.45: CISD -> short.
def _sbcm():
    aaa = [("08:00", 100.5, 101, 100, 100.5), ("09:00", 101, 101.4, 100.9, 101.0), ("09:05", 101.0, 101.0, 100.5, 100.6),
           ("09:10", 100.6, 100.7, 100.3, 100.4), ("10:20", 100.6, 100.8, 100.55, 100.78),
           ("10:21", 100.78, 100.8, 100.4, 100.45)]
    bbb = [("08:00", 50.5, 51, 50, 50.5), ("09:00", 50.5, 50.8, 50.4, 50.6)]
    ccc = [("08:00", 200.5, 201, 200, 200.5), ("09:00", 200.5, 200.8, 200.4, 200.6)]
    return aaa, bbb, ccc


def test_silverbullet_continuation_short_from_c2_fvg():
    e = SilverBulletEngine(sbcfg(models=("continuation",)))
    decs = [d for d in run_engine(e, nyday(*_sbcm())) if d.action]
    assert [(d.symbol, d.action, d.entry, d.stop, d.target) for d in decs] == [("AAA", SHORT, 100.45, 100.8, 100)]
    # a close above the FVG top before the CISD: the PDA was not respected
    aaa, bbb, ccc = _sbcm()
    aaa = replace_row(aaa, "10:20", ("10:20", 100.6, 101.0, 100.55, 100.95))
    assert not [d for d in run_engine(SilverBulletEngine(sbcfg(models=("continuation",))), nyday(aaa, bbb, ccc))
                if d.action]


def test_silverbullet_backtest_and_config():
    aaa, bbb, ccc = _sbrm()
    trades = Backtester(sbcfg(models=("reversal",))).run(nyday(aaa + [("10:23", 100.55, 100.6, 99.4, 99.45)], bbb, ccc))
    assert len(trades) == 1 and trades[0].model == "reversal" and trades[0].range_high == 101
    assert trades[0].exit_reason == "target" and trades[0].pnl > 0
    c = load_config(env={})
    assert c.silverbullet.sessions == ((10, 0), (14, 0)) and c.silverbullet.macros == ((20, 40), (50, 70))
    with pytest.raises(ValueError):
        SilverBulletConfig(sessions=((11, 0),))


def _shift(rows, hours):
    out = []
    for hhmm, *ohlc in rows:
        h, m = map(int, hhmm.split(":"))
        out.append((f"{h + hours:02d}:{m:02d}", *ohlc))
    return out


def _trade_all(e, df):
    taken = []
    for ts, minute in iter_minutes(df):
        for d in e.on_minute(ts, minute):
            if d.action:
                taken.append(d.ts.strftime("%H:%M"))
                e.record_trade(d.symbol)
    return taken


def test_silverbullet_pm_session_and_one_trade_per_session():
    e = SilverBulletEngine(sbcfg(models=("reversal",), sessions=((10, 0), (14, 0))))
    pm = [_shift(rows, 4) for rows in _sbrm()]                      # same setup, C1 13:00, CISD 14:22
    assert _trade_all(e, nyday(*[a + p for a, p in zip(_sbrm(), pm)])) == ["10:22", "14:22"]
    # a second CISD in the same hour is not taken
    aaa, bbb, ccc = _sbrm()
    aaa = aaa + [("10:30", 100.5, 100.9, 100.5, 100.85), ("10:31", 100.85, 101.4, 100.8, 101.35),
                 ("10:32", 101.35, 101.36, 100.4, 100.45)]
    assert _trade_all(SilverBulletEngine(sbcfg(models=("reversal",))), nyday(aaa, bbb, ccc)) == ["10:22"]
    e = SilverBulletEngine(sbcfg(models=("reversal",), max_trades_per_session=2))
    assert _trade_all(e, nyday(aaa, bbb, ccc)) == ["10:22", "10:32"]


# ------------------------------------------------------------------ several strategies in one bot
def multicfg(**kw) -> QuarterlyConfig:
    return QuarterlyConfig(strategy="nyopen", strategies=("nyopen", "silverbullet"),
                           pairs=[Pair("AAA", "BBB"), Pair("AAA", "CCC"), Pair("BBB", "CCC")], rr=3.0, risk_pct=1.0,
                           nyopen=NyOpenConfig(entry_times=((9, 30),), smt_from=(7, 30), other_open_rule="first_to_target"),
                           silverbullet=SilverBulletConfig(models=("reversal",)), **kw)


def test_multi_engine_runs_both_and_tags_strategy():
    from quarterly.multi import MultiEngine

    e = make_engine(multicfg())
    assert isinstance(e, MultiEngine) and set(e.engines) == {"nyopen", "silverbullet"}
    decs = [d for d in run_engine(e, nyday(*_sbrm())) if d.action]
    assert [(d.symbol, d.strategy) for d in decs] == [("AAA", "silverbullet")]
    e.record_trade("AAA", "silverbullet")
    assert e.trades("AAA", "silverbullet") == 1 and e.trades("AAA", "nyopen") == 0 and e.owner["AAA"] == "silverbullet"
    # an index already in a position (e.g. from the other strategy) is not entered again
    e2 = make_engine(multicfg())
    assert not [d for d in run_engine(e2, nyday(*_sbrm()), has_position=lambda s: s == "AAA") if d.action]


def test_multi_engine_backtest_records_strategy():
    aaa, bbb, ccc = _sbrm()
    trades = Backtester(multicfg()).run(nyday(aaa + [("10:23", 100.55, 100.6, 99.4, 99.45)], bbb, ccc))
    assert [(t.symbol, t.strategy, t.model) for t in trades] == [("AAA", "silverbullet", "reversal")]


def test_single_strategy_override():
    c = dataclasses.replace(multicfg(), strategy="silverbullet", strategies=("silverbullet",))
    assert c.strategies == ("silverbullet",) and type(make_engine(c)).__name__ == "SilverBulletEngine"


def test_multi_live_trades_with_the_strategy_prefix_and_restarts_safely(fake_client):
    client, fake = fake_client
    df = nyday(*_sbrm())
    now = [ny(DAY, "10:22") + pd.Timedelta(seconds=30)]
    c = multicfg(premarket_feed=None)
    bot = QuarterlyBot(c, {}, client=client, bars=StubBars(df), now_fn=lambda: now[0])
    bot.connect()
    bot.rebuild()
    assert set(bot.executors) == {"nyopen", "silverbullet"} and not fake.submitted
    now[0] = ny(DAY, "10:23") + pd.Timedelta(seconds=5)
    bot.step()
    assert [r.client_order_id for r in fake.submitted] == ["qtsb-AAA-20260715-1"]
    # restart: today's silver bullet trade is read back from the broker under its prefix only
    bot2 = QuarterlyBot(c, {}, client=client, bars=StubBars(df), now_fn=lambda: now[0])
    bot2.connect()
    bot2.rebuild()
    assert bot2._sub("silverbullet").days["AAA"].trades == 1 and bot2._sub("nyopen").days["AAA"].trades == 0
