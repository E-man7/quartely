"""Broker integration against a fake Alpaca TradingClient (no network, no real orders)."""
from __future__ import annotations

import copy
import itertools
import json
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from broker.alpaca_client import LIVE_CONFIRMATION, LIVE_URL, PAPER_URL, AlpacaClient, confirm_live_trading
from broker.order_executor import OrderExecutor, TradeJournal
from broker.position_tracker import PositionTracker
from config import load_credentials
from core.regime_strategies import Direction, Signal
from core.risk_manager import RiskManager
from data.market_data import MarketData, find_gaps, parse_timeframe

CREDS = {"alpaca": {"api_key": "k", "secret_key": "s"}}


# ------------------------------------------------------------------ fakes
def _v(x):
    return str(getattr(x, "value", x)).lower()


class FakeTradingClient:
    def __init__(self, price=100.0, fill_limits=True):
        self.price = price
        self.fill_limits = fill_limits
        self.orders = {}
        self.submitted, self.cancelled, self.replaced = [], [], []
        self.positions = {}
        self.cash = 100_000.0
        self._ids = itertools.count(1)
        self.fail_account = []          # exceptions to raise from get_account, in order

    # account / clock / assets
    def get_account(self):
        if self.fail_account:
            raise self.fail_account.pop(0)
        mv = sum(p["qty"] * self.price for p in self.positions.values())
        eq = self.cash + mv
        return SimpleNamespace(status="ACTIVE", equity=eq, last_equity=eq, cash=self.cash, buying_power=2 * eq,
                               regt_buying_power=2 * eq, daytrading_buying_power=4 * eq, multiplier="2",
                               initial_margin=mv * 0.5, maintenance_margin=mv * 0.25, trading_blocked=False,
                               account_blocked=False, pattern_day_trader=False)

    def get_clock(self):
        return SimpleNamespace(is_open=True, next_open=None, next_close=None, timestamp=pd.Timestamp.now())

    def get_asset(self, symbol):
        return SimpleNamespace(tradable=True)

    def get_all_positions(self):
        return [SimpleNamespace(symbol=s, qty=p["qty"], avg_entry_price=p["avg"], current_price=self.price,
                                market_value=p["qty"] * self.price, unrealized_pl=0.0, unrealized_plpc=0.0)
                for s, p in self.positions.items() if p["qty"]]

    # orders
    def _new(self, **kw):
        oid = f"o{next(self._ids)}"
        fields = {"status": "new", "filled_qty": 0.0, "filled_avg_price": None, "legs": [], "parent_id": None,
                  "stop_price": None, "limit_price": None, "filled_at": None, **kw}
        o = SimpleNamespace(id=oid, **fields)
        self.orders[oid] = o
        return o

    def _fill(self, o, qty):
        o.status, o.filled_qty, o.filled_avg_price = "filled", qty, self.price
        sign = 1 if _v(o.side) == "buy" else -1
        p = self.positions.setdefault(o.symbol, {"qty": 0.0, "avg": self.price})
        p["qty"] += sign * qty
        self.cash -= sign * qty * self.price

    def submit_order(self, req):
        self.submitted.append(req)
        kind = _v(req.type)
        o = self._new(symbol=req.symbol, qty=float(req.qty), side=req.side, type=kind,
                      client_order_id=req.client_order_id, order_class=_v(req.order_class or "simple"),
                      limit_price=getattr(req, "limit_price", None), stop_price=getattr(req, "stop_price", None))
        if kind == "market" or (kind == "limit" and self.fill_limits):
            self._fill(o, o.qty)
            if req.stop_loss is not None:     # bracket / OTO: stop leg goes live once the parent fills
                leg = self._new(symbol=req.symbol, qty=o.qty, side="sell", type="stop",
                                client_order_id=None, order_class="oto", stop_price=req.stop_loss.stop_price)
                leg.parent_id = o.id
                o.legs.append(leg)
        return o

    def get_order_by_id(self, oid):
        return self.orders[oid]

    def cancel_order_by_id(self, oid):
        o = self.orders[oid]
        if o.status == "filled":                     # Alpaca: 422 order is not cancelable
            err = Exception(f"order {oid} is already filled")
            err.status_code = 422
            raise err
        self.cancelled.append(oid)
        o.status = "canceled"
        for leg in o.legs:
            leg.status = "canceled"

    def replace_order_by_id(self, oid, req):
        self.replaced.append((oid, req))
        o = self.orders[oid]
        if req.stop_price is not None:
            o.stop_price = req.stop_price
        if req.qty is not None:
            o.qty = float(req.qty)
        return o

    def get_orders(self, filter=None):
        open_ = lambda o: o.status in ("new", "accepted")  # noqa: E731
        out = []
        for o in self.orders.values():
            if o.parent_id is not None:
                continue
            if _v(filter.status) == "open" and not (open_(o) or any(open_(leg) for leg in o.legs)):
                continue
            if filter.symbols and o.symbol not in filter.symbols:
                continue
            out.append(o)
        return out

    def close_all_positions(self, cancel_orders=True):
        self.closed_all = True

    def close_position(self, symbol):
        self.positions.pop(symbol, None)


@pytest.fixture
def broker_settings(settings):
    s = copy.deepcopy(settings)
    s["broker"].update(paper_trading=True, dry_run=False, fractional=False, protective_stops=True,
                       fill_timeout_seconds=30, retry_at_market=True)
    return s


def make(settings, fake, tmp_path, risk=None):
    client = AlpacaClient(settings, CREDS, client_factory=lambda: fake)
    journal = TradeJournal(tmp_path / "journal.jsonl")
    tracker = PositionTracker(client, tmp_path / "positions.json", risk=risk, journal=journal)
    clock = itertools.count(0, 1.0)
    ex = OrderExecutor(settings, client, tracker, lambda s: fake.price, risk=risk, journal=journal,
                       sleep=lambda s: None, clock=lambda: next(clock))
    tracker.refresh()
    return client, tracker, ex, journal


def long_signal(stop=95.0, tp=None, weight=0.5):
    return Signal("SPY", Direction.LONG, 0.9, 100.0, stop, tp, weight, 1.0, 0, "BULL", 0.9,
                  pd.Timestamp("2024-01-08"), "test", "LowVolBullStrategy", {})


# ------------------------------------------------------------ credentials
def test_dotenv_credentials(tmp_path, monkeypatch):
    env_copy = {k: v for k, v in os.environ.items() if not k.startswith("ALPACA_")}
    monkeypatch.setattr(os, "environ", env_copy)                 # .env loading must not leak into the real env
    os.environ["ALPACA_SECRET_KEY"] = "from-env"
    env = tmp_path / ".env"
    env.write_text('# comment\nexport ALPACA_API_KEY="abc123"\nALPACA_SECRET_KEY=from-file  # inline\n')
    yaml_file = tmp_path / "creds.yaml"
    yaml_file.write_text("alpaca:\n  api_key: HARDCODED\n")
    creds = load_credentials(str(yaml_file), env_file=env)
    assert creds["alpaca"] == {"api_key": "abc123", "secret_key": "from-env"}   # env wins, yaml keys ignored


def test_paper_default_and_live_confirmation(broker_settings):
    fake = FakeTradingClient()
    assert AlpacaClient(broker_settings, CREDS, client_factory=lambda: fake).base_url == PAPER_URL
    live = copy.deepcopy(broker_settings)
    live["broker"]["paper_trading"] = False
    with pytest.raises(RuntimeError):
        AlpacaClient(live, CREDS, confirm=lambda: confirm_live_trading(lambda _: "yes", interactive=True),
                     client_factory=lambda: fake)
    with pytest.raises(RuntimeError, match="interactive"):
        confirm_live_trading(lambda _: LIVE_CONFIRMATION, interactive=False)
    client = AlpacaClient(live, CREDS, confirm=lambda: confirm_live_trading(lambda _: LIVE_CONFIRMATION, True),
                          client_factory=lambda: fake)
    assert client.base_url == LIVE_URL and not client.paper
    with pytest.raises(RuntimeError, match="credentials"):
        AlpacaClient(broker_settings, {}, client_factory=lambda: fake)


def test_retry_reconnects_with_backoff(broker_settings, monkeypatch):
    sleeps = []
    monkeypatch.setattr("broker.alpaca_client.time.sleep", sleeps.append)
    fake = FakeTradingClient()
    fake.fail_account = [ConnectionError("reset"), ConnectionError("reset")]
    builds = []
    client = AlpacaClient(broker_settings, CREDS, client_factory=lambda: builds.append(1) or fake)
    assert client.get_account()["status"] == "active"
    assert len(sleeps) == 2 and len(builds) == 3                 # initial connect + 2 reconnects
    forbidden = Exception("forbidden")
    forbidden.status_code = 403
    fake.fail_account = [forbidden]
    with pytest.raises(Exception, match="forbidden"):
        client.get_account()
    assert len(sleeps) == 2                                      # 4xx: no retry


def test_health_check_and_margin(broker_settings):
    client = AlpacaClient(broker_settings, CREDS, client_factory=lambda: FakeTradingClient())
    assert client.health_check()["paper"] is True and client.is_market_open()
    m = client.get_available_margin()
    assert m["multiplier"] == 2 and m["excess_margin"] == pytest.approx(100_000)


# ----------------------------------------------------------------- orders
def test_limit_order_then_market_retry(broker_settings, tmp_path):
    fake = FakeTradingClient(fill_limits=False)
    _, _, ex, _ = make(broker_settings, fake, tmp_path)
    res = ex.submit_order(symbol="SPY", side="buy", qty=10)
    limit, market = fake.submitted
    assert _v(limit.type) == "limit" and limit.limit_price == pytest.approx(100.10)
    assert _v(market.type) == "market" and market.qty == 10
    assert fake.cancelled == [res.order_ids[0]]
    assert res.status == "filled" and res.filled_qty == 10
    assert limit.client_order_id == f"{res.trade_id}-1" and market.client_order_id == f"{res.trade_id}-2"


def test_limit_sell_price_and_no_retry(broker_settings, tmp_path):
    s = copy.deepcopy(broker_settings)
    s["broker"]["retry_at_market"] = False
    fake = FakeTradingClient(fill_limits=False)
    _, _, ex, _ = make(s, fake, tmp_path)
    res = ex.submit_order(symbol="SPY", side="sell", qty=5)
    assert fake.submitted[0].limit_price == pytest.approx(99.90)
    assert len(fake.submitted) == 1 and res.status == "canceled"


@pytest.mark.parametrize("tp, order_class", [(None, "oto"), (110.0, "bracket")])
def test_bracket_order(broker_settings, tmp_path, tp, order_class):
    fake = FakeTradingClient()
    _, tracker, ex, _ = make(broker_settings, fake, tmp_path)
    res = ex.submit_bracket_order(long_signal(stop=95.0, tp=tp), qty=10.7)
    req = fake.submitted[0]
    assert _v(req.order_class) == order_class and req.qty == 10 and req.stop_loss.stop_price == 95.0
    assert (req.take_profit.limit_price if req.take_profit else None) == tp
    assert res.status == "filled"
    bad = ex.submit_bracket_order(long_signal(stop=101.0), qty=1)
    assert bad.status == "rejected" and "stop" in bad.error


def test_modify_stop_only_tightens(broker_settings, tmp_path):
    fake = FakeTradingClient()
    _, _, ex, _ = make(broker_settings, fake, tmp_path)
    ex.submit_bracket_order(long_signal(stop=95.0), qty=10)
    assert ex.modify_stop("SPY", 94.0) is False and not fake.replaced
    assert ex.modify_stop("SPY", 97.0) is True
    assert fake.replaced[-1][1].stop_price == 97.0


def test_rebalance_links_trade_id_and_keeps_stop(broker_settings, tmp_path):
    fake = FakeTradingClient()
    risk = RiskManager(broker_settings, backtest=True)
    _, tracker, ex, journal = make(broker_settings, fake, tmp_path, risk=risk)
    sig = long_signal(stop=95.0, weight=0.5)
    decision = risk.validate_signal(sig, tracker.portfolio_state())
    ex.rebalance({"SPY": 0.5}, sig, decision)
    trade_id = journal.read()[0]["trade_id"]
    assert [r["stage"] for r in journal.read(trade_id)] == ["signal", "risk_decision", "order", "order_status"]
    assert fake.submitted[0].client_order_id == f"{trade_id}-1"            # bracket entry
    assert fake.positions["SPY"]["qty"] == 500

    # Downsize: the stop reserving the shares is cancelled first, then re-placed for what's left
    fake.price = 100.0
    ex.rebalance({"SPY": 0.25}, sig, decision)
    stop_leg = next(o for o in fake.orders.values() if o.parent_id)
    assert stop_leg.id in fake.cancelled
    new_stop = fake.submitted[-1]
    assert _v(new_stop.type) == "stop" and new_stop.qty == fake.positions["SPY"]["qty"] == 250


def test_close_all_and_dry_run(broker_settings, tmp_path):
    fake = FakeTradingClient()
    _, _, ex, _ = make(broker_settings, fake, tmp_path)
    ex.close_all_positions()
    assert fake.closed_all
    s = copy.deepcopy(broker_settings)
    s["broker"]["dry_run"] = True
    fake2 = FakeTradingClient()
    _, _, ex2, _ = make(s, fake2, tmp_path)
    assert ex2.submit_order(symbol="SPY", side="buy", qty=1).status == "dry_run" and not fake2.submitted


# ---------------------------------------------------------------- tracker
def test_fill_updates_position_journal_and_breaker(broker_settings, tmp_path):
    fake = FakeTradingClient()
    risk = RiskManager(broker_settings, backtest=True)
    _, tracker, ex, journal = make(broker_settings, fake, tmp_path, risk=risk)
    res = ex.submit_bracket_order(long_signal(stop=95.0), qty=100)
    fired = []
    tracker.on_breaker.append(fired.append)
    update = SimpleNamespace(event="fill", qty=100, price=100.0, position_qty=100, timestamp=pd.Timestamp.now(tz="UTC"),
                             order=SimpleNamespace(symbol="SPY", side="buy", client_order_id=f"{res.trade_id}-1"))
    tracker.handle_trade_update(update)
    pos = tracker.tracked["SPY"]
    assert (pos.entry_price, pos.stop_level, pos.regime_at_entry, pos.trade_id) == (100.0, 95.0, "BULL", res.trade_id)
    assert journal.read(res.trade_id)[-1]["stage"] == "fill"
    assert risk.breaker.equity == pytest.approx(tracker.equity)
    fake.price = 50.0                                              # equity collapses: breakers must fire on the next fill
    tracker.handle_trade_update(SimpleNamespace(**{**vars(update), "event": "partial_fill", "qty": 1}))
    assert fired and risk.breaker.check().flatten
    assert tracker.handle_trade_update(SimpleNamespace(**{**vars(update), "event": "new"})) is None


def test_reconcile_adopts_and_drops(broker_settings, tmp_path):
    fake = FakeTradingClient()
    fake.positions["SPY"] = {"qty": 20.0, "avg": 98.0}
    buy = fake._new(symbol="SPY", qty=20, side="buy", type="market", client_order_id=None, order_class="simple")
    buy.status, buy.filled_at = "filled", pd.Timestamp("2024-01-02 15:46", tz="UTC")
    stop = fake._new(symbol="SPY", qty=20, side="sell", type="stop", client_order_id=None, order_class="simple")
    stop.stop_price = 93.0
    (tmp_path / "positions.json").write_text(json.dumps({"QQQ": {"symbol": "QQQ", "qty": 5, "entry_time": None,
                                                                  "entry_price": 400, "current_price": 400}}))
    _, tracker, _, _ = make(broker_settings, fake, tmp_path)
    report = tracker.reconcile()
    assert report["adopted"] == ["SPY"] and report["dropped"] == ["QQQ"]
    spy = tracker.tracked["SPY"]
    assert spy.entry_time == buy.filled_at and spy.stop_level == 93.0 and spy.entry_price == 98.0
    assert set(json.loads((tmp_path / "positions.json").read_text())) == {"SPY"}


# ------------------------------------------------------------ market data
@pytest.fixture
def csv_md(settings, data, tmp_path):
    s = copy.deepcopy(settings)
    s["data"].update(source="csv", csv_dir=str(tmp_path))
    bars = data[0]["SPY"].iloc[:300].drop(index=data[0]["SPY"].index[100:105])   # a 1-week "halt"
    bars.to_csv(tmp_path / "SPY.csv")
    md = MarketData(s)
    md.cache_dir = tmp_path
    return md


def test_market_data_gaps_and_closures(csv_md):
    weekend = csv_md.history("SPY", start="2012-01-07", end="2012-01-08")      # Sat-Sun
    assert weekend.empty
    df = csv_md.history("SPY", start="2012-01-01")
    gaps = find_gaps(df)
    assert len(gaps) == 1 and gaps["days"].iloc[0] > 4
    assert csv_md.get_historical_bars("SPY", "1Day", start="2012-01-01").equals(df)


def test_market_data_latest_and_streams(csv_md):
    bar = csv_md.get_latest_bar("SPY")
    assert bar["symbol"] == "SPY" and bar["age_seconds"] > 0
    snap = csv_md.get_snapshot("SPY")
    assert snap["latest_trade"] == pytest.approx(bar["close"])
    assert csv_md.get_latest_quote("SPY") is None                  # no quotes without Alpaca: spread check skipped
    with pytest.raises(ValueError):
        csv_md.subscribe_quotes(["SPY"])
    with pytest.raises(ValueError):
        csv_md.subscribe_bars(["SPY"], "5Min", print)


def test_parse_timeframe():
    assert [str(parse_timeframe(t)) for t in ("1Min", "15Min", "1Hour", "1Day")] == ["1Min", "15Min", "1Hour", "1Day"]
    with pytest.raises(ValueError):
        parse_timeframe("3Fortnights")


def test_env_not_leaked():
    assert "ALPACA_API_KEY" not in os.environ or os.environ["ALPACA_API_KEY"] != "abc123"
