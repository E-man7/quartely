"""Main loop orchestration against the fake broker and CSV data (no network, no real orders)."""
from __future__ import annotations

import copy
import json
import logging
import threading
import time

import pandas as pd
import pytest

import main as main_mod
from broker.alpaca_client import AlpacaClient
from core.trading_system import BarAggregator, TradingSystem, timeframe_minutes
from data.market_data import MarketData
from monitoring.alerts import AlertManager
from test_broker import CREDS, FakeTradingClient


class FakeStream:
    def __init__(self):
        self._stop = threading.Event()

    def subscribe_trade_updates(self, handler):
        self.handler = handler

    def run(self):
        self._stop.wait()

    def stop(self):
        self._stop.set()


@pytest.fixture
def live_settings(settings, data, tmp_path):
    s = copy.deepcopy(settings)
    for sym, df in data[0].items():
        df.to_csv(tmp_path / f"{sym}.csv")
    s["data"].update(source="csv", csv_dir=str(tmp_path))
    s["hmm"].update(model_dir=str(tmp_path / "models"), train_window=None)
    s["broker"].update(paper_trading=True, dry_run=False, fractional=False)
    s["risk"]["lock_file"] = str(tmp_path / "trading_halted.lock")
    s["execution"].update(state_file=str(tmp_path / "state_snapshot.json"), when_closed="wait")
    s["monitoring"].update(log_dir=str(tmp_path / "logs"), status_file=str(tmp_path / "status.json"))
    return s


def build(settings, tmp_path, fake=None, dry_run=False):
    fake = fake or FakeTradingClient()
    md = MarketData(settings)
    md.cache_dir = tmp_path
    fake.price = float(md.history("SPY")["close"].iloc[-1])
    client = AlpacaClient(settings, CREDS, client_factory=lambda: fake)
    client.trading_stream = FakeStream
    alerts = AlertManager(settings, console=False)
    system = TradingSystem(settings, CREDS, dry_run=dry_run, client=client, market_data=md, alerts=alerts)
    return system, fake


@pytest.fixture
def started(live_settings, tmp_path):
    system, fake = build(live_settings, tmp_path)
    assert system.startup()
    yield system, fake
    system.shutdown()


def after_run_time(system) -> pd.Timestamp:
    return pd.Timestamp.now().normalize() + pd.Timedelta(hours=15, minutes=50)


# --------------------------------------------------------------- startup
def test_startup_sequence(started, live_settings, tmp_path, caplog):
    system, fake = started
    assert system.api_ok and system.sig.hmm.model is not None
    assert list((tmp_path / "models").glob("hmm_*.pkl"))                    # trained and saved
    assert any(e == "hmm_retrained" for e, _ in system.alerts.sent)
    assert system.sig.risk.breaker.equity == pytest.approx(system.tracker.equity)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["system"]["mode"] == "PAPER" and status["portfolio"]["equity"] == pytest.approx(100_000)


def test_startup_reuses_fresh_model(live_settings, tmp_path, monkeypatch):
    first, _ = build(live_settings, tmp_path)
    assert first.startup()
    first.shutdown()
    second, _ = build(live_settings, tmp_path)
    fitted = first.sig.hmm.fitted_at
    monkeypatch.setattr(second, "now", lambda: pd.Timestamp(fitted) + pd.Timedelta(days=2, hours=10))
    assert second.startup()
    assert second.sig.hmm.metadata()["training_date"] == str(fitted.date())
    assert not any(e == "hmm_retrained" for e, _ in second.alerts.sent)     # loaded, not retrained
    second.shutdown()


def test_market_closed_exit(live_settings, tmp_path):
    s = copy.deepcopy(live_settings)
    s["execution"]["when_closed"] = "exit"
    system, fake = build(s, tmp_path)
    fake.get_clock = lambda: type("C", (), {"is_open": False, "next_open": None, "next_close": None,
                                            "timestamp": None})()
    assert system.startup() is False


def test_snapshot_recovery(live_settings, tmp_path):
    (tmp_path / "state_snapshot.json").write_text(json.dumps({
        "saved_at": "2024-01-08", "last_decision": "2024-01-08 15:50",
        "risk": {"breaker": {"peak": 150_000.0, "equity": 140_000.0, "day_start": 140_000.0, "week_start": 140_000.0,
                             "day": "2024-01-08", "week": [2024, 2], "daily_level": 0, "weekly_level": 0,
                             "locked": False}}}))
    system, _ = build(live_settings, tmp_path)
    assert system.startup()
    b = system.sig.risk.breaker
    assert b.peak == 150_000.0 and b.check().level.name == "LOCKED"     # 100k vs 150k peak: -33%
    assert system.last_decision == pd.Timestamp("2024-01-08 15:50")
    system.shutdown()


# --------------------------------------------------------------- the loop
def test_decision_submits_orders(started):
    system, fake = started
    system.decide(after_run_time(system))
    assert system.last_signal is not None and system.session["decisions"] == 1
    if system.last_signal.rebalance:
        assert fake.submitted and system.signals[0]["action"].startswith("Rebalance")
    snap = json.loads(system.snapshot_path.read_text())
    assert snap["last_signal"]["regime"] == system.last_signal.regime.label


def test_dry_run_sends_nothing(live_settings, tmp_path):
    system, fake = build(live_settings, tmp_path, dry_run=True)
    assert system.startup()
    system.decide(after_run_time(system))
    assert system.last_signal is not None and not fake.submitted
    assert system.status()["system"]["mode"] == "DRY-RUN"
    system.shutdown()


def test_on_bar_decides_once_per_day(started, monkeypatch):
    system, _ = started
    t = after_run_time(system)
    monkeypatch.setattr(system, "now", lambda: t)
    bar = {"symbol": "SPY", "close": 100.0, "timestamp": t}
    system.on_bar(bar)
    system.on_bar(bar)
    assert system.session["decisions"] == 1 and system.session["bars"] == 2
    monkeypatch.setattr(system, "now", lambda: t.normalize() + pd.Timedelta(hours=10))
    system.last_decision = None
    system.on_bar(bar)                                    # before run_time: no decision
    assert system.session["decisions"] == 1


def test_feed_down_pauses_signals(started):
    system, _ = started
    system.streaming, system.last_bar_at = True, time.time() - 10_000
    system.check_feed()
    assert system.feed_down and any(e == "data_feed_down" for e, _ in system.alerts.sent)
    system.decide(after_run_time(system))
    assert system.last_signal is None                     # paused
    system.on_bar({"symbol": "SPY", "close": 100.0, "timestamp": after_run_time(system)})
    assert not system.feed_down


def test_hmm_error_holds_regime(started, monkeypatch):
    system, _ = started
    t = after_run_time(system)
    system.decide(t)
    held = system.last_signal.regime.label

    def boom(*a, **k):
        raise RuntimeError("covariance not positive definite")

    monkeypatch.setattr(system.sig.hmm, "current_regime", boom)
    system.decide(t + pd.Timedelta(days=1))
    assert system.last_signal.regime.label == held and system.sig.hmm_error
    assert any(e == "error" for e, _ in system.alerts.sent)


def test_breaker_halt_closes_everything(started):
    system, fake = started
    system.decide(after_run_time(system))
    fake.positions.setdefault("SPY", {"qty": 900.0, "avg": fake.price})
    fake.price *= 0.94                                    # -6% on a ~100% position: daily halt
    system.on_bar({"symbol": "SPY", "close": fake.price, "timestamp": after_run_time(system)})
    assert getattr(fake, "closed_all", False)
    assert any(e == "circuit_breaker" for e, _ in system.alerts.sent)


def test_trailing_stop_only_tightens(started):
    system, fake = started
    system.decide(after_run_time(system))
    if system.tracker.qty("SPY") <= 0:
        pytest.skip("strategy chose no SPY position on this data")
    from core.regime_strategies import LowVolBullStrategy

    system.stop_inputs = {"strategy": LowVolBullStrategy(), "ema": fake.price * 0.97, "atr": fake.price * 0.01}
    before = len(fake.replaced) + len(fake.submitted)
    system.update_trailing_stop(fake.price * 1.05)       # price up: stop trails up
    assert len(fake.replaced) + len(fake.submitted) > before
    after = len(fake.replaced) + len(fake.submitted)
    system.update_trailing_stop(fake.price * 0.98)       # price down: stop stays
    assert len(fake.replaced) + len(fake.submitted) == after


def test_shutdown_keeps_positions(started, caplog):
    system, fake = started
    system.decide(after_run_time(system))
    with caplog.at_level(logging.INFO):
        summary = system.shutdown()
    assert not getattr(fake, "closed_all", False)
    assert {"start_equity", "end_equity", "session_pnl", "decisions"} <= set(summary)
    assert json.loads(system.snapshot_path.read_text())["risk"]["breaker"]["peak"] is not None


# ------------------------------------------------------------ components
def test_bar_aggregator():
    out = []
    agg = BarAggregator(5, out.append)
    t0 = pd.Timestamp("2024-01-08 09:30", tz="America/New_York")
    for i, px in enumerate([100, 101, 99, 102, 103, 104]):
        agg.on_minute_bar({"symbol": "SPY", "timestamp": t0 + pd.Timedelta(minutes=i), "open": px, "high": px + 0.5,
                           "low": px - 0.5, "close": px, "volume": 10})
    (bar,) = out                                          # 09:30-09:34 complete; 09:35 still open
    assert (bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]) == (100, 103.5, 98.5, 103, 50)
    assert bar["timestamp"] == t0 + pd.Timedelta(minutes=5)
    agg.on_minute_bar({"symbol": "SPY", "timestamp": t0 + pd.Timedelta(minutes=12), "open": 1, "high": 1,
                       "low": 1, "close": 1, "volume": 1})
    assert len(out) == 2 and out[1]["open"] == 104        # gap in minutes still closes the bucket


def test_timeframe_minutes():
    assert [timeframe_minutes(t) for t in ("1Min", "5Min", "1Hour", "1Day")] == [1, 5, 60, 390]


@pytest.mark.parametrize("argv, expected", [
    ([], "cmd_live"), (["--dry-run"], "cmd_live"), (["--backtest"], "cmd_backtest"),
    (["backtest", "--symbols", "SPY"], "cmd_backtest"), (["--compare"], "cmd_backtest"),
    (["--stress-test"], "cmd_backtest"), (["--train-only"], "cmd_train"), (["--dashboard"], "cmd_dashboard"),
    (["regimes"], "cmd_regimes"),
])
def test_cli_routing(monkeypatch, argv, expected):
    called = []
    for name in ("cmd_live", "cmd_backtest", "cmd_train", "cmd_dashboard", "cmd_regimes"):
        monkeypatch.setattr(main_mod, name, lambda s, c, a, name=name: called.append(name))
    monkeypatch.setattr(main_mod, "setup_logging", lambda *a, **k: None)
    main_mod.main(argv)
    assert called == [expected]
