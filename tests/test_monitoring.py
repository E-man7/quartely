"""Logging channels and context, alert rate limiting, dashboard rendering."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta

import pytest
from rich.console import Console

from monitoring.alerts import AlertManager
from monitoring.dashboard import mark_stale, read_status, render, risk_bar, write_status
from monitoring.logger import SizeAndAgeRotatingHandler, get_context, set_context, setup_logging


@pytest.fixture
def log_dir(tmp_path):
    setup_logging({"log_dir": "logs", "log_level": "INFO"}, tmp_path)
    yield tmp_path / "logs"
    setup_logging({})                                    # back to console-only for other tests
    set_context(regime=None, probability=None, equity=None, positions=None, daily_pnl=None)


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_log_channels_and_context(log_dir):
    set_context(regime="BULL", probability=0.72, equity=105_230.0, positions={"SPY": 190}, daily_pnl=340.0)
    logging.getLogger("broker.order_executor").info("order submitted", extra={"symbol": "SPY"})
    logging.getLogger("broker.alpaca_client").info("stream connecting")        # connection noise: main.log only
    logging.getLogger("monitoring.alerts").warning("alert", extra={"event": "regime_change"})
    logging.getLogger("core.hmm_engine").info("regime change confirmed")
    logging.getLogger("core.risk_manager").info("risk modification")
    for h in logging.getLogger().handlers:
        h.flush()
    main = _lines(log_dir / "main.log")
    assert [r["msg"] for r in main] == ["order submitted", "stream connecting", "alert", "regime change confirmed",
                                        "risk modification"]
    assert all(r["regime"] == "BULL" and r["equity"] == 105_230.0 and r["daily_pnl"] == 340.0 for r in main)
    assert [r["msg"] for r in _lines(log_dir / "trades.log")] == ["order submitted"]
    assert [r["msg"] for r in _lines(log_dir / "alerts.log")] == ["alert"]
    assert [r["msg"] for r in _lines(log_dir / "regime.log")] == ["regime change confirmed"]
    assert get_context()["positions"] == {"SPY": 190}


def test_rotation_by_size_and_age(tmp_path):
    h = SizeAndAgeRotatingHandler(tmp_path / "main.log", max_bytes=200, max_age_days=30)
    h.setFormatter(logging.Formatter("%(message)s"))
    old = tmp_path / "main.log.20200101-000000-000000"
    old.write_text("old")
    os.utime(old, (time.time() - 31 * 86400,) * 2)
    rec = logging.makeLogRecord({"msg": "x" * 150})
    for _ in range(3):
        h.emit(rec)
    h.close()
    rotated = list(tmp_path.glob("main.log.*"))
    assert rotated and old not in rotated                 # rotated at 200 bytes, 31-day-old file pruned


def test_alert_rate_limit(capsys):
    now = [0.0]
    am = AlertManager({"monitoring": {"alerts": {"rate_limit_minutes": 15}}}, clock=lambda: now[0])
    assert am.send("regime_change", "BULL -> BEAR")
    now[0] += 60
    assert not am.send("regime_change", "BEAR -> BULL")   # same type within 15 min
    assert am.send("data_feed_down", "no bars")           # other types unaffected
    assert am.send("circuit_breaker", "daily_reduce", key="circuit_breaker:daily_reduce")
    assert am.send("circuit_breaker", "daily_halt", key="circuit_breaker:daily_halt")   # halt never swallowed
    now[0] += 15 * 60
    assert am.send("regime_change", "BULL -> NEUTRAL")
    assert "(+1 similar" in am.sent[-1][1]
    assert "ALERT [regime_change]" in capsys.readouterr().err


def test_alert_disabled_event():
    am = AlertManager({"monitoring": {"alerts": {"on": ["error"]}}}, console=False)
    assert not am.send("regime_change", "x") and am.send("error", "y")


STATUS = {
    "updated": datetime.now().isoformat(timespec="seconds"),
    "regime": {"label": "BULL", "probability": 0.72, "stability_bars": 14, "flicker_rate": 1, "flicker_window": 20,
               "confirmed": True, "flickering": False, "strategy": "LowVolBullStrategy"},
    "portfolio": {"equity": 105_230.0, "daily_pnl": 340.0, "daily_pnl_pct": 0.0032, "allocation": 0.95,
                  "leverage": 1.25},
    "positions": [{"symbol": "SPY", "side": "LONG", "price": 520.30, "pnl_pct": 0.012, "stop": 508.0,
                   "holding": "3h"}],
    "signals": [{"time": "14:30", "symbol": "SPY", "action": "Rebalance 60%→95%", "reason": "LowVolBull"}],
    "risk": {"daily_dd": 0.003, "daily_limit": 0.03, "weekly_dd": 0.0, "weekly_limit": 0.07, "peak_dd": 0.012,
             "peak_limit": 0.10, "breaker": "normal"},
    "system": {"data_ok": True, "api_ok": True, "api_latency_ms": 23.0, "hmm_age_days": 2, "mode": "PAPER"},
}


def test_dashboard_render():
    console = Console(record=True, width=110)
    console.print(render(STATUS))
    text = console.export_text()
    for expected in ("REGIME", "BULL (72%)", "Stability: 14 bars", "Flicker: 1/20", "PORTFOLIO", "$105,230",
                     "+$340", "Allocation: 95%", "Leverage: 1.25x", "POSITIONS", "Stop: $508.00", "3h",
                     "RECENT SIGNALS", "Rebalance 60%→95%", "RISK STATUS", "0.3%/3%", "1.2%/10%", "SYSTEM",
                     "23ms", "HMM: 2d ago", "PAPER"):
        assert expected in text, expected
    empty = Console(record=True, width=80)
    empty.print(render({}))
    assert "Waiting for a running instance" in empty.export_text()


def test_risk_bar_colours():
    assert risk_bar(0.003, 0.03).spans[0].style == "green"
    assert risk_bar(0.02, 0.03).spans[0].style == "yellow"
    assert risk_bar(0.029, 0.03).spans[0].style == "red"


def test_status_file_roundtrip_and_staleness(tmp_path):
    path = tmp_path / "status.json"
    write_status(path, STATUS)
    assert read_status(path)["regime"]["label"] == "BULL"
    assert read_status(tmp_path / "missing.json") == {}
    stale = mark_stale(read_status(path), now=datetime.now() + timedelta(hours=1))
    assert stale["system"]["data_ok"] is False and "stale" in stale["updated"]


# ------------------------------------------------------------ web dashboard
def test_web_dashboard_serves_page_and_status(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    from monitoring.web_dashboard import make_server

    status_file = tmp_path / "status.json"
    server = make_server(status_file, port=0)                  # any free port, 127.0.0.1 only
    assert server.server_address[0] == "127.0.0.1"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>Quartely Trader</title>" in page and "/api/status" in page
        empty = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert empty["status"] == {} and empty["history"] == []
        write_status(status_file, STATUS)
        data = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert data["status"]["regime"]["label"] == "BULL" and data["history"][-1][1] == 105_230.0
        urllib.request.urlopen(base + "/api/status")              # same update: no duplicate sample
        assert len(json.loads(urllib.request.urlopen(base + "/api/status").read())["history"]) == 1
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/../config/settings.yaml")
        assert err.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
