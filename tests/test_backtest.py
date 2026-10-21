"""Walk-forward backtester, allocation math, performance report and stress tests."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from backtest import (
    SimConfig,
    StressTester,
    WalkForwardBacktester,
    build_report,
    save_report,
    simulate_allocation,
    stress_summary,
    trade_log,
)
from backtest.performance import (
    drawdown_episodes,
    longest_underwater,
    max_consecutive_losses,
    max_drawdown_duration,
    trade_stats,
)


# --------------------------------------------------------- allocation math
def _prices(opens, closes):
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame({"SPY": opens}, index=idx), pd.DataFrame({"SPY": closes}, index=idx)


def test_allocation_math_with_margin():
    opens, closes = _prices([100, 105, 115, 118], [100, 110, 120, 100])
    cfg = SimConfig(initial_capital=10_000, slippage=0.001, fill_delay=1)
    sim = simulate_allocation(opens, closes, lambda t, ts, eq, w: ({"SPY": 1.25} if t == 0 else None, {}), cfg)

    assert sim.shares["SPY"].tolist() == [0, 119, 119, 119]          # signal at bar 0 fills at bar 1 open
    target_shares = int(10_000 * 1.25 / 105)                          # equity at the fill bar's open
    assert target_shares == 119
    assert sim.cash.iloc[1] == pytest.approx(10_000 - 119 * 105 * 1.001)
    assert sim.cash.iloc[1] < 0                                       # leverage = margin debt
    np.testing.assert_allclose(sim.equity, sim.cash + sim.shares["SPY"] * closes["SPY"])
    assert sim.fills.iloc[0]["cost"] == pytest.approx(119 * 105 * 0.001)


def test_newer_signal_supersedes_pending_and_sells():
    opens, closes = _prices([100, 100, 100, 100], [100, 100, 100, 100])
    targets = {0: {"SPY": 1.0}, 2: {"SPY": 0.5}}
    sim = simulate_allocation(opens, closes, lambda t, ts, eq, w: (targets.get(t), {}), SimConfig(10_000, 0.0))
    assert sim.shares["SPY"].tolist() == [0, 100, 100, 50]
    assert sim.fills["side"].tolist() == ["BUY", "SELL"]
    trades = trade_log(sim)
    assert trades["holding_bars"].tolist() == [2, 1]
    assert trades["pnl"].tolist() == [0.0, 0.0]


def test_margin_interest_charged_on_negative_cash():
    opens, closes = _prices([100] * 3, [100] * 3)
    cfg = SimConfig(10_000, 0.0, margin_rate=0.10, cash_rate=0.0)
    sim = simulate_allocation(opens, closes, lambda t, ts, eq, w: ({"SPY": 1.5} if t == 0 else None, {}), cfg)
    debt = 10_000 - 150 * 100
    assert sim.cash.iloc[1] == pytest.approx(debt * (1.10 ** (1 / 252)))


# ------------------------------------------------------------- metrics
def test_drawdown_and_streak_metrics():
    r = pd.Series([0.1, -0.1, -0.05, 0.2, -0.01, 0.0, 0.05],
                  index=pd.bdate_range("2020-01-01", periods=7))
    # wealth: 1.1, .99, .9405, 1.1286, 1.1173, 1.1173, 1.1732 -> two 2-bar episodes, both recovered
    ep = drawdown_episodes(r)
    assert ep["duration_bars"].tolist() == [2, 2] and ep["recovered"].all()
    assert ep["depth"].iloc[0] == pytest.approx(0.9405 / 1.1 - 1)
    assert max_consecutive_losses(r) == 2
    assert max_drawdown_duration(r) == longest_underwater(r) == 2


def test_trade_stats():
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 25.0], "pnl_pct": [0.01, -0.005, 0.0025],
                           "holding_bars": [10, 5, 3]})
    st = trade_stats(trades)
    assert st["total_trades"] == 3 and st["win_rate"] == pytest.approx(2 / 3)
    assert st["profit_factor"] == pytest.approx(125 / 50) and st["avg_holding_bars"] == 6


# ---------------------------------------------------------- walk-forward
@pytest.fixture(scope="module")
def bt_settings(settings):
    s = copy.deepcopy(settings)
    s["stress_test"].update(n_simulations=3)
    s["stress_test"]["misclassification"] = {"n_simulations": 2}
    return s


@pytest.fixture(scope="module")
def backtest(bt_settings, data):
    bt = WalkForwardBacktester(bt_settings)
    result = bt.run(data[0])
    bt.run_benchmarks(data[0], result, n_random=5)
    return bt, result


def test_walk_forward_windows(bt_settings, backtest):
    _, result = backtest
    wf = bt_settings["backtest"]["walk_forward"]
    w = result.windows
    assert all(x.test_start - x.train_start == wf["in_sample"] for x in w)
    assert all(b.test_start - a.test_start == wf["step"] for a, b in zip(w, w[1:]))
    assert result.regimes["window"].nunique() == len(w)


def test_backtest_invariants(bt_settings, backtest):
    _, result = backtest
    sim = result.sim
    assert (sim.equity > 0).all() and len(result.trades) > 0
    assert (sim.shares >= 0).all().all()                                # always long
    gross = sim.weights.abs().sum(axis=1)
    assert gross.max() <= bt_settings["risk"]["max_portfolio_leverage"] * 1.05
    # Every rebalance moved some allocation by more than the threshold, or was forced by a risk event
    threshold = bt_settings["strategy"]["rebalance_threshold"]
    trades = result.trades
    syms = [c[len("target_w_"):] for c in trades.columns if c.startswith("target_w_")]
    drift = pd.concat([(trades[f"target_w_{s}"] - trades[f"prev_w_{s}"]).abs() for s in syms], axis=1).max(axis=1)
    forced = trades["events"].astype(bool)
    assert ((drift > threshold * 0.9) | forced).all()                  # 0.9: weights drift overnight before the fill


def test_backtest_has_no_look_ahead(bt_settings, data):
    bars, _ = data
    full = WalkForwardBacktester(bt_settings).run(bars)
    cut = full.equity.index[len(full.equity) // 2]
    short = WalkForwardBacktester(bt_settings).run({s: df.loc[:cut] for s, df in bars.items()})
    pd.testing.assert_series_equal(short.equity, full.equity.loc[:cut])
    cols = ["regime_id", "confidence", "strategy", "target_allocation"]
    pd.testing.assert_frame_equal(short.regimes[cols], full.regimes.loc[:cut, cols])


def test_report_tables(backtest, tmp_path):
    _, result = backtest
    report = build_report(result)
    assert set(report["confidence"].index) == {"< 50%", "50-60%", "60-70%", "70%+"}
    assert report["regimes"]["pct_time"].sum() == pytest.approx(1.0, abs=0.01)
    bench = report["benchmarks"]
    assert {"Buy & Hold", "200 SMA Trend", "Random x5 (mean)", "Random x5 (std)"} <= set(bench.index)
    # Vol-based sizing should cut the drawdown versus buy-and-hold on the synthetic regime market
    assert bench.iloc[0]["max_drawdown"] > bench.loc["Buy & Hold", "max_drawdown"]
    names = {p.name for p in save_report(result, report, tmp_path)}
    assert {"equity_curve.csv", "trade_log.csv", "regime_history.csv", "benchmark_comparison.csv"} <= names


def test_stress_tests(bt_settings, data, backtest):
    _, result = backtest
    tester = StressTester(bt_settings)
    out = tester.run(result, data[0])
    assert len(out["crash"]) == 3 and (out["crash"]["shocks"] == 10).all()
    assert len(out["gap"]) == 3 and (out["gap"]["gap_pct"] < 0).all()
    assert len(out["misclassification"]) == 2
    summary = stress_summary(out, result, tester.containment_limit)
    assert set(summary) == {"Crash injection", "Gap risk", "Regime misclassification"}
