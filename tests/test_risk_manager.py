"""Risk manager: circuit breakers, position sizing, leverage, correlation, order validation."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from core.regime_strategies import Direction, Signal
from core.risk_manager import BreakerLevel, CircuitBreaker, RiskManager

BREAKERS = {"daily_reduce_dd": 0.02, "daily_halt_dd": 0.03, "weekly_reduce_dd": 0.05, "weekly_halt_dd": 0.07,
            "peak_halt_dd": 0.10, "size_mult": 0.5}
MON = pd.Timestamp("2024-01-08")   # a Monday


def day(n: int) -> pd.Timestamp:
    return MON + pd.Timedelta(days=n)


def signal(size=0.95, lev=1.25, entry=100.0, stop=97.0, strategy="LowVolBullStrategy", symbol="SPY", **meta):
    return Signal(symbol, Direction.LONG, 0.9, entry, stop, None, size, lev, 0, "BULL", 0.9, MON, "test", strategy,
                  {"atr_14": 1.5, **meta})


@pytest.fixture
def spec_settings(settings):
    """Phase 5 spec values (multi-stock profile)."""
    s = copy.deepcopy(settings)
    s["risk"].update(max_total_exposure=0.80, max_single_position=0.15, max_sector_exposure=0.30,
                     max_risk_per_trade=0.01, max_overnight_gap_loss=0.02, breakers=BREAKERS)
    return s


@pytest.fixture
def rm(settings, tmp_path):
    return RiskManager(settings, lock_file=tmp_path / "trading_halted.lock")


# -------------------------------------------------------------- breakers
def test_daily_breakers_reduce_then_halt_then_reset():
    cb = CircuitBreaker(BREAKERS)
    cb.update(100.0, day(0))
    ev = cb.update(97.9, day(0))
    assert [e.breaker for e in ev] == ["daily_reduce"] and cb.check().size_mult == 0.5
    ev = cb.update(96.9, day(0))
    assert [e.breaker for e in ev] == ["daily_halt"] and cb.check().flatten
    cb.update(96.9, day(1))                                  # new day: daily breaker resets
    assert cb.check().level == BreakerLevel.NORMAL


def test_weekly_breaker_lasts_the_week():
    cb = CircuitBreaker(BREAKERS)
    for d, equity in enumerate([100.0, 98.2, 96.43, 94.70]):  # -1.8%/day: no daily breaker; Thu weekly -5.3%
        cb.update(equity, day(d))
    events = [e.breaker for e in cb.get_history()]
    assert events == ["weekly_reduce"]
    cb.update(95.0, day(4))
    assert cb.check().size_mult == 0.5                       # still reduced on Friday
    cb.update(95.0, day(7))                                  # next Monday
    assert cb.check().level == BreakerLevel.NORMAL


def test_peak_lock_writes_file_and_needs_manual_reset(tmp_path):
    lock = tmp_path / "trading_halted.lock"
    cb = CircuitBreaker(BREAKERS, lock_file=lock)
    equity = [100, 102, 99, 97, 95, 93, 91.5]                 # -10.3% from the 102 peak over 2 weeks
    for d, e in enumerate(equity):
        cb.update(e, day(d * 2))
    assert lock.exists() and cb.check().level == BreakerLevel.LOCKED
    peak_event = [e for e in cb.get_history() if e.breaker == "peak_lock"][0]
    assert peak_event.drawdown == pytest.approx(91.5 / 102 - 1)
    for d in range(20, 40):                                   # stays locked however long we wait
        cb.update(95.0, day(d))
    assert cb.check().level == BreakerLevel.LOCKED
    lock.unlink()
    events = cb.update(95.0, day(41))
    assert [e.breaker for e in events] == ["lock_released"] and cb.peak == 95.0


def test_backtest_lock_resumes_after_cooldown():
    cb = CircuitBreaker(BREAKERS, lock_cooldown_bars=3)
    for d, e in enumerate([100, 95, 93, 89]):
        cb.update(e, day(d * 7))                              # weekly steps so only the peak breaker matters
    assert cb.check().level == BreakerLevel.LOCKED
    for d in range(1, 4):
        cb.update(89, day(30 + d))
    assert cb.check().level == BreakerLevel.LOCKED
    cb.update(89, day(34))
    assert cb.check().level == BreakerLevel.NORMAL and cb.peak == 89


def test_breaker_logs_hmm_context(settings):
    from core.hmm_engine import RegimeState

    cb = CircuitBreaker(BREAKERS)
    state = RegimeState("BULL", 2, 0.9, np.array([0.05, 0.05, 0.9]), MON, True, 10, np.array([0.3, 0.2, 0.1]))
    cb.update(100.0, day(0))
    (ev,) = cb.update(97.5, day(0), positions=["SPY"], regime=state, strategy="LowVolBullStrategy")
    assert (ev.hmm_regime, ev.hmm_wrong, ev.positions_closed) == ("BULL", True, ["SPY"])


def test_breaker_state_roundtrip():
    cb = CircuitBreaker(BREAKERS)
    cb.update(100.0, day(0))
    cb.update(97.5, day(0))
    restored = CircuitBreaker(BREAKERS)
    restored.load_state(cb.state_dict())
    assert restored.check().size_mult == 0.5 and restored.peak == 100.0


# ---------------------------------------------------------------- sizing
def test_allocation_profile_keeps_strategy_size(rm):
    state = rm.portfolio_state(100_000, {})
    d = rm.validate_signal(signal(), state)
    assert d.approved and d.modified_signal.target_weight == pytest.approx(0.95 * 1.25)


def test_spec_profile_sizing(spec_settings, tmp_path):
    rm = RiskManager(spec_settings, lock_file=tmp_path / "l.lock")
    d = rm.validate_signal(signal(stop=97.0), rm.portfolio_state(100_000, {}))
    # risk: 1% / 3% = 33%; overnight gap: 2% / (3 x 3%) = 22%; max single position 15%
    assert d.approved and d.modified_signal.target_weight == pytest.approx(0.15)
    assert any("max single position" in m for m in d.modifications)


def test_risk_per_trade_formula(spec_settings, tmp_path):
    s = copy.deepcopy(spec_settings)
    s["risk"].update(max_single_position=1.0, max_total_exposure=1.0, max_overnight_gap_loss=1.0)
    rm = RiskManager(s, lock_file=tmp_path / "l.lock")
    d = rm.validate_signal(signal(stop=95.0, lev=1.0), rm.portfolio_state(100_000, {}))
    shares_value = 100_000 * 0.01 / (100 - 95) * 100           # (portfolio * 1%) / |entry - stop| shares
    assert d.modified_signal.target_weight * 100_000 == pytest.approx(shares_value)


def test_rejects_missing_stop_and_tiny_positions(rm):
    state = rm.portfolio_state(100_000, {})
    assert rm.validate_signal(signal(stop=float("nan")), state).rejection_reason == "no stop loss"
    tiny = rm.validate_signal(signal(size=0.0005), state)
    assert not tiny.approved and "below minimum" in tiny.rejection_reason


def test_stop_above_entry_moved_below(rm):
    d = rm.validate_signal(signal(stop=101.0, strategy="MidVolCautiousStrategy", lev=1.0),
                           rm.portfolio_state(100_000, {}))
    assert d.approved and d.modified_signal.stop_loss == pytest.approx(100 - 1.5)


def test_breaker_vetoes_signal(rm):
    rm.process(None, 100_000, {}, "SPY", "SHY", day(0))
    out = rm.process(signal(), 96_500, {"SPY": 1.0}, "SPY", "SHY", day(0))
    assert out.status.flatten and out.weights == {"SPY": 0.0, "SHY": 0.0}
    assert not out.decision.approved and "circuit breaker" in out.decision.rejection_reason


def test_breaker_reduce_halves_size_and_removes_leverage(rm):
    rm.process(None, 100_000, {}, "SPY", "SHY", day(0))
    out = rm.process(signal(), 97_800, {"SPY": 1.0}, "SPY", "SHY", day(0))
    assert out.weights["SPY"] == pytest.approx(0.95 * 1.0 * 0.5) and out.weights["SHY"] == pytest.approx(1 - 0.475)


# -------------------------------------------------------------- leverage
@pytest.mark.parametrize("kw, state_kw, levered", [
    ({}, {}, True),
    ({"strategy": "MidVolCautiousStrategy"}, {}, False),
    ({"uncertain": True}, {}, False),
    ({}, {"flicker_rate": 6}, False),
    ({}, {"weights": {"QQQ": 0.1, "IWM": 0.1}}, False),        # this would be the 3rd position
])
def test_leverage_rules(rm, kw, state_kw, levered):
    weights = state_kw.pop("weights", {})
    state = rm.portfolio_state(100_000, weights, **state_kw)
    d = rm.validate_signal(signal(**kw), state)
    assert d.modified_signal.leverage == (1.25 if levered else 1.0)


# ----------------------------------------------------------- correlation
def _history(rho: float, n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    a = rng.normal(0, 0.01, n)
    b = rho * a + np.sqrt(1 - rho**2) * rng.normal(0, 0.01, n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"SPY": 100 * np.exp(np.cumsum(a)), "QQQ": 100 * np.exp(np.cumsum(b))}, index=idx)


@pytest.mark.parametrize("rho, approved, halved", [(0.2, True, False), (0.78, True, True), (0.95, False, False)])
def test_correlation_check(rm, rho, approved, halved):
    state = rm.portfolio_state(100_000, {"QQQ": 0.2}, price_history=_history(rho))
    d = rm.validate_signal(signal(lev=1.0, strategy="MidVolCautiousStrategy"), state)
    assert d.approved == approved
    if approved:
        assert (d.modified_signal.target_weight == pytest.approx(0.475)) == halved


def test_position_and_trade_count_limits(rm):
    weights = {s: 0.1 for s in ("A", "B", "C", "D", "E")}
    assert "concurrent" in rm.validate_signal(signal(), rm.portfolio_state(100_000, weights)).rejection_reason
    rm.record_trades(20, MON)
    assert "daily trades" in rm.validate_signal(signal(), rm.portfolio_state(100_000, {})).rejection_reason


# ------------------------------------------------------- order validation
def test_order_validation(rm):
    state = rm.portfolio_state(10_000, {}, buying_power=5_000)
    assert "buying power" in rm.validate_order("SPY", "buy", 6_000, state).reason
    assert "not tradable" in rm.validate_order("XYZ", "buy", 100, state, tradable=False).reason
    assert "spread" in rm.validate_order("SPY", "buy", 100, state, quote=(99.0, 100.0)).reason
    assert rm.validate_order("SPY", "buy", 100, state, quote=(99.99, 100.01), now=0.0).approved
    assert "duplicate" in rm.validate_order("SPY", "buy", 100, state, now=30.0).reason
    assert rm.validate_order("SPY", "buy", 100, state, now=61.0).approved
    assert rm.validate_order("SPY", "sell", 100, state, now=62.0).approved   # other direction is fine
    assert len(rm.rejections) == 4
