"""Volatility-based allocation: rank mapping, sizing, stops, uncertainty, rebalancing."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import REGIME_LABELS, RegimeInfo, RegimeState
from core.regime_strategies import (
    LABEL_TO_STRATEGY,
    UNCERTAINTY_NOTE,
    BearTrendStrategy,
    BullTrendStrategy,
    CrashDefensiveStrategy,
    Direction,
    EuphoriaCautiousStrategy,
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MeanReversionStrategy,
    MidVolCautiousStrategy,
    StrategyOrchestrator,
    strategy_for_vol_rank,
)
from core.signal_generator import SignalGenerator
from data.feature_engineering import atr

Low, Mid, High = LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy


def infos(vols, labels=None):
    labels = labels or REGIME_LABELS[len(vols)]
    return [RegimeInfo(k, labels[k], 0.0, v, "", 1.0, 1.0, 0.55) for k, v in enumerate(vols)]


def state(state_id, n, prob=0.9, confirmed=True, flickering=False, vols=None):
    p = np.full(n, (1 - prob) / (n - 1))
    p[state_id] = prob
    return RegimeState(REGIME_LABELS[n][state_id], state_id, prob, p, pd.Timestamp("2020-01-02"),
                       confirmed, 10, np.asarray(vols if vols is not None else np.arange(n), float), flickering)


def trend_bars(direction: float, n: int = 300) -> pd.DataFrame:
    """Steady trend (+1 up, -1 down) so price ends clearly above/below its 50 EMA."""
    close = 100 * np.exp(direction * 0.002 * np.arange(n))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1e6}, index=pd.bdate_range("2019-01-01", periods=n))


@pytest.mark.parametrize("n, expected", [
    (3, [Low, Mid, High]),
    (4, [Low, Mid, Mid, High]),
    (5, [Low, Low, Mid, High, High]),
    (6, [Low, Low, Mid, Mid, High, High]),
    (7, [Low, Low, Mid, Mid, Mid, High, High]),
])
def test_vol_rank_mapping(n, expected):
    assert [strategy_for_vol_rank(r, n) for r in range(n)] == expected


def test_orchestrator_ignores_labels(settings):
    # Return-sorted labels BEAR, NEUTRAL, BULL, but BULL is the most volatile
    orch = StrategyOrchestrator(settings, infos([0.15, 0.10, 0.30]))
    assert orch.vol_rank == {1: 0, 0: 1, 2: 2}
    assert type(orch.strategy_for(2)) is High and type(orch.strategy_for(1)) is Low


def test_update_regime_infos_rebuilds(settings):
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    orch.update_regime_infos(infos([0.3, 0.2, 0.1, 0.25]))
    assert type(orch.strategy_for(2)) is Low and type(orch.strategy_for(0)) is High
    with pytest.raises(ValueError):
        orch.generate_signals(["SPY"], {"SPY": trend_bars(1)}, state(0, 3))


def test_low_vol_sizing_and_stop(settings):
    bars = trend_bars(1)
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    (sig,) = orch.generate_signals(["SPY"], {"SPY": bars}, state(0, 3))
    assert (sig.direction, sig.position_size_pct, sig.leverage) == (Direction.LONG, 0.95, 1.25)
    assert sig.target_weight == pytest.approx(0.95 * 1.25)
    price, a = bars["close"].iat[-1], atr(bars.iloc[-250:]).iat[-1]
    ema = bars["close"].iloc[-250:].ewm(span=50, adjust=False).mean().iat[-1]
    assert sig.stop_loss == pytest.approx(max(price - 3 * a, ema - 0.5 * a))
    assert sig.take_profit is None and sig.strategy_name == "LowVolBullStrategy"


@pytest.mark.parametrize("direction, size", [(1, 0.95), (-1, 0.60)])
def test_mid_vol_follows_trend(settings, direction, size):
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    (sig,) = orch.generate_signals(["SPY"], {"SPY": trend_bars(direction)}, state(1, 3))
    assert (sig.position_size_pct, sig.leverage) == (size, 1.0)
    assert sig.stop_loss == pytest.approx(sig.metadata["ema_50"] - 0.5 * sig.metadata["atr_14"])


def test_high_vol_stays_long(settings):
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    (sig,) = orch.generate_signals(["SPY"], {"SPY": trend_bars(-1)}, state(2, 3))
    assert (sig.direction, sig.position_size_pct, sig.leverage) == (Direction.LONG, 0.60, 1.0)
    assert sig.stop_loss == pytest.approx(sig.metadata["ema_50"] - 1.0 * sig.metadata["atr_14"])


@pytest.mark.parametrize("kw", [{"prob": 0.5}, {"flickering": True}])
def test_uncertainty_halves_size_and_removes_leverage(settings, kw):
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    (sig,) = orch.generate_signals(["SPY"], {"SPY": trend_bars(1)}, state(0, 3, **kw))
    assert (sig.position_size_pct, sig.leverage) == (pytest.approx(0.475), 1.0)
    assert sig.reasoning.endswith(UNCERTAINTY_NOTE)


def test_flicker_argument_overrides_state(settings):
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    (sig,) = orch.generate_signals(["SPY"], {"SPY": trend_bars(1)}, state(0, 3), is_flickering=True)
    assert sig.leverage == 1.0 and UNCERTAINTY_NOTE in sig.reasoning


def test_pending_confirmation_cuts_size(settings):
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    (sig,) = orch.generate_signals(["SPY"], {"SPY": trend_bars(1)}, state(0, 3, confirmed=False))
    assert (sig.position_size_pct, sig.leverage) == (pytest.approx(0.95 * 0.75), 1.25)


def test_never_short(settings, data):
    bars = data[0]["SPY"]
    orch = StrategyOrchestrator(settings, infos([0.1, 0.15, 0.2, 0.3, 0.5]))
    for t in range(60, len(bars), 37):
        for k in range(5):
            for kw in ({}, {"prob": 0.3}, {"flickering": True}, {"confirmed": False}):
                (sig,) = orch.generate_signals(["SPY"], {"SPY": bars.iloc[:t]}, state(k, 5, **kw))
                assert sig.direction is Direction.LONG and 0 < sig.target_weight <= 0.95 * 1.25


def test_insufficient_history_gives_no_signal(settings):
    orch = StrategyOrchestrator(settings, infos([0.1, 0.2, 0.3]))
    assert orch.generate_signals(["SPY"], {"SPY": trend_bars(1, n=30)}, state(0, 3)) == []


def test_aliases_and_label_map():
    assert CrashDefensiveStrategy is BearTrendStrategy is High
    assert MeanReversionStrategy is Mid
    assert BullTrendStrategy is EuphoriaCautiousStrategy is Low
    all_labels = {label for labels in REGIME_LABELS.values() for label in labels}
    assert set(LABEL_TO_STRATEGY) == all_labels


def test_rebalance_only_beyond_threshold(settings):
    s = copy.deepcopy(settings)
    gen = SignalGenerator(s)
    gen.strategy.update_regime_infos(infos([0.1, 0.2, 0.3]))
    bars = trend_bars(-1)
    # High-vol target: 60% SPY / 40% SHY
    sig = gen.from_regime(state(2, 3), bars, current_weights={"SPY": 0.55, "SHY": 0.45})
    assert sig.target_weights["SPY"] == pytest.approx(0.60) and not sig.rebalance
    gen._last_period = None
    sig = gen.from_regime(state(2, 3), bars, current_weights={"SPY": 0.45, "SHY": 0.55})
    assert sig.rebalance


def test_fast_indicators_match_pandas(data):
    from core.regime_strategies import ema_atr

    bars = data[0]["SPY"].iloc[100:350]
    ema, a = ema_atr(bars, 50, 14)
    assert ema == pytest.approx(bars["close"].ewm(span=50, adjust=False).mean().iat[-1], rel=1e-12)
    assert a == pytest.approx(atr(bars, 14).iat[-1], rel=1e-12)
