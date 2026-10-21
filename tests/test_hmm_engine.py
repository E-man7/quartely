"""HMM engine: model selection, labeling, stability filter, persistence."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from core.hmm_engine import REGIME_LABELS, HMMRegimeEngine, stability_filter
from data.feature_engineering import HMM_FEATURES, build_features


@pytest.fixture(scope="module")
def feats(settings, data):
    return build_features(data[0]["SPY"], settings["features"]).dropna()


@pytest.fixture(scope="module")
def engine(settings, feats):
    return HMMRegimeEngine(settings).fit(feats)


def test_features_complete(feats):
    assert set(HMM_FEATURES) <= set(feats.columns)
    z = feats[HMM_FEATURES]
    assert np.isfinite(z.values).all()
    assert (z.mean().abs() < 1.0).all()


def test_bic_selects_minimum(settings, engine):
    assert set(engine.bic_scores) == set(settings["hmm"]["n_states_candidates"])
    assert engine.n_states == min(engine.bic_scores, key=engine.bic_scores.get)
    assert engine.bic == pytest.approx(engine.bic_scores[engine.n_states])


def test_labels_sorted_by_return(engine):
    assert engine.labels == REGIME_LABELS[engine.n_states]
    returns = [i.expected_return for i in engine.regime_info]
    assert returns == sorted(returns)
    assert np.allclose(engine.get_transition_matrix().sum(axis=1), 1.0)


def test_metadata_follows_volatility(engine):
    by_vol = sorted(engine.regime_info, key=lambda i: i.expected_volatility)
    sizes = [i.max_position_size_pct for i in by_vol]
    assert sizes == sorted(sizes, reverse=True)       # calmest regime gets the largest allowance
    assert by_vol[0].recommended_strategy_type == "LowVolBullStrategy"
    assert by_vol[-1].recommended_strategy_type == "HighVolDefensiveStrategy"


def test_stability_filter_confirms_after_n_bars():
    raw = np.array([0] * 5 + [1] * 2 + [0] * 3 + [1] * 4)
    st = stability_filter(raw, confirm_bars=3, flicker_window=20, flicker_threshold=10)
    assert st.regime.tolist() == [0] * 12 + [1] * 2       # 2-bar blip ignored, 4-bar move confirmed on bar 3
    assert st.is_confirmed.tolist() == [True] * 5 + [False] * 2 + [True] * 3 + [False] * 2 + [True] * 2
    assert st.changed.tolist().index(True) == 12
    assert st.consecutive_bars.iat[12] == 3 and st.consecutive_bars.iat[13] == 4


def test_flickering_blocks_confirmation():
    raw = np.array([0, 1] * 4 + [1] * 5)                 # 7 switches in the first 8 bars
    st = stability_filter(raw, confirm_bars=3, flicker_window=20, flicker_threshold=4)
    assert st.flickering.iloc[-1] and not st.is_confirmed.iloc[-1]
    assert (st.regime == 0).all()                         # held through uncertainty


def test_stateful_queries(engine, feats):
    state = engine.current_regime(feats)
    assert engine.get_regime_stability() == state.consecutive_bars >= 1
    assert engine.is_flickering() == state.is_flickering
    assert isinstance(engine.detect_regime_change(), bool)
    assert 0 <= engine.get_regime_flicker_rate() <= engine.flicker_window
    assert state.probability == pytest.approx(state.state_probabilities[state.state_id])


def test_regime_change_logging(settings, feats, caplog):
    eng = HMMRegimeEngine(settings).fit(feats.iloc[:700])
    eng.current_regime(feats.iloc[:700])
    with caplog.at_level(logging.INFO, logger="core.hmm_engine"):
        eng.regime_states(feats)
    changes = [r for r in caplog.records if r.getMessage() == "regime change detected"]
    confirms = [r for r in caplog.records if r.getMessage() == "regime change confirmed"]
    assert changes and all(r.levelno == logging.WARNING for r in changes)
    assert confirms and all(r.levelno == logging.INFO for r in confirms)


def test_save_load_roundtrip(settings, engine, feats, tmp_path):
    path = engine.save(tmp_path / "hmm.pkl")
    loaded = HMMRegimeEngine.load(path, settings)
    meta = loaded.metadata()
    assert meta["n_regimes"] == engine.n_states and meta["labels"] == engine.labels
    assert meta["training_date"] == str(feats.index[-1].date())
    np.testing.assert_allclose(loaded.predict_regime_proba(feats), engine.predict_regime_proba(feats))
