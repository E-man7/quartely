"""Look-ahead bias checks: nothing at bar T may change when data after T is appended."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import HMMRegimeEngine
from data.feature_engineering import build_features


@pytest.fixture(scope="module")
def feats(settings, data):
    return build_features(data[0]["SPY"], settings["features"]).dropna()


@pytest.fixture(scope="module")
def model(settings, feats):
    return HMMRegimeEngine(settings).fit(feats)


def test_no_look_ahead_bias(model, feats):
    """Regime at T must be identical with data[0:T] vs data[0:T+100]."""
    T = 400
    regime_short = model.predict_regime_filtered(feats.iloc[:T])[-1]
    regime_long = model.predict_regime_filtered(feats.iloc[: T + 100])[T - 1]
    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"


def test_filtered_probabilities_unchanged_by_future(model, feats):
    # Clear the forward-pass cache so both calls run a full pass
    model._cache = None
    short = model.predict_regime_proba(feats.iloc[:900])
    model._cache = None
    long = model.predict_regime_proba(feats.iloc[:1000])
    pd.testing.assert_frame_equal(short, long.iloc[:900], check_exact=False, rtol=0, atol=1e-12)


def test_forward_cache_matches_fresh_pass(settings, model, feats):
    model._cache = None
    step_by_step = [model.predict_regime_proba(feats.iloc[:t]).iloc[-1].values for t in range(800, 820)]
    model._cache = None
    full = model.predict_regime_proba(feats.iloc[:819]).values[799:819]
    np.testing.assert_allclose(np.vstack(step_by_step), full, rtol=0, atol=1e-12)


def test_stability_filter_has_no_look_ahead(model, feats):
    short = model.regime_states(feats.iloc[:700])
    long = model.regime_states(feats.iloc[:900])[:700]
    assert [(s.state_id, s.is_confirmed, s.consecutive_bars, s.is_flickering) for s in short] == [
        (s.state_id, s.is_confirmed, s.consecutive_bars, s.is_flickering) for s in long
    ]


def test_features_have_no_look_ahead(settings, data):
    bars = data[0]["SPY"]
    full = build_features(bars, settings["features"])
    truncated = build_features(bars.iloc[:1200], settings["features"])
    pd.testing.assert_frame_equal(full.iloc[:1200], truncated)
