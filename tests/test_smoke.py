"""Smoke tests on synthetic regime-switching prices (no network)."""
from __future__ import annotations

import copy

import numpy as np

from conftest import CALM_VOL, TURBULENT_VOL
from core.hmm_engine import HMMRegimeEngine
from data.feature_engineering import build_features


def test_hmm_recovers_regimes(settings, data):
    bars, true_states = data
    feats = build_features(bars["SPY"], settings["features"]).dropna()
    # Synthetic regimes differ only in volatility, so direction/trend features
    # are pure noise here (all 14 features: ~0.82 agreement on this data).
    s = copy.deepcopy(settings)
    s["features"]["hmm_features"] = ["z_ret_1", "z_rvol_20", "z_vol_ratio", "z_natr_14", "z_volume_norm"]
    eng = HMMRegimeEngine(s).fit(feats)
    probs = eng.predict_regime_proba(feats)
    # The HMM may split a true regime into several states; classify each bar by
    # the expected volatility of its most likely state.
    vols = np.array([i.expected_volatility for i in eng.regime_info])
    cutoff = np.sqrt(252) * (CALM_VOL + TURBULENT_VOL) / 2
    turbulent = (vols[probs.values.argmax(axis=1)] > cutoff).astype(int)
    agreement = (turbulent == true_states.reindex(probs.index).values).mean()
    assert agreement > 0.85, agreement
