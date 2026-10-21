"""Shared fixtures: synthetic regime-switching prices (no network)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_settings  # noqa: E402

CALM_VOL, TURBULENT_VOL = 0.007, 0.025   # daily


def synthetic_bars(n: int = 2000, seed: int = 0):
    """Two-state Markov market: calm (+drift, 0.7% vol) and turbulent (-drift, 2.5% vol)."""
    rng = np.random.RandomState(seed)
    P = np.array([[0.99, 0.01], [0.04, 0.96]])
    mu, sd = np.array([0.0006, -0.0015]), np.array([CALM_VOL, TURBULENT_VOL])
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=P[states[t - 1]])
    r = rng.normal(mu[states], sd[states])
    idx = pd.bdate_range("2012-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(r))
    intraday = np.abs(rng.normal(0, sd[states] / 2))
    volume = 1e6 * np.exp(rng.normal(0, 0.2, n) + 0.5 * states)   # busier in turbulence
    spy = pd.DataFrame({"open": close, "high": close * (1 + intraday), "low": close * (1 - intraday),
                        "close": close, "volume": volume}, index=idx)
    shy_close = 80 * np.exp(np.cumsum(rng.normal(0.0001, 0.0008, n)))
    shy = pd.DataFrame({"open": shy_close, "high": shy_close, "low": shy_close, "close": shy_close,
                        "volume": 1e6}, index=idx)
    return {"SPY": spy, "SHY": shy}, pd.Series(states, index=idx)


@pytest.fixture(scope="session")
def settings():
    s = copy.deepcopy(load_settings())
    s["backtest"]["start"] = None
    s["backtest"]["end"] = None
    s["hmm"]["n_states_candidates"] = [3, 4]
    s["hmm"]["n_init"] = 2
    s["hmm"]["train_window"] = 750
    s["hmm"]["min_train_size"] = 500
    s["hmm"]["model_dir"] = None
    s["stress_test"]["n_simulations"] = 20
    return s


@pytest.fixture(scope="session")
def data():
    return synthetic_bars()
