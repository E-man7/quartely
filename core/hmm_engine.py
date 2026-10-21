"""Gaussian HMM regime detection.

The HMM is a volatility classifier: it identifies calm / moderate / turbulent
environments from causal, rolling-z-scored features. It does not forecast
direction.

Design notes
------------
* Model order is selected by BIC over `n_states_candidates`, with `n_init`
  random restarts per candidate (EM only finds local optima).
* After every fit, model states are permuted so state id k is the k-th lowest
  mean return. Labels (CRASH ... EUPHORIA) follow that order and are for human
  readability only. The strategy layer ranks states by `expected_volatility`
  independently; labels never drive allocation.
* Inference uses the *forward* (filtering) recursion only:
  P(state_t | x_1..x_t). hmmlearn's `predict` (Viterbi) and `predict_proba`
  (forward-backward) both condition on the whole sequence, so past states get
  revised with future observations - look-ahead bias.
* A causal stability filter confirms a regime change only after it persists
  `confirm_bars` bars, and enters uncertainty mode when the filtered argmax
  flickers more than `flicker_threshold` times per `flicker_window` bars.
"""
from __future__ import annotations

import pickle
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from core.regime_strategies import strategy_for_vol_rank
from data.feature_engineering import HMM_FEATURES, RETURN_COL, TRADING_DAYS
from monitoring.logger import get_logger

log = get_logger(__name__)

# Ordered by increasing mean return
REGIME_LABELS: Dict[int, List[str]] = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}


def regime_labels(n: int) -> List[str]:
    return REGIME_LABELS.get(n, [f"REGIME_{i}" for i in range(n)])


@dataclass
class RegimeInfo:
    regime_id: int
    regime_name: str
    expected_return: float            # annualized mean log return (training window, filtered weights)
    expected_volatility: float        # annualized std of log returns
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float      # max risk-asset weight (1.0 = 100% of equity)
    min_confidence_to_act: float


@dataclass
class RegimeState:
    label: str
    state_id: int                     # confirmed regime id (ids sorted by mean return)
    probability: float                # filtered P(state_id | data up to t)
    state_probabilities: np.ndarray   # filtered distribution over all state ids
    timestamp: Optional[pd.Timestamp]
    is_confirmed: bool                # False while a change is pending or in uncertainty mode
    consecutive_bars: int             # bars spent in the confirmed regime
    state_volatilities: np.ndarray = field(default_factory=lambda: np.empty(0))  # expected vol per state id
    is_flickering: bool = False
    flicker_rate: int = 0

    @property
    def n_states(self) -> int:
        return len(self.state_probabilities)

    @property
    def raw_state_id(self) -> int:
        """Filtered argmax at this bar, before the stability filter."""
        return int(np.argmax(self.state_probabilities))

    @property
    def vol_rank(self) -> int:
        """Rank of the confirmed regime by expected volatility (0 = calmest)."""
        return int(np.argsort(np.argsort(self.state_volatilities, kind="stable"))[self.state_id])


def stability_filter(
    raw_ids: np.ndarray, confirm_bars: int = 3, flicker_window: int = 20, flicker_threshold: int = 4
) -> pd.DataFrame:
    """Causal regime-confirmation filter over a sequence of filtered argmax ids.

    Row t depends only on raw_ids[:t + 1]. Columns: `regime` (confirmed id),
    `is_confirmed`, `consecutive_bars`, `flicker_rate` (argmax switches in the
    trailing window), `flickering`, `pending` (bar a new candidate appeared),
    `changed` (bar a change was confirmed).
    """
    raw = np.asarray(raw_ids, dtype=int)
    T = len(raw)
    switches = np.r_[0, (raw[1:] != raw[:-1]).astype(int)] if T else np.zeros(0, dtype=int)
    flicker = pd.Series(switches).rolling(flicker_window, min_periods=1).sum().to_numpy().astype(int)
    cols = {k: np.zeros(T, dtype=int) for k in ("regime", "consecutive_bars")}
    flags = {k: np.zeros(T, dtype=bool) for k in ("is_confirmed", "flickering", "pending", "changed")}

    current, candidate, run, streak = (int(raw[0]) if T else 0), -1, 0, 0
    for t in range(T):
        flickering = flicker[t] > flicker_threshold
        if raw[t] == current:
            candidate, run = -1, 0
        else:
            if raw[t] == candidate:
                run += 1
            else:
                candidate, run = int(raw[t]), 1
                flags["pending"][t] = True
            # No confirmations in uncertainty mode: hold the last confirmed regime
            if run >= confirm_bars and not flickering:
                current, streak = candidate, run - 1
                candidate, run = -1, 0
                flags["changed"][t] = True
        streak += 1
        cols["regime"][t] = current
        cols["consecutive_bars"][t] = streak
        flags["is_confirmed"][t] = candidate == -1 and not flickering
        flags["flickering"][t] = flickering
    return pd.DataFrame({**cols, **flags, "flicker_rate": flicker})


class HMMRegimeEngine:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.cfg = settings.get("hmm", {})
        self.feature_cols: List[str] = settings.get("features", {}).get("hmm_features", HMM_FEATURES)
        stab = self.cfg.get("stability", {})
        self.confirm_bars = stab.get("confirm_bars", 3)
        self.flicker_window = stab.get("flicker_window", 20)
        self.flicker_threshold = stab.get("flicker_threshold", 4)

        self.model: Optional[GaussianHMM] = None
        self.n_states: int = 0
        self.labels: List[str] = []
        self.regime_info: List[RegimeInfo] = []
        self.fitted_at: Optional[pd.Timestamp] = None
        self.bic_scores: Dict[int, float] = {}
        self.bic: float = np.nan
        self.log_likelihood: float = np.nan
        self.n_obs: int = 0
        self._fits_since_selection = 0
        self._cache: Optional[Dict] = None            # forward-pass cache (see _forward)
        self._last: Optional[pd.Series] = None        # stability-filter row for the latest bar
        self._last_logged: Optional[pd.Timestamp] = None

    # ---------------------------------------------------------------- fitting
    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            raise KeyError(f"HMM feature(s) not found: {missing}")
        return features[self.feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _n_params(n: int, d: int, cov_type: str) -> int:
        cov = {"full": n * d * (d + 1) // 2, "diag": n * d, "spherical": n, "tied": d * (d + 1) // 2}[cov_type]
        return n * (n - 1) + (n - 1) + n * d + cov

    def _new_model(self, n: int, **kw) -> GaussianHMM:
        return GaussianHMM(
            n_components=n,
            covariance_type=self.cfg.get("covariance_type", "full"),
            n_iter=self.cfg.get("n_iter", 200),
            tol=self.cfg.get("tol", 1e-4),
            min_covar=1e-4,
            **kw,
        )

    def _fit_one(self, X: np.ndarray, n: int) -> "tuple[Optional[GaussianHMM], float]":
        best, best_ll = None, -np.inf
        rng = np.random.RandomState(self.cfg.get("random_state", 42))
        for _ in range(self.cfg.get("n_init", 10)):
            model = self._new_model(n, random_state=rng.randint(0, 2**31 - 1))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X)
                ll = model.score(X)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = model, ll
        return best, best_ll

    def _warm_start(self, X: np.ndarray) -> Optional[GaussianHMM]:
        """Refit from the previous parameters (fast; keeps state identities stable)."""
        prev = self.model
        model = self._new_model(prev.n_components, init_params="")
        model.n_features = prev.n_features
        model.startprob_ = prev.startprob_.copy()
        model.transmat_ = prev.transmat_.copy()
        model.means_ = prev.means_.copy()
        model._covars_ = prev._covars_.copy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
            return model if np.isfinite(model.score(X)) else None
        except (ValueError, np.linalg.LinAlgError):
            return None

    @staticmethod
    def _fit_stats(model: GaussianHMM) -> Dict:
        mon = model.monitor_
        return {"converged": bool(mon.converged), "iterations": int(mon.iter)}

    def fit(self, features: pd.DataFrame) -> "HMMRegimeEngine":
        X = self._prepare(features)
        min_size = self.cfg.get("min_train_size", 504)
        if len(X) < min_size:
            raise ValueError(f"Need >= {min_size} observations to fit HMM, got {len(X)}")
        arr = X.values
        T, d = arr.shape
        cov_type = self.cfg.get("covariance_type", "full")
        train_end = str(X.index[-1].date())

        reselect_every = self.cfg.get("model_selection_every", 252)
        if self.model is not None and self._fits_since_selection * self.cfg.get("retrain_every", 21) < reselect_every:
            model = self._warm_start(arr)
            if model is not None:
                ll = model.score(arr)
                self.bic = -2 * ll + self._n_params(model.n_components, d, cov_type) * np.log(T)
                self.log_likelihood = ll
                self._fits_since_selection += 1
                self._finalize(model, X, features)
                log.debug("hmm warm refit", extra={"train_end": train_end, "log_likelihood": round(ll, 1),
                                                   "bic": round(self.bic, 1), **self._fit_stats(model)})
                return self

        best_model, best_bic, best_ll = None, np.inf, np.nan
        self.bic_scores = {}
        for n in self.cfg.get("n_states_candidates", [3, 4, 5, 6, 7]):
            model, ll = self._fit_one(arr, n)
            if model is None:
                log.warning("hmm candidate failed", extra={"n_states": n, "train_end": train_end})
                continue
            bic = -2 * ll + self._n_params(n, d, cov_type) * np.log(T)
            self.bic_scores[n] = bic
            log.info("hmm candidate", extra={"n_states": n, "log_likelihood": round(ll, 1), "bic": round(bic, 1),
                                             **self._fit_stats(model)})
            if bic < best_bic:
                best_model, best_bic, best_ll = model, bic, ll
        if best_model is None:
            raise RuntimeError("HMM failed to converge for every candidate state count")

        self.bic, self.log_likelihood = best_bic, best_ll
        self._fits_since_selection = 1
        self._finalize(best_model, X, features)
        log.info(
            "hmm selected",
            extra={"n_states": self.n_states, "bic": round(best_bic, 1), "log_likelihood": round(best_ll, 1),
                   "bic_scores": {k: round(v, 1) for k, v in self.bic_scores.items()},
                   "labels": self.labels, "train_end": train_end, "obs": T, **self._fit_stats(best_model)},
        )
        return self

    def _finalize(self, model: GaussianHMM, X: pd.DataFrame, features: pd.DataFrame) -> None:
        """Sort states by mean return, then build per-regime metadata."""
        self.model, self._cache = model, None
        weights = np.exp(self._forward(X.values))
        if RETURN_COL in features.columns:
            r = features.loc[X.index, RETURN_COL].to_numpy()
        else:  # fall back to the standardized return feature
            r = X.iloc[:, 0].to_numpy()
        w = weights / np.maximum(weights.sum(axis=0), 1e-12)
        mean = w.T @ r
        std = np.sqrt(np.maximum(w.T @ (r**2) - mean**2, 0.0))

        order = np.argsort(mean, kind="stable")
        model.startprob_ = model.startprob_[order]
        model.transmat_ = model.transmat_[np.ix_(order, order)]
        model.means_ = model.means_[order]
        if model.covariance_type != "tied":
            model._covars_ = model._covars_[order]
        mean, std = mean[order], std[order]

        self.model, self._cache = model, None
        self.n_states = model.n_components
        self.labels = regime_labels(self.n_states)
        self.fitted_at = X.index[-1]
        self.n_obs = len(X)
        self.regime_info = self._regime_info(mean * TRADING_DAYS, std * np.sqrt(TRADING_DAYS))

    def _regime_info(self, ann_ret: np.ndarray, ann_vol: np.ndarray) -> List[RegimeInfo]:
        """Metadata from the strategy each regime maps to by volatility rank."""
        min_conf = self.settings.get("strategy", {}).get("min_confidence", 0.55)
        vol_rank = np.argsort(np.argsort(ann_vol, kind="stable"))
        infos = []
        for k in range(self.n_states):
            strategy = strategy_for_vol_rank(int(vol_rank[k]), self.n_states)
            infos.append(RegimeInfo(
                regime_id=k,
                regime_name=self.labels[k],
                expected_return=float(ann_ret[k]),
                expected_volatility=float(ann_vol[k]),
                recommended_strategy_type=strategy.__name__,
                max_leverage_allowed=strategy.LEVERAGE,
                max_position_size_pct=strategy.MAX_ALLOCATION,
                min_confidence_to_act=float(min_conf),
            ))
        return infos

    # -------------------------------------------------------------- inference
    def _log_emissions(self, X: np.ndarray) -> np.ndarray:
        """log N(x_t | mean_k, cov_k) for every row and state."""
        m = self.model
        T, d = X.shape
        out = np.empty((T, m.n_components))
        for k in range(m.n_components):
            cov = m.covars_[k]
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(cov + 1e-6 * np.eye(d))
            z = solve_triangular(L, (X - m.means_[k]).T, lower=True)
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + (z**2).sum(axis=0)) - np.log(np.diag(L)).sum()
        return out

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Normalized log alpha_t = log P(state_t | x_1..x_t) for every row of X.

        The previous pass is cached; if X extends it (identical leading rows,
        same model) only the new rows are processed, so a live/backtest loop
        costs one step per bar. Results are identical to a fresh pass.
        """
        if self.model is None:
            raise RuntimeError("HMM not fitted")
        c = self._cache
        n_prev = 0
        if c is not None and c["model"] is self.model:
            n_c = len(c["X"])
            if len(X) <= n_c and np.array_equal(X, c["X"][: len(X)]):
                return c["log_alpha"][: len(X)]
            if np.array_equal(X[:n_c], c["X"]):
                n_prev = n_c

        log_alpha = np.empty((len(X), self.model.n_components))
        if n_prev:
            log_alpha[:n_prev] = c["log_alpha"]
        if len(X) > n_prev:
            log_b = self._log_emissions(X[n_prev:])
            log_A = np.log(np.maximum(self.model.transmat_, 1e-300))
            if n_prev == 0:
                log_alpha[0] = np.log(np.maximum(self.model.startprob_, 1e-300)) + log_b[0]
                log_alpha[0] -= logsumexp(log_alpha[0])
            for t in range(max(n_prev, 1), len(X)):
                log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t - n_prev]
                log_alpha[t] -= logsumexp(log_alpha[t])
        self._cache = {"model": self.model, "X": X.copy(), "log_alpha": log_alpha}
        return log_alpha

    def predict_regime_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Filtered P(state_t | data up to t) for every row; columns are regime labels."""
        X = self._prepare(features)
        return pd.DataFrame(np.exp(self._forward(X.values)), index=X.index, columns=self.labels)

    def predict_regime_filtered(self, features: pd.DataFrame) -> np.ndarray:
        """Most likely state id per row under the forward filter (no future data)."""
        return self.predict_regime_proba(features).to_numpy().argmax(axis=1)

    def regime_states(self, features: pd.DataFrame) -> List[RegimeState]:
        """Filtered + stability-filtered regime state for every row of `features`."""
        probs = self.predict_regime_proba(features)
        P = probs.to_numpy()
        stab = stability_filter(P.argmax(axis=1), self.confirm_bars, self.flicker_window, self.flicker_threshold)
        vols = np.array([info.expected_volatility for info in self.regime_info])
        states = [
            RegimeState(
                label=self.labels[row.regime],
                state_id=int(row.regime),
                probability=float(P[t, row.regime]),
                state_probabilities=P[t],
                timestamp=ts,
                is_confirmed=bool(row.is_confirmed),
                consecutive_bars=int(row.consecutive_bars),
                state_volatilities=vols,
                is_flickering=bool(row.flickering),
                flicker_rate=int(row.flicker_rate),
            )
            for t, (ts, row) in enumerate(zip(probs.index, stab.itertuples(index=False)))
        ]
        if len(stab):
            self._last = stab.iloc[-1]
            self._log_events(probs.index, P, stab)
        return states

    def _log_events(self, index: pd.Index, P: np.ndarray, stab: pd.DataFrame) -> None:
        """Log transitions for bars not logged before (first call: latest bar only)."""
        start = len(index) - 1 if self._last_logged is None else int(index.searchsorted(self._last_logged, "right"))
        for t in range(start, len(index)):
            ts, row = index[t], stab.iloc[t]
            raw = int(P[t].argmax())
            extra = {"timestamp": str(ts), "regime": self.labels[row.regime]}
            if row.pending:
                log.warning("regime change detected", extra={**extra, "candidate": self.labels[raw],
                                                              "probability": round(float(P[t, raw]), 3)})
            if row.changed:
                log.info("regime change confirmed", extra={**extra, "after_bars": self.confirm_bars})
            if row.flickering and (t == 0 or not stab.flickering.iat[t - 1]):
                log.warning("regime flickering: uncertainty mode", extra={**extra, "flicker_rate": int(row.flicker_rate)})
        self._last_logged = index[-1]

    def current_regime(self, features: pd.DataFrame) -> RegimeState:
        return self.regime_states(features)[-1]

    # ------------------------------------------------ state of the latest bar
    def _require_state(self) -> pd.Series:
        if self._last is None:
            raise RuntimeError("No regime computed yet; call regime_states()/current_regime() first")
        return self._last

    def get_regime_stability(self) -> int:
        """Consecutive bars in the current confirmed regime."""
        return int(self._require_state().consecutive_bars)

    def detect_regime_change(self) -> bool:
        """True only on the bar where a regime change was confirmed."""
        return bool(self._require_state().changed)

    def get_regime_flicker_rate(self) -> int:
        """Filtered-argmax switches over the trailing `flicker_window` bars."""
        return int(self._require_state().flicker_rate)

    def is_flickering(self) -> bool:
        return bool(self._require_state().flickering)

    def get_transition_matrix(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("HMM not fitted")
        return pd.DataFrame(self.model.transmat_, index=self.labels, columns=self.labels)

    # ----------------------------------------------------------- scheduling
    def needs_refit(self, index: pd.Index) -> bool:
        """True if unfitted or `retrain_every` new bars have arrived since the last fit."""
        if self.model is None or self.fitted_at is None:
            return True
        return int((index > self.fitted_at).sum()) >= self.cfg.get("retrain_every", 21)

    def train_slice(self, features: pd.DataFrame) -> pd.DataFrame:
        """Training data: trailing `train_window` bars, or everything (expanding) if null."""
        window = self.cfg.get("train_window")
        return features.iloc[-window:] if window else features

    def describe(self) -> pd.DataFrame:
        """Per-regime metadata, persistence and expected duration (in bars)."""
        if self.model is None:
            raise RuntimeError("HMM not fitted")
        df = pd.DataFrame([asdict(i) for i in self.regime_info]).set_index("regime_name")
        df["persistence"] = np.diag(self.model.transmat_)
        df["expected_duration"] = 1 / (1 - df["persistence"].clip(upper=0.9999))
        return df

    # ---------------------------------------------------------- persistence
    def metadata(self) -> Dict:
        return {
            "n_regimes": self.n_states,
            "bic": float(self.bic),
            "log_likelihood": float(self.log_likelihood),
            "bic_scores": dict(self.bic_scores),
            "training_date": str(self.fitted_at.date()) if self.fitted_at is not None else None,
            "n_observations": self.n_obs,
            "labels": list(self.labels),
            "feature_cols": list(self.feature_cols),
            **(self._fit_stats(self.model) if self.model is not None else {}),
        }

    def save(self, path: Union[str, Path]) -> Path:
        if self.model is None:
            raise RuntimeError("HMM not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model, "metadata": self.metadata(), "regime_info": self.regime_info,
                   "fitted_at": self.fitted_at}
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        log.info("hmm saved", extra={"path": str(path), **self.metadata()})
        return path

    @classmethod
    def load(cls, path: Union[str, Path], settings: Dict) -> "HMMRegimeEngine":
        """Only load pickles you created: unpickling can execute arbitrary code."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        meta = payload["metadata"]
        eng = cls(settings)
        eng.model = payload["model"]
        eng.regime_info = payload["regime_info"]
        eng.fitted_at = payload["fitted_at"]
        eng.n_states = meta["n_regimes"]
        eng.labels = meta["labels"]
        eng.feature_cols = meta["feature_cols"]
        eng.bic, eng.log_likelihood = meta["bic"], meta["log_likelihood"]
        eng.bic_scores, eng.n_obs = meta["bic_scores"], meta["n_observations"]
        eng._fits_since_selection = 1
        return eng
