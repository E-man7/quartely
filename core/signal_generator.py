"""Combines HMM regime detection, allocation strategy and risk manager into signals."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import pandas as pd

from config import PROJECT_ROOT
from core.hmm_engine import HMMRegimeEngine, RegimeState
from core.regime_strategies import Signal, StrategyOrchestrator
from core.risk_manager import RiskDecision, RiskManager
from data.feature_engineering import build_features
from monitoring.logger import get_logger

log = get_logger(__name__)

_PERIOD_KEY = {
    "daily": lambda ts: (ts.year, ts.dayofyear),
    "weekly": lambda ts: tuple(ts.isocalendar()[:2]),
    "monthly": lambda ts: (ts.year, ts.month),
    "quarterly": lambda ts: (ts.year, ts.quarter),
}


@dataclass
class PortfolioSignal:
    timestamp: pd.Timestamp
    regime: RegimeState
    raw_weight: float                   # strategy output before risk constraints
    target_weights: Dict[str, float]    # final weights to rebalance toward
    risk_scale: float                   # final risk weight / strategy weight
    drawdown: float                     # from peak equity
    halted: bool                        # circuit breaker halt or lock: flat
    rebalance: bool                     # whether the scheduler says to trade now
    events: List[str] = field(default_factory=list)
    trade: Optional[Signal] = None      # strategy signal for the risk asset (None = not enough history)
    risk_decision: Optional[RiskDecision] = None
    breaker: str = "normal"

    def as_dict(self) -> Dict:
        return {
            "timestamp": str(self.timestamp),
            "regime": self.regime.label,
            "regime_probs": [round(float(p), 4) for p in self.regime.state_probabilities],
            "regime_confirmed": self.regime.is_confirmed,
            "regime_bars": self.regime.consecutive_bars,
            "regime_flickering": self.regime.is_flickering,
            "trade": self.trade.as_dict() if self.trade else None,
            "risk": {"approved": self.risk_decision.approved, "rejection": self.risk_decision.rejection_reason,
                     "modifications": self.risk_decision.modifications} if self.risk_decision else None,
            "breaker": self.breaker,
            "raw_weight": round(self.raw_weight, 4),
            "target_weights": {k: round(v, 4) for k, v in self.target_weights.items()},
            "risk_scale": round(self.risk_scale, 3),
            "drawdown": round(self.drawdown, 4),
            "halted": self.halted,
            "rebalance": self.rebalance,
            "events": self.events,
        }


class SignalGenerator:
    def __init__(self, settings: Dict, backtest: bool = False):
        self.settings = settings
        uni = settings.get("universe", {})
        self.risk_symbol = uni.get("risk_symbol", "SPY")
        self.safe_symbol = uni.get("safe_symbol")
        self.feature_cfg = settings.get("features", {})
        freq = settings.get("strategy", {}).get("rebalance_frequency", "daily")
        if freq not in _PERIOD_KEY:
            raise ValueError(f"rebalance_frequency must be one of {list(_PERIOD_KEY)}")
        self._period_key = _PERIOD_KEY[freq]
        self.threshold = settings.get("strategy", {}).get("rebalance_threshold", 0.10)

        self.hmm = HMMRegimeEngine(settings)
        self.strategy = StrategyOrchestrator(settings)
        self.risk = RiskManager(settings, backtest=backtest)
        model_dir = settings.get("hmm", {}).get("model_dir")
        self.model_dir = PROJECT_ROOT / model_dir if model_dir else None
        self._last_period = None
        self._last_regime: Optional[str] = None
        self.last_state: Optional[RegimeState] = None
        self.refit_happened = False          # set by generate(): the HMM was (re)trained this call
        self.hmm_error: Optional[str] = None  # set by generate(): HMM failed, regime held

    def features(self, bars: pd.DataFrame) -> pd.DataFrame:
        return build_features(bars, self.feature_cfg)

    def model_age_days(self, as_of: Optional[pd.Timestamp] = None) -> Optional[float]:
        if self.hmm.fitted_at is None:
            return None
        as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
        return (as_of.normalize() - pd.Timestamp(self.hmm.fitted_at).normalize()).days

    def load_latest_model(self, max_age_days: float, as_of: Optional[pd.Timestamp] = None) -> bool:
        """Load the newest saved model if it is at most `max_age_days` old. Returns True if loaded."""
        if self.model_dir is None or not self.model_dir.exists():
            return False
        paths = sorted(self.model_dir.glob("hmm_*.pkl"))
        if not paths:
            return False
        try:
            engine = HMMRegimeEngine.load(paths[-1], self.settings)
        except Exception as exc:
            log.warning("could not load saved HMM", extra={"path": str(paths[-1]), "error": str(exc)})
            return False
        previous, self.hmm = self.hmm, engine
        age = self.model_age_days(as_of)
        if age is None or age > max_age_days:
            self.hmm = previous
            log.info("saved HMM too old: retraining", extra={"path": str(paths[-1]), "age_days": age})
            return False
        self.strategy.update_regime_infos(self.hmm.regime_info)
        log.info("loaded saved HMM", extra={"path": str(paths[-1]), "age_days": age, **self.hmm.metadata()})
        return True

    def ensure_fitted(self, features: pd.DataFrame, max_age_days: Optional[float] = None,
                      as_of: Optional[pd.Timestamp] = None) -> bool:
        """Fit/refit the HMM if due (and save it). Returns True if refit.

        Due when unfitted, or (live) the model is older than `max_age_days`, or
        (default) `hmm.retrain_every` new bars have arrived since the last fit.
        """
        if max_age_days is not None and self.hmm.model is not None:
            due = self.model_age_days(as_of) > max_age_days
        else:
            due = self.hmm.needs_refit(features.index)
        if due:
            self.hmm.fit(self.hmm.train_slice(features))
            self.strategy.update_regime_infos(self.hmm.regime_info)
            if self.model_dir is not None:
                self.hmm.save(self.model_dir / f"hmm_{self.hmm.fitted_at.date()}.pkl")
            return True
        return False

    def is_rebalance_bar(self, ts: pd.Timestamp) -> bool:
        key = self._period_key(ts)
        if key != self._last_period:
            self._last_period = key
            return True
        return False

    def from_regime(
        self,
        state: RegimeState,
        risk_bars: pd.DataFrame,
        equity: Optional[float] = None,
        current_weights: Optional[Dict[str, float]] = None,
        day_start_equity: Optional[float] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> PortfolioSignal:
        """`risk_bars`: OHLCV of the risk asset up to and including the signal bar.

        `as_of`: wall-clock time for the risk layer's day/week boundaries (live);
        defaults to the bar's timestamp (backtests).
        """
        signals = self.strategy.generate_signals([self.risk_symbol], {self.risk_symbol: risk_bars}, state)
        trade = signals[0] if signals else None
        raw = trade.target_weight if trade else 0.0
        outcome = self.risk.process(trade, equity, current_weights or {}, self.risk_symbol, self.safe_symbol,
                                    as_of if as_of is not None else state.timestamp, state,
                                    day_start_equity=day_start_equity)
        weights = outcome.weights
        events = [e.breaker for e in outcome.events]
        forced = bool(events)
        # Confirmed regime only; labels are compared because ids can shift across refits
        if self._last_regime is not None and state.label != self._last_regime:
            events.append("regime_change")
        self._last_regime = state.label

        ts = state.timestamp
        rebalance = self.is_rebalance_bar(ts) or forced
        if rebalance and current_weights is not None and not forced:
            # Ignore small target changes (probability noise) to avoid churn
            drift = max(abs(weights.get(k, 0) - current_weights.get(k, 0)) for k in weights)
            rebalance = drift > self.threshold
        if rebalance:
            self.risk.record_trades(sum(abs(weights.get(k, 0) - (current_weights or {}).get(k, 0)) > 1e-9
                                        for k in weights), ts)

        b = self.risk.breaker
        return PortfolioSignal(
            timestamp=ts,
            regime=state,
            raw_weight=raw,
            target_weights=weights,
            risk_scale=weights.get(self.risk_symbol, 0.0) / raw if raw else 0.0,
            drawdown=b.peak_dd,
            halted=outcome.status.flatten,
            rebalance=rebalance,
            events=events,
            trade=trade,
            risk_decision=outcome.decision,
            breaker=str(outcome.status),
        )

    def generate(
        self,
        bars: pd.DataFrame,
        equity: Optional[float] = None,
        current_weights: Optional[Dict[str, float]] = None,
        risk_bars: Optional[pd.DataFrame] = None,
        day_start_equity: Optional[float] = None,
        as_of: Optional[pd.Timestamp] = None,
        max_model_age_days: Optional[float] = None,
    ) -> PortfolioSignal:
        """Live entry point: bars of the signal asset up to the latest completed bar.

        Pass `risk_bars` when the risk asset differs from the signal asset, and
        `day_start_equity` (the broker's previous-close equity) for the daily breaker.
        """
        feats = self.features(bars)
        self.refit_happened, self.hmm_error = False, None
        try:
            self.refit_happened = self.ensure_fitted(feats.dropna(), max_model_age_days, as_of)
            # Filter from the start of the train window for a well-initialized posterior
            state = self.hmm.current_regime(self.hmm.train_slice(feats))
        except Exception as exc:
            if self.last_state is None:
                raise
            # HMM failure: hold the current regime rather than trade on nothing
            self.hmm_error = str(exc)
            log.error("HMM error: holding current regime", extra={"error": str(exc), "regime": self.last_state.label})
            state = replace(self.last_state, timestamp=feats.index[-1])
        self.last_state = state
        risk_bars = (bars if risk_bars is None else risk_bars).loc[: state.timestamp]
        signal = self.from_regime(state, risk_bars, equity, current_weights, day_start_equity, as_of)
        log.info("signal", extra=signal.as_dict())
        return signal
