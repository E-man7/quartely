"""Volatility-based allocation strategies.

The HMM detects volatility environments, not direction. Calm markets tend to
trend up and the worst drawdowns cluster in volatility spikes, so allocation
is set by each regime's *volatility rank*:

* low vol  -> LowVolBullStrategy:       95% at 1.25x
* mid vol  -> MidVolCautiousStrategy:   95% if price > 50 EMA, else 60%
* high vol -> HighVolDefensiveStrategy: 60% (stay partly invested for rebounds)

Always long, never short: in walk-forward tests shorts gave back crash gains
on V-shaped rebounds, which the HMM detects 2-3 bars late. High volatility
means *less* exposure, never reversed exposure.

The orchestrator ranks regimes by `expected_volatility`, independently of the
HMM's return-sorted labels: a "BULL" label does not imply low volatility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from data.feature_engineering import atr
from monitoring.logger import get_logger

if TYPE_CHECKING:
    from core.hmm_engine import RegimeInfo, RegimeState

log = get_logger(__name__)

UNCERTAINTY_NOTE = "[UNCERTAINTY — size halved]"
TRANSITION_NOTE = "[TRANSITION — size reduced pending confirmation]"


class Direction(str, Enum):
    LONG = "LONG"
    FLAT = "FLAT"


@dataclass
class Signal:
    symbol: str
    direction: Direction
    confidence: float                 # filtered probability of the regime acted on
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float          # fraction of portfolio before leverage (0.60-0.95 nominal)
    leverage: float                   # 1.0 or 1.25
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: Optional[pd.Timestamp]
    reasoning: str
    strategy_name: str
    metadata: Dict = field(default_factory=dict)

    @property
    def target_weight(self) -> float:
        """Portfolio weight implied by size and leverage (0 when FLAT)."""
        return self.position_size_pct * self.leverage if self.direction is Direction.LONG else 0.0

    def as_dict(self) -> Dict:
        return {
            "symbol": self.symbol, "direction": self.direction.value, "strategy": self.strategy_name,
            "position_size_pct": round(self.position_size_pct, 4), "leverage": self.leverage,
            "entry_price": round(self.entry_price, 4), "stop_loss": round(self.stop_loss, 4),
            "regime": self.regime_name, "regime_probability": round(self.regime_probability, 4),
            "reasoning": self.reasoning,
        }


# ------------------------------------------------------------- strategies
def _ewm_last(x: np.ndarray, alpha: float) -> float:
    """Last value of pandas `ewm(alpha=alpha, adjust=False)` (y0 = x0) via a linear filter."""
    if len(x) == 1:
        return float(x[0])
    y, _ = lfilter([alpha], [1.0, alpha - 1.0], x[1:], zi=[(1.0 - alpha) * x[0]])
    return float(y[-1])


def ema_atr(bars: pd.DataFrame, ema_window: int, atr_window: int) -> "tuple[float, float]":
    """Latest EMA(close) and Wilder ATR; same values as the pandas indicators, ~50x faster."""
    o = bars[["high", "low", "close"]].to_numpy(float)
    if not np.isfinite(o).all() or len(o) < atr_window:
        ema = float(bars["close"].ewm(span=ema_window, adjust=False).mean().iat[-1])
        return ema, float(atr(bars, atr_window).iat[-1])
    high, low, close = o.T
    prev = np.r_[np.nan, close[:-1]]
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    return _ewm_last(close, 2.0 / (ema_window + 1)), _ewm_last(tr, 1.0 / atr_window)


class BaseStrategy(ABC):
    EMA_WINDOW = 50
    ATR_WINDOW = 14
    MAX_ALLOCATION = 0.95
    LEVERAGE = 1.0

    def __init__(self, lookback: int = 250):
        # Indicators use the trailing `lookback` bars so live and backtest agree
        # regardless of how much history is loaded (EMA/ATR weights beyond it are ~0).
        self.lookback = max(lookback, self.EMA_WINDOW + 1)

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def plan(self, price: float, ema: float, atr_value: float) -> Tuple[float, float, float, str]:
        """Return (position_size_pct, leverage, stop_loss, reasoning)."""

    def generate_signal(self, symbol: str, bars: pd.DataFrame, regime_state: RegimeState) -> Optional[Signal]:
        """Long signal from bars up to the current bar; None if there is too little history."""
        if len(bars) < self.EMA_WINDOW + 1:
            return None
        window = bars.iloc[-self.lookback:]
        price = float(window["close"].iat[-1])
        ema, atr_value = ema_atr(window, self.EMA_WINDOW, self.ATR_WINDOW)
        if not np.isfinite([price, ema, atr_value]).all():
            return None
        size, leverage, stop, why = self.plan(price, ema, atr_value)
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop,
            take_profit=None,
            position_size_pct=size,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=regime_state.timestamp,
            reasoning=why,
            strategy_name=self.name,
            metadata={"ema_50": ema, "atr_14": atr_value},
        )


class LowVolBullStrategy(BaseStrategy):
    """Calm markets trend up: fully invested with modest leverage."""

    LEVERAGE = 1.25

    def plan(self, price, ema, atr_value):
        stop = max(price - 3.0 * atr_value, ema - 0.5 * atr_value)
        return self.MAX_ALLOCATION, self.LEVERAGE, stop, (
            f"Low-vol regime: long {self.MAX_ALLOCATION:.0%} at {self.LEVERAGE}x; "
            f"stop max(price - 3 ATR, 50 EMA - 0.5 ATR) = {stop:.2f}"
        )


class MidVolCautiousStrategy(BaseStrategy):
    """Stay invested while the trend holds, cut exposure when price loses the 50 EMA."""

    BROKEN_TREND_ALLOCATION = 0.60

    def plan(self, price, ema, atr_value):
        stop = ema - 0.5 * atr_value
        if price > ema:
            size, why = self.MAX_ALLOCATION, f"price {price:.2f} > 50 EMA {ema:.2f}, trend intact"
        else:
            size, why = self.BROKEN_TREND_ALLOCATION, f"price {price:.2f} <= 50 EMA {ema:.2f}, trend broken"
        return size, self.LEVERAGE, stop, (
            f"Mid-vol regime: {why}; long {size:.0%} at {self.LEVERAGE}x; stop 50 EMA - 0.5 ATR = {stop:.2f}"
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """Reduce, don't reverse: 60% long still catches sharp rebounds after selloffs."""

    MAX_ALLOCATION = 0.60

    def plan(self, price, ema, atr_value):
        stop = ema - 1.0 * atr_value
        return self.MAX_ALLOCATION, self.LEVERAGE, stop, (
            f"High-vol regime: defensive long {self.MAX_ALLOCATION:.0%} at {self.LEVERAGE}x; "
            f"stop 50 EMA - 1.0 ATR = {stop:.2f}"
        )


# Backward-compatible names from the label-based design
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

# Legacy label -> strategy lookup. The orchestrator does NOT use it; allocation
# follows volatility rank, never labels.
LABEL_TO_STRATEGY: Dict[str, Type[BaseStrategy]] = {
    "CRASH": CrashDefensiveStrategy,
    "STRONG_BEAR": BearTrendStrategy,
    "BEAR": BearTrendStrategy,
    "WEAK_BEAR": BearTrendStrategy,
    "NEUTRAL": MeanReversionStrategy,
    "WEAK_BULL": BullTrendStrategy,
    "BULL": BullTrendStrategy,
    "STRONG_BULL": BullTrendStrategy,
    "EUPHORIA": EuphoriaCautiousStrategy,
}


def strategy_for_vol_rank(rank: int, n_regimes: int) -> Type[BaseStrategy]:
    """Map a volatility rank (0 = calmest) to a strategy class for any regime count."""
    position = rank / (n_regimes - 1) if n_regimes > 1 else 0.0
    if position <= 0.33:
        return LowVolBullStrategy
    if position >= 0.67:
        return HighVolDefensiveStrategy
    return MidVolCautiousStrategy


# ----------------------------------------------------------- orchestrator
class StrategyOrchestrator:
    def __init__(self, config: Dict, regime_infos: Optional[Sequence[RegimeInfo]] = None):
        cfg = config.get("strategy", {})
        self.min_confidence = cfg.get("min_confidence", 0.55)
        self.uncertainty_mult = cfg.get("uncertainty_size_mult", 0.5)
        self.transition_mult = cfg.get("transition_size_mult", 0.75)
        self.lookback = max(cfg.get("indicator_lookback", 250), BaseStrategy.EMA_WINDOW + 1)
        lookback = self.lookback
        self._strategies = {cls: cls(lookback) for cls in
                            (LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy)}
        self.vol_rank: Dict[int, int] = {}
        self.strategy_map: Dict[int, BaseStrategy] = {}
        if regime_infos:
            self.update_regime_infos(regime_infos)

    def update_regime_infos(self, regime_infos: Sequence[RegimeInfo]) -> None:
        """Rebuild regime_id -> vol rank -> strategy (call after every HMM retrain)."""
        ordered = sorted(regime_infos, key=lambda i: (i.expected_volatility, i.regime_id))
        n = len(ordered)
        vol_rank = {info.regime_id: rank for rank, info in enumerate(ordered)}
        strategy_map = {rid: self._strategies[strategy_for_vol_rank(rank, n)] for rid, rank in vol_rank.items()}
        changed = {k: s.name for k, s in strategy_map.items()} != {k: s.name for k, s in self.strategy_map.items()}
        self.vol_rank, self.strategy_map = vol_rank, strategy_map
        if changed:
            log.info("strategy mapping", extra={"mapping": {
                f"{i.regime_name} (vol {i.expected_volatility:.1%})": strategy_map[i.regime_id].name for i in ordered}})

    def strategy_for(self, regime_id: int) -> BaseStrategy:
        if regime_id not in self.strategy_map:
            raise KeyError(f"No strategy for regime {regime_id}; call update_regime_infos() after retraining")
        return self.strategy_map[regime_id]

    def generate_signals(
        self,
        symbols: Sequence[str],
        bars: Mapping[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: Optional[bool] = None,
    ) -> List[Signal]:
        """One long signal per symbol with history; the regime allocation is split equally across symbols."""
        if regime_state.n_states != len(self.strategy_map):
            raise ValueError(f"Regime state has {regime_state.n_states} states but the strategy map has "
                             f"{len(self.strategy_map)}; call update_regime_infos() after retraining")
        flickering = regime_state.is_flickering if is_flickering is None else is_flickering
        uncertain = regime_state.probability < self.min_confidence or flickering
        pending = not regime_state.is_confirmed and not uncertain
        strategy = self.strategy_for(regime_state.state_id)

        signals = []
        for symbol in symbols:
            sig = strategy.generate_signal(symbol, bars[symbol], regime_state)
            if sig is None:
                continue
            if uncertain:
                sig.position_size_pct *= self.uncertainty_mult
                sig.leverage = 1.0
                sig.reasoning += f" {UNCERTAINTY_NOTE}"
            elif pending:
                sig.position_size_pct *= self.transition_mult
                sig.reasoning += f" {TRANSITION_NOTE}"
            sig.position_size_pct /= len(symbols)
            sig.metadata.update(vol_rank=self.vol_rank[regime_state.state_id], n_regimes=len(self.vol_rank),
                                uncertain=uncertain, pending_confirmation=pending)
            signals.append(sig)
        return signals
