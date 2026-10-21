"""Risk management: independent of the HMM, with absolute veto over every signal.

Defense in depth: circuit breakers fire on actual P&L (equity marks), never on
the regime model, so they still protect the account if the HMM is completely
wrong. Every signal passes through `validate_signal`, every order through
`validate_order`; rejections and modifications are logged with reasons.

Circuit breakers (thresholds in settings.yaml `risk.breakers`):
    daily  DD > 2%  -> sizes x0.5 for the rest of the day
    daily  DD > 3%  -> close everything, halt for the rest of the day
    weekly DD > 5%  -> sizes x0.5 for the rest of the week
    weekly DD > 7%  -> close everything, halt for the rest of the week
    peak   DD > 10% -> close everything, halt, write the lock file. Live trading
                       resumes only after the file is deleted by hand; backtests
                       simulate that review with `lock_cooldown_bars`.

Daily/weekly drawdowns are measured from the equity at the start of the day /
week (the previous close for a daily-bar system).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.regime_strategies import Direction, LowVolBullStrategy, Signal
from monitoring.logger import get_logger

if TYPE_CHECKING:
    from core.hmm_engine import RegimeState

log = get_logger(__name__)


class BreakerLevel(IntEnum):
    NORMAL = 0
    REDUCED = 1      # sizes cut
    HALTED = 2       # flat, no new risk until the day/week rolls
    LOCKED = 3       # flat until the lock is cleared


@dataclass
class BreakerStatus:
    level: BreakerLevel = BreakerLevel.NORMAL
    size_mult: float = 1.0
    reasons: List[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.level > BreakerLevel.NORMAL

    @property
    def flatten(self) -> bool:
        return self.level >= BreakerLevel.HALTED

    def __str__(self) -> str:
        return self.level.name.lower() + (f" ({', '.join(self.reasons)})" if self.reasons else "")


@dataclass
class BreakerEvent:
    timestamp: Optional[pd.Timestamp]
    breaker: str                      # daily_reduce | daily_halt | weekly_reduce | weekly_halt | peak_lock | lock_released
    drawdown: float
    equity: float
    positions_closed: List[str]
    hmm_regime: Optional[str]
    hmm_strategy: Optional[str]
    hmm_wrong: Optional[bool]         # breaker fired while the HMM called the market calm


class CircuitBreaker:
    """Drawdown circuit breakers on actual equity, independent of any model."""

    def __init__(self, cfg: Dict, lock_file: Optional[Path] = None, lock_cooldown_bars: Optional[int] = None):
        self.daily_reduce = cfg.get("daily_reduce_dd", 0.02)
        self.daily_halt = cfg.get("daily_halt_dd", 0.03)
        self.weekly_reduce = cfg.get("weekly_reduce_dd", 0.05)
        self.weekly_halt = cfg.get("weekly_halt_dd", 0.07)
        self.peak_halt = cfg.get("peak_halt_dd", 0.10)
        self.size_mult = cfg.get("size_mult", 0.5)
        self.lock_file = lock_file
        self.lock_cooldown_bars = lock_cooldown_bars
        self.day_start: Optional[float] = None
        self.week_start: Optional[float] = None
        self.peak: Optional[float] = None
        self.equity: Optional[float] = None
        self._day = self._week = None
        self.daily_level = self.weekly_level = BreakerLevel.NORMAL
        self.locked = False
        self._lock_bars = 0
        self.history: List[BreakerEvent] = []

    # ------------------------------------------------------------ drawdowns
    @staticmethod
    def _dd(equity: Optional[float], ref: Optional[float]) -> float:
        return equity / ref - 1 if equity is not None and ref else 0.0

    @property
    def daily_dd(self) -> float:
        return self._dd(self.equity, self.day_start)

    @property
    def weekly_dd(self) -> float:
        return self._dd(self.equity, self.week_start)

    @property
    def peak_dd(self) -> float:
        return self._dd(self.equity, self.peak)

    # ---------------------------------------------------------------- resets
    def reset_daily(self, equity: Optional[float] = None) -> None:
        self.day_start = self.equity if equity is None else equity
        self.daily_level = BreakerLevel.NORMAL

    def reset_weekly(self, equity: Optional[float] = None) -> None:
        self.week_start = self.equity if equity is None else equity
        self.weekly_level = BreakerLevel.NORMAL

    def _lock_present(self) -> bool:
        return self.lock_file is not None and self.lock_file.exists()

    # ---------------------------------------------------------------- update
    def update(
        self,
        equity: float,
        timestamp: Optional[pd.Timestamp] = None,
        positions: Sequence[str] = (),
        regime: Optional[RegimeState] = None,
        strategy: Optional[str] = None,
        day_start: Optional[float] = None,
    ) -> List[BreakerEvent]:
        """Mark equity (e.g. at each close) and fire any breaker it crosses.

        `day_start`: the broker's previous-close equity, when known (live); by
        default the day starts from the last mark before it.
        """
        ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp(datetime.now())
        if self.equity is not None:
            # Period starts are the last mark before the new day/week (the previous close)
            if self._week is not None and tuple(ts.isocalendar()[:2]) != self._week:
                self.reset_weekly(self.equity)
            if self._day is not None and ts.date() != self._day:
                self.reset_daily(self.equity)
        new_day = self._day is None or ts.date() != self._day
        self._day, self._week = ts.date(), tuple(ts.isocalendar()[:2])
        self.equity = equity
        if self.day_start is None:
            self.day_start = self.week_start = self.peak = day_start or equity
        if day_start and new_day:
            self.day_start = day_start
        events: List[BreakerEvent] = []

        def fire(name: str, dd: float) -> None:
            ev = BreakerEvent(ts, name, dd, equity, list(positions) if name != "lock_released" else [],
                              regime.label if regime else None, strategy,
                              (strategy == LowVolBullStrategy.__name__) if strategy else None)
            events.append(ev)
            self.history.append(ev)
            log.warning("circuit breaker", extra={k: (str(v) if k == "timestamp" else v) for k, v in asdict(ev).items()})

        if self.locked:
            self._lock_bars += 1
            cooled = self.lock_cooldown_bars is not None and self._lock_bars > self.lock_cooldown_bars
            if (self.lock_file is not None and not self._lock_present()) or (self.lock_file is None and cooled):
                self.locked, self.peak = False, equity      # drawdown measured afresh after review
                fire("lock_released", 0.0)
            return events
        if self.lock_file is not None and self._lock_present():
            self.locked = True                              # e.g. left over from a previous run
            return events

        self.peak = max(self.peak, equity)
        if self.peak_dd <= -self.peak_halt:
            self.locked, self._lock_bars = True, 0
            fire("peak_lock", self.peak_dd)
            self._write_lock(events[-1])
            return events
        for scope, dd, reduce_at, halt_at in (("weekly", self.weekly_dd, self.weekly_reduce, self.weekly_halt),
                                              ("daily", self.daily_dd, self.daily_reduce, self.daily_halt)):
            level = getattr(self, f"{scope}_level")
            if dd <= -halt_at and level < BreakerLevel.HALTED:
                setattr(self, f"{scope}_level", BreakerLevel.HALTED)
                fire(f"{scope}_halt", dd)
            elif dd <= -reduce_at and level < BreakerLevel.REDUCED:
                setattr(self, f"{scope}_level", BreakerLevel.REDUCED)
                fire(f"{scope}_reduce", dd)
        return events

    def _write_lock(self, event: BreakerEvent) -> None:
        if self.lock_file is None:
            return
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.write_text(json.dumps({
            **asdict(event), "timestamp": str(event.timestamp),
            "message": "Trading halted by the peak-drawdown breaker. Review, then delete this file to resume.",
        }, indent=2, default=str))

    def check(self) -> BreakerStatus:
        if self.locked or self._lock_present():
            return BreakerStatus(BreakerLevel.LOCKED, 0.0, ["peak drawdown lock"])
        level = max(self.daily_level, self.weekly_level)
        reasons = [f"{s} {getattr(self, f'{s}_level').name.lower()}" for s in ("daily", "weekly")
                   if getattr(self, f"{s}_level") > BreakerLevel.NORMAL]
        mult = 0.0 if level >= BreakerLevel.HALTED else self.size_mult if level else 1.0
        return BreakerStatus(level, mult, reasons)

    def get_history(self) -> List[BreakerEvent]:
        return list(self.history)

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> Dict:
        return {"day_start": self.day_start, "week_start": self.week_start, "peak": self.peak, "equity": self.equity,
                "day": str(self._day) if self._day else None, "week": list(self._week) if self._week else None,
                "daily_level": int(self.daily_level), "weekly_level": int(self.weekly_level), "locked": self.locked}

    def load_state(self, state: Dict) -> None:
        self.day_start, self.week_start = state.get("day_start"), state.get("week_start")
        self.peak, self.equity = state.get("peak"), state.get("equity")
        self._day = pd.Timestamp(state["day"]).date() if state.get("day") else None
        self._week = tuple(state["week"]) if state.get("week") else None
        self.daily_level = BreakerLevel(state.get("daily_level", 0))
        self.weekly_level = BreakerLevel(state.get("weekly_level", 0))
        self.locked = bool(state.get("locked", False))


# ------------------------------------------------------------------ state
@dataclass
class PortfolioState:
    equity: float
    cash: float
    buying_power: float
    positions: Dict[str, float]                 # symbol -> market value (signed)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: Optional[float] = None
    drawdown: float = 0.0
    circuit_breaker_status: BreakerStatus = field(default_factory=BreakerStatus)
    flicker_rate: int = 0
    timestamp: Optional[pd.Timestamp] = None
    price_history: Optional[pd.DataFrame] = None   # closes by symbol, for the correlation check
    equity_known: bool = True                       # False: sizing in weights only (no account attached)

    def weight(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0) / self.equity if self.equity else 0.0


@dataclass
class RiskDecision:
    approved: bool
    modified_signal: Optional[Signal]
    rejection_reason: Optional[str] = None
    modifications: List[str] = field(default_factory=list)


@dataclass
class OrderCheck:
    approved: bool
    reason: Optional[str] = None


@dataclass
class RiskOutcome:
    """Result of one bar through the risk layer: decision, final weights, breaker events."""
    decision: Optional[RiskDecision]
    weights: Dict[str, float]
    status: BreakerStatus
    events: List[BreakerEvent]


# ----------------------------------------------------------- risk manager
class RiskManager:
    def __init__(self, settings: Dict, backtest: bool = False, lock_file: Optional[Path] = None):
        cfg = settings.get("risk", {})
        self.cfg = cfg
        self.max_total_exposure = cfg.get("max_total_exposure", 0.80)
        self.max_single_position = cfg.get("max_single_position", 0.15)
        self.max_sector_exposure = cfg.get("max_sector_exposure", 0.30)
        self.max_positions = cfg.get("max_concurrent_positions", 5)
        self.max_daily_trades = cfg.get("max_daily_trades", 20)
        self.max_leverage = cfg.get("max_portfolio_leverage", 1.25)
        self.default_leverage = cfg.get("default_leverage", 1.0)
        self.leverage_max_positions = cfg.get("leverage_max_positions", 3)
        self.flicker_threshold = settings.get("hmm", {}).get("stability", {}).get("flicker_threshold", 4)
        self.max_risk_per_trade = cfg.get("max_risk_per_trade", 0.01)
        self.min_position_value = cfg.get("min_position_value", 100.0)
        self.gap_multiple = cfg.get("gap_multiple", 3.0)
        self.max_gap_loss = cfg.get("max_overnight_gap_loss", 0.02)
        self.corr_window = cfg.get("correlation_window", 60)
        self.corr_reduce = cfg.get("correlation_reduce", 0.70)
        self.corr_reject = cfg.get("correlation_reject", 0.85)
        self.max_spread = cfg.get("max_spread_pct", 0.005)
        self.duplicate_window = cfg.get("duplicate_window_seconds", 60)
        self.sectors: Dict[str, str] = cfg.get("sectors", {}) or {}
        uni = settings.get("universe", {})
        # The defensive sleeve is a cash equivalent: exempt from position limits and stops
        self.cash_equivalents = {s for s in [uni.get("safe_symbol"), *cfg.get("cash_equivalents", [])] if s}

        if backtest:
            lock, cooldown = None, settings.get("backtest", {}).get("lock_cooldown_bars", 21)
        else:
            from config import PROJECT_ROOT

            lock = lock_file or PROJECT_ROOT / cfg.get("lock_file", "trading_halted.lock")
            cooldown = None
        self.breaker = CircuitBreaker(cfg.get("breakers", {}), lock, cooldown)
        self._trades_day = None
        self.trades_today = 0
        self._recent_orders: Dict[Tuple[str, str], float] = {}
        self.rejections: List[Dict] = []

    # --------------------------------------------------------------- helpers
    def _reject(self, signal: Optional[Signal], reason: str, mods: Optional[List[str]] = None) -> RiskDecision:
        rec = {"symbol": signal.symbol if signal else None, "reason": reason,
               "strategy": signal.strategy_name if signal else None,
               "timestamp": str(signal.timestamp) if signal and signal.timestamp is not None else None}
        self.rejections.append(rec)
        log.warning("risk rejection", extra=rec)
        return RiskDecision(False, None, reason, mods or [])

    def _risk_positions(self, state: PortfolioState, exclude: Optional[str] = None) -> Dict[str, float]:
        return {s: v for s, v in state.positions.items()
                if s not in self.cash_equivalents and s != exclude and abs(v) > 1e-9}

    def sector(self, symbol: str) -> str:
        return self.sectors.get(symbol, symbol)

    def max_correlation(self, symbol: str, state: PortfolioState) -> Tuple[float, Optional[str]]:
        hist = state.price_history
        others = [s for s in self._risk_positions(state, exclude=symbol) if hist is not None and s in hist]
        if hist is None or symbol not in hist or not others:
            return 0.0, None
        rets = hist[[symbol, *others]].pct_change().iloc[-self.corr_window:]
        if len(rets.dropna()) < self.corr_window // 2:
            return 0.0, None
        corr = rets.corr()[symbol].drop(symbol)
        return float(corr.max()), str(corr.idxmax())

    def allowed_leverage(self, signal: Signal, state: PortfolioState) -> Tuple[float, Optional[str]]:
        """Leverage cap for this signal and the rule that set it (None when uncapped)."""
        if signal.strategy_name != LowVolBullStrategy.__name__:
            return self.default_leverage, "not a low-vol regime"
        if signal.metadata.get("uncertain"):
            return self.default_leverage, "regime uncertain"
        if state.circuit_breaker_status.active:
            return self.default_leverage, f"circuit breaker {state.circuit_breaker_status}"
        if len(self._risk_positions(state, exclude=signal.symbol)) + 1 >= self.leverage_max_positions:
            return self.default_leverage, f"{self.leverage_max_positions}+ positions open"
        if state.flicker_rate > self.flicker_threshold:
            return self.default_leverage, f"flicker rate {state.flicker_rate}"
        return self.max_leverage, None

    # ------------------------------------------------------------ validation
    def validate_signal(self, signal: Signal, state: PortfolioState) -> RiskDecision:
        """Approve, resize or reject a strategy signal. The risk manager has the final word."""
        status = state.circuit_breaker_status
        if status.flatten:
            return self._reject(signal, f"circuit breaker: {status}")
        if signal.direction is Direction.FLAT:
            return RiskDecision(True, signal)                 # reducing risk is always allowed

        mods: List[str] = []
        entry, stop = signal.entry_price, signal.stop_loss
        if stop is None or not math.isfinite(stop) or stop <= 0:
            return self._reject(signal, "no stop loss")
        if stop >= entry:
            atr_value = signal.metadata.get("atr_14")
            if not atr_value or not math.isfinite(atr_value):
                return self._reject(signal, f"stop {stop:.2f} at/above entry {entry:.2f}")
            # The strategy's stop is already through the price: a sell-stop there
            # can't be placed, so fall back to one ATR below entry.
            mods.append(f"stop {stop:.2f} >= entry {entry:.2f}: moved to entry - 1 ATR")
            stop = entry - atr_value
            if stop <= 0:
                return self._reject(signal, "stop below zero after adjustment", mods)

        held = self._risk_positions(state, exclude=signal.symbol)
        is_new = signal.symbol not in self._risk_positions(state)
        if is_new and len(held) >= self.max_positions:
            return self._reject(signal, f"max concurrent positions ({self.max_positions})", mods)
        if self.trades_today >= self.max_daily_trades:
            return self._reject(signal, f"max daily trades ({self.max_daily_trades})", mods)

        size_mult = 1.0
        corr, peer = self.max_correlation(signal.symbol, state)
        if corr > self.corr_reject:
            return self._reject(signal, f"correlation {corr:.2f} with {peer} > {self.corr_reject}", mods)
        if corr > self.corr_reduce:
            size_mult *= 0.5
            mods.append(f"correlation {corr:.2f} with {peer}: size x0.5")

        leverage = signal.leverage
        cap, why = self.allowed_leverage(signal, state)
        if leverage > cap:
            mods.append(f"leverage {leverage}x -> {cap}x ({why})")
            leverage = cap

        weight = signal.position_size_pct * leverage * size_mult
        stop_dist = (entry - stop) / entry
        limits = {
            "risk per trade": self.max_risk_per_trade / stop_dist,
            "overnight gap": self.max_gap_loss / (self.gap_multiple * stop_dist),
            "max single position": self.max_single_position,
            "max total exposure": self.max_total_exposure - sum(abs(v) for v in held.values()) / state.equity,
            "max sector exposure": self.max_sector_exposure - sum(
                abs(v) for s, v in held.items() if self.sector(s) == self.sector(signal.symbol)) / state.equity,
            "max portfolio leverage": self.max_leverage - sum(abs(v) for v in held.values()) / state.equity,
        }
        for name, limit in limits.items():
            if weight > limit:
                mods.append(f"{name}: weight {weight:.3f} -> {max(limit, 0.0):.3f}")
                weight = max(limit, 0.0)
        if status.size_mult < 1.0:
            weight *= status.size_mult
            mods.append(f"circuit breaker {status}: size x{status.size_mult}")
        if state.equity_known and weight * state.equity < self.min_position_value:
            return self._reject(signal, f"position ${weight * state.equity:,.0f} below minimum "
                                        f"${self.min_position_value:,.0f}", mods)

        modified = replace(signal, stop_loss=stop, leverage=leverage, position_size_pct=weight / leverage,
                           metadata={**signal.metadata, "risk_modifications": mods})
        if mods:
            log.info("risk modification", extra={"symbol": signal.symbol, "modifications": mods,
                                                 "target_weight": round(weight, 4)})
        return RiskDecision(True, modified, None, mods)

    def validate_order(
        self,
        symbol: str,
        side: str,
        notional: float,
        state: PortfolioState,
        quote: Optional[Tuple[float, float]] = None,     # (bid, ask)
        tradable: bool = True,
        now: Optional[float] = None,
    ) -> OrderCheck:
        """Pre-submission checks: tradable, buying power, spread, duplicates."""
        now = datetime.now().timestamp() if now is None else now
        side = side.lower()
        reason = None
        if not tradable:
            reason = "asset not tradable"
        elif side == "buy" and notional > state.buying_power:
            reason = f"insufficient buying power: ${notional:,.2f} > ${state.buying_power:,.2f}"
        elif quote is not None:
            bid, ask = quote
            mid = (bid + ask) / 2
            if bid <= 0 or ask <= 0 or mid <= 0:
                reason = f"invalid quote bid={bid} ask={ask}"
            elif (ask - bid) / mid > self.max_spread:
                reason = f"spread {(ask - bid) / mid:.2%} > {self.max_spread:.2%}"
        last = self._recent_orders.get((symbol, side))
        if reason is None and last is not None and now - last < self.duplicate_window:
            reason = f"duplicate {side} {symbol} within {self.duplicate_window}s"
        if reason:
            rec = {"symbol": symbol, "side": side, "notional": round(notional, 2), "reason": reason}
            self.rejections.append(rec)
            log.warning("order rejected by risk", extra=rec)
            return OrderCheck(False, reason)
        self._recent_orders[(symbol, side)] = now
        return OrderCheck(True)

    # ------------------------------------------------------ per-bar pipeline
    def record_trades(self, n: int, timestamp: Optional[pd.Timestamp] = None) -> None:
        day = pd.Timestamp(timestamp).date() if timestamp is not None else datetime.now().date()
        if day != self._trades_day:
            self._trades_day, self.trades_today = day, 0
        self.trades_today += n

    def portfolio_state(self, equity: float, weights: Dict[str, float], timestamp=None, flicker_rate: int = 0,
                        price_history: Optional[pd.DataFrame] = None, cash: Optional[float] = None,
                        buying_power: Optional[float] = None) -> PortfolioState:
        positions = {s: w * equity for s, w in weights.items()}
        cash = equity - sum(positions.values()) if cash is None else cash
        b = self.breaker
        return PortfolioState(
            equity=equity, cash=cash,
            buying_power=buying_power if buying_power is not None else max(cash, 0.0) + equity,  # Reg T ~2x
            positions=positions, daily_pnl=b.daily_dd * (b.day_start or 0), weekly_pnl=b.weekly_dd * (b.week_start or 0),
            peak_equity=b.peak, drawdown=b.peak_dd, circuit_breaker_status=b.check(), flicker_rate=flicker_rate,
            timestamp=timestamp, price_history=price_history,
        )

    def process(
        self,
        signal: Optional[Signal],
        equity: Optional[float],
        weights: Dict[str, float],
        risk_symbol: str,
        safe_symbol: Optional[str] = None,
        timestamp: Optional[pd.Timestamp] = None,
        regime: Optional[RegimeState] = None,
        price_history: Optional[pd.DataFrame] = None,
        day_start_equity: Optional[float] = None,
    ) -> RiskOutcome:
        """Mark equity, update breakers, validate the signal and return final target weights.

        Halts/locks flatten everything (defensive sleeve included). Otherwise the
        defensive sleeve takes whatever the risk position leaves, unlevered.
        """
        events: List[BreakerEvent] = []
        if equity is not None:
            held = [s for s, w in weights.items() if abs(w) > 1e-9]
            events = self.breaker.update(equity, timestamp, held, regime, signal.strategy_name if signal else None,
                                         day_start_equity)
        status = self.breaker.check()
        decision = None
        risk_w = 0.0
        if signal is not None:
            state = self.portfolio_state(equity or 1.0, weights, timestamp, regime.flicker_rate if regime else 0,
                                         price_history)
            state.equity_known = bool(equity)
            decision = self.validate_signal(signal, state)
            risk_w = decision.modified_signal.target_weight if decision.approved else 0.0
        out = {risk_symbol: risk_w}
        if safe_symbol and safe_symbol != risk_symbol:
            out[safe_symbol] = 0.0 if status.flatten else float(np.clip(1.0 - max(risk_w, 0.0), 0.0, 1.0))
        return RiskOutcome(decision, out, status, events)

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> Dict:
        return {"breaker": self.breaker.state_dict(), "trades_day": str(self._trades_day) if self._trades_day else None,
                "trades_today": self.trades_today}

    def load_state(self, state: Dict) -> None:
        if state.get("breaker"):
            self.breaker.load_state(state["breaker"])
        if state.get("trades_day"):
            self._trades_day = pd.Timestamp(state["trades_day"]).date()
            self.trades_today = int(state.get("trades_today", 0))
