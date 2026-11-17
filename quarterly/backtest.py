"""Backtest: replay historical 1-minute bars through the SAME StrategyEngine.

Fills and exits are simulated bar by bar:
* an entry signal at a bar's close fills at the symbol's next bar open (a market
  order sent after the close); stop and target are the bracket prices fixed at
  the signal, exactly as the live order sends them;
* from the fill bar on, a bar touching the stop exits at the stop (at the open
  if it gapped through), touching the target exits at the target; if both are
  hit in one bar, the stop is assumed;
* open trades close at the open of the first bar from the engine's
  `flatten_at` (triad930: 15:55; q3: session end with flat_at_session_end).
Sizing uses the live formula on the running equity, capped at max_leverage x
equity (live: also the account's buying power). R = P&L per share / risk
per share planned at the signal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quarterly.config import QuarterlyConfig
from quarterly.data import iter_minutes
from quarterly.execution import BP_HEADROOM, bracket_prices, capped_size, position_size
from quarterly.signals import SHORT, Bar, Decision
from quarterly.triad import make_engine
from quarterly.timemodel import trading_day


@dataclass
class Trade:
    symbol: str
    side: str
    signal_ts: pd.Timestamp
    entry_ts: Optional[pd.Timestamp] = None
    entry: Optional[float] = None
    stop: float = 0.0
    target: float = 0.0
    qty: int = 0
    risk_per_share: float = 0.0
    exit_ts: Optional[pd.Timestamp] = None
    exit: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    r: float = 0.0
    # Context at the signal, for charts and review
    partner: Optional[str] = None
    tdo: Optional[float] = None
    tdo_source: Optional[str] = None
    session_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    smt_ts: Optional[pd.Timestamp] = None
    pivot_ts: Optional[pd.Timestamp] = None
    pivot_price: Optional[float] = None
    range_high: Optional[float] = None         # triad930: the 8am range
    range_low: Optional[float] = None
    model: Optional[str] = None                # silverbullet: reversal | continuation
    strategy: Optional[str] = None             # when several strategies run together


class Backtester:
    def __init__(self, cfg: QuarterlyConfig, initial_equity: float = 100_000.0):
        self.cfg = cfg
        self.engine = make_engine(cfg, verbose=False)
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.pending: Dict[str, Trade] = {}
        self.open: Dict[str, Trade] = {}
        self.opt_out_next: set = set()
        self.trades: List[Trade] = []
        self.decisions: List[Decision] = []

    def _close(self, t: Trade, ts: pd.Timestamp, price: float, reason: str) -> None:
        sign = -1 if t.side == SHORT else 1
        t.exit_ts, t.exit, t.exit_reason = ts, price, reason
        t.pnl = sign * (price - t.entry) * t.qty
        t.r = sign * (price - t.entry) / t.risk_per_share
        self.equity += t.pnl
        del self.open[t.symbol]

    def _exits(self, bar: Bar) -> None:
        t = self.open.get(bar.symbol)
        if t is None:
            return
        if t.side == SHORT:
            hit_stop, hit_target = bar.high >= t.stop, bar.low <= t.target
            gap_stop = bar.open >= t.stop
        else:
            hit_stop, hit_target = bar.low <= t.stop, bar.high >= t.target
            gap_stop = bar.open <= t.stop
        if hit_stop:                                  # stop first when both are touched
            self._close(t, bar.ts, bar.open if gap_stop else t.stop, "stop")
        elif hit_target:
            gap_target = bar.open <= t.target if t.side == SHORT else bar.open >= t.target
            self._close(t, bar.ts, bar.open if gap_target else t.target, "target")

    def run(self, bars: pd.DataFrame) -> List[Trade]:
        for ts, minute in iter_minutes(bars):
            for sym, bar in minute.items():
                t = self.pending.pop(sym, None)
                if t is not None:                     # market order fills at the next bar open
                    t.entry_ts, t.entry = bar.ts, bar.open
                    self.open[sym] = t
                    self.trades.append(t)
                self._exits(bar)
                flat = self.engine.flatten_at
                if sym in self.open and flat is not None and trading_day(ts) == self.engine.day and ts >= flat:
                    self._close(self.open[sym], ts, bar.open, "flatten")
                if sym in self.opt_out_next:                  # opted out at the last close: market order fills now
                    self.opt_out_next.discard(sym)
                    if sym in self.open:
                        self._close(self.open[sym], ts, bar.open, "opt_out")
            decisions = self.engine.on_minute(ts, minute, has_position=lambda s: s in self.open or s in self.pending)
            exit_signal = getattr(self.engine, "exit_signal", None)
            if exit_signal is not None:
                for sym, t in self.open.items():
                    if exit_signal(sym, t.side):
                        self.opt_out_next.add(sym)
            for dec in decisions:
                if dec.action is None:
                    continue
                self.decisions.append(dec)
                qty = capped_size(position_size(self.equity, self.cfg.risk_pct, dec.risk_per_share), dec.entry,
                                  BP_HEADROOM * self.cfg.max_leverage * self.equity)
                if qty <= 0:
                    continue
                stop, target = bracket_prices(dec.action, dec.stop, dec.target)
                self.pending[dec.symbol] = Trade(
                    dec.symbol, dec.action, ts, stop=stop, target=target, qty=qty, risk_per_share=dec.risk_per_share,
                    **self.engine.trade_context(dec))
                self.engine.record_trade(dec.symbol)
        for t in list(self.open.values()):             # mark open trades at the last close
            last = bars[bars["symbol"] == t.symbol].iloc[-1]
            self._close(t, last["ts"], float(last["close"]), "end_of_data")
        return self.trades


def _max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(values)])
    return float((curve - np.maximum.accumulate(curve)).min())


def report(trades: List[Trade], initial_equity: float = 100_000.0) -> pd.DataFrame:
    """Per symbol (and ALL): trades, win rate, average R, total R, max drawdown in R and dollars."""
    rows = []
    df = pd.DataFrame([asdict(t) for t in trades])
    groups = [(sym, g) for sym, g in df.groupby("symbol")] if not df.empty else []
    for name, g in groups + ([("ALL", df)] if not df.empty else []):
        g = g.sort_values("exit_ts")
        rows.append({
            "symbol": name, "trades": len(g), "longs": int((g["side"] != SHORT).sum()),
            "shorts": int((g["side"] == SHORT).sum()), "win_rate": float((g["r"] > 0).mean()),
            "avg_r": float(g["r"].mean()), "total_r": float(g["r"].sum()),
            "max_dd_r": _max_drawdown(g["r"].to_numpy()), "pnl": float(g["pnl"].sum()),
            "max_dd_usd": _max_drawdown(g["pnl"].to_numpy()),
            "max_dd_pct": _max_drawdown(g["pnl"].to_numpy()) / initial_equity,
        })
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
