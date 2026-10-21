"""Stress tests for the walk-forward allocation strategy.

Every scenario re-runs the strategy on modified data, so the features, the
forward-filtered regimes, the strategy layer and the risk manager all react to
the shock. To keep Monte Carlo affordable, each window's HMM is reused from the
base run rather than retrained: a shock changes what the models *see*, not how
they were trained.

* crash injection: `shocks_per_sim` single-day gaps of -5%..-15% per simulation
* gap risk: one overnight gap of 2-5x ATR per simulation; expected vs actual loss
* regime misclassification: state ids shuffled within each window, so every
  regime gets the wrong strategy. Risk management must contain the damage.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from backtest.backtester import BREAKER_EVENTS, BacktestResult, WalkForwardBacktester
from backtest.performance import max_drawdown, total_return
from core.hmm_engine import RegimeState
from data.feature_engineering import atr
from monitoring.logger import get_logger

log = get_logger(__name__)


def shock_bars(bars: Dict[str, pd.DataFrame], symbols: Sequence[str], dates: Sequence[pd.Timestamp],
               sizes: Sequence[float]) -> Dict[str, pd.DataFrame]:
    """Overnight gaps: every OHLC price from each date onward is scaled by (1 + size)."""
    out = dict(bars)
    for sym in set(symbols):
        df = bars[sym].copy()
        factor = np.ones(len(df))
        for d, size in zip(dates, sizes):
            factor[int(df.index.searchsorted(d)):] *= 1 + size
        for col in ("open", "high", "low", "close"):
            if col in df:
                df[col] = df[col] * factor
        out[sym] = df
    return out


class StressTester:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.cfg = settings.get("stress_test", {})
        self.backtester = WalkForwardBacktester(settings)
        uni = settings.get("universe", {})
        self.shocked_symbols = [s for s in {uni.get("signal_symbol"), uni.get("risk_symbol")} if s]
        breakers = settings.get("risk", {}).get("breakers", {})
        # With working risk management no drawdown episode (peak to trough between lock
        # resets) should run much past the peak lock plus one day's halt threshold, since
        # the lock fills at the next open.
        self.containment_limit = -(breakers.get("peak_halt_dd", 0.10) + breakers.get("daily_halt_dd", 0.03))

    def _rng(self, offset: int = 0) -> np.random.RandomState:
        return np.random.RandomState(self.cfg.get("random_state", 7) + offset)

    def _replay(self, bars, base: BacktestResult, **kw) -> BacktestResult:
        return self.backtester.replay(bars, base.windows, **kw)

    # ---------------------------------------------------------------- crash
    def crash_injection(self, base: BacktestResult, bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cfg = self.cfg.get("crash", {})
        lo, hi = cfg.get("size_range", [-0.15, -0.05])
        k = cfg.get("shocks_per_sim", 10)
        index = base.equity.index
        rng = self._rng(1)
        rows = []
        for i in range(self.cfg.get("n_simulations", 100)):
            pos = np.sort(rng.choice(np.arange(1, len(index)), size=min(k, len(index) - 1), replace=False))
            dates, sizes = index[pos], rng.uniform(lo, hi, len(pos))
            res = self._replay(shock_bars(bars, self.shocked_symbols, dates, sizes), base)
            breaker_dates = {pd.Timestamp(e["date"]) for e in res.events if e["event"] in BREAKER_EVENTS}
            fired = [any(index[min(p + j, len(index) - 1)] in breaker_dates for j in (0, 1)) for p in pos]
            rows.append({"sim": i, "max_drawdown": max_drawdown(res.returns), "total_return": total_return(res.returns),
                         "worst_day": float(res.returns.min()), "shocks": len(pos),
                         "shocks_with_breaker": int(sum(fired)), "breaker_fired": any(fired)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ gap
    def gap_risk(self, base: BacktestResult, bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cfg = self.cfg.get("gap", {})
        lo, hi = cfg.get("atr_multiples", [2, 5])
        sym = base.symbol
        index = base.equity.index
        atr_series = (atr(bars[sym]) / bars[sym]["close"]).reindex(index).ffill()
        rng = self._rng(2)
        rows = []
        for i in range(self.cfg.get("n_simulations", 100)):
            p = int(rng.randint(1, len(index) - 6))
            mult = rng.uniform(lo, hi)
            gap = -mult * float(atr_series.iloc[p - 1])
            res = self._replay(shock_bars(bars, self.shocked_symbols, [index[p]], [gap]), base)
            exposure = float(base.weights[sym].iloc[p - 1])   # identical in both runs before the gap
            rows.append({
                "sim": i, "date": index[p], "atr_multiple": mult, "gap_pct": gap, "exposure": exposure,
                "expected_loss": exposure * gap,
                "actual_loss_day": float(res.equity.iloc[p] / base.equity.iloc[p] - 1),
                "actual_loss_5d": float(res.equity.iloc[p + 5] / base.equity.iloc[p + 5] - 1),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------- misclassification
    def misclassification(self, base: BacktestResult, bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cfg = self.cfg.get("misclassification", {})
        labels = {w.number: w.engine.labels for w in base.windows}
        rows = []
        for i in range(cfg.get("n_simulations", 20)):
            rng = self._rng(100 + i)
            perms = {w.number: rng.permutation(w.engine.n_states) for w in base.windows}

            def shuffle(states: List[RegimeState], owner: List[int]) -> List[RegimeState]:
                out = []
                for s, w in zip(states, owner):
                    perm = perms[w]
                    probs = np.empty_like(s.state_probabilities)
                    probs[perm] = s.state_probabilities
                    out.append(replace(s, state_id=int(perm[s.state_id]), label=labels[w][perm[s.state_id]],
                                       state_probabilities=probs))
                return out

            res = self._replay(bars, base, state_transform=shuffle)
            # The breaker's own drawdown restarts after each lock is released, so it measures
            # individual episodes; the whole-path max drawdown compounds across episodes.
            episode_dd = float(res.regimes["drawdown"].min())
            rows.append({"sim": i, "max_drawdown": max_drawdown(res.returns), "worst_episode_drawdown": episode_dd,
                         "total_return": total_return(res.returns), "worst_day": float(res.returns.min()),
                         "breaker_events": sum(e["event"] in BREAKER_EVENTS for e in res.events),
                         "contained": episode_dd >= self.containment_limit})
        return pd.DataFrame(rows)

    def run(self, base: BacktestResult, bars: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out = {}
        for name, fn in (("crash", self.crash_injection), ("gap", self.gap_risk),
                         ("misclassification", self.misclassification)):
            log.info("stress test", extra={"scenario": name})
            out[name] = fn(base, bars)
        return out


def stress_summary(results: Dict[str, pd.DataFrame], base: BacktestResult, containment_limit: float) -> Dict:
    """Headline numbers per scenario, next to the unshocked strategy and buy-and-hold."""
    base_mdd = max_drawdown(base.returns)
    bh_mdd = max_drawdown(base.asset_returns)
    summary: Dict[str, Dict] = {}
    crash = results.get("crash")
    if crash is not None and not crash.empty:
        summary["Crash injection"] = {
            "simulations": len(crash),
            "mean_max_drawdown": crash["max_drawdown"].mean(),
            "worst_max_drawdown": crash["max_drawdown"].min(),
            "worst_day": crash["worst_day"].min(),
            "pct_sims_breaker_fired": crash["breaker_fired"].mean(),
            "pct_shocks_breaker_fired": crash["shocks_with_breaker"].sum() / crash["shocks"].sum(),
            "base_max_drawdown": base_mdd,
        }
    gap = results.get("gap")
    if gap is not None and not gap.empty:
        summary["Gap risk"] = {
            "simulations": len(gap),
            "mean_gap": gap["gap_pct"].mean(),
            "mean_expected_loss": gap["expected_loss"].mean(),
            "mean_actual_loss_day": gap["actual_loss_day"].mean(),
            "mean_actual_loss_5d": gap["actual_loss_5d"].mean(),
            "worst_actual_loss_5d": gap["actual_loss_5d"].min(),
            "actual_over_expected": gap["actual_loss_day"].sum() / gap["expected_loss"].sum()
            if gap["expected_loss"].sum() else float("nan"),
        }
    mis = results.get("misclassification")
    if mis is not None and not mis.empty:
        summary["Regime misclassification"] = {
            "simulations": len(mis),
            "mean_max_drawdown": mis["max_drawdown"].mean(),
            "worst_max_drawdown": mis["max_drawdown"].min(),
            "worst_episode_drawdown": mis["worst_episode_drawdown"].min(),
            "mean_total_return": mis["total_return"].mean(),
            "pct_contained": mis["contained"].mean(),
            "containment_limit": containment_limit,
            "base_max_drawdown": base_mdd,
            "base_worst_episode_drawdown": float(base.regimes["drawdown"].min()),
            "buy_hold_max_drawdown": bh_mdd,
        }
    return summary


def render_stress(summary: Dict, console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    console.rule("[bold]Stress tests")
    for title, values in summary.items():
        t = Table(title=title, title_justify="left", show_header=False)
        t.add_column(style="dim")
        t.add_column(justify="right")
        for k, v in values.items():
            pct = isinstance(v, float) and k not in {"actual_over_expected"}
            t.add_row(k.replace("_", " "), f"{v:.2%}" if pct else f"{v:.2f}" if isinstance(v, float) else str(v))
        console.print(t)
    mis = summary.get("Regime misclassification")
    if mis:
        ok = mis["pct_contained"] == 1.0
        console.print("[green]Risk management contained every misclassified run." if ok else
                      "[red]Some misclassified runs breached the containment limit: "
                      "risk management is not independent enough of the regime model.")
