"""Walk-forward allocation backtester.

Allocation-based: every bar the strategy sets a target allocation from the
detected volatility regime and the portfolio rebalances, in whole shares, only
when the target moves more than `strategy.rebalance_threshold`. There are no
per-trade entries/exits or stops here (stops are for live trading).

Walk-forward: each window trains a fresh HMM (with BIC model selection) on the
trailing `in_sample` bars, then runs the forward filter over the next
`out_of_sample` bars, stepping `step` bars. Features are causal (row t uses
bars <= t; see tests/test_look_ahead.py) and a signal at bar N's close fills
at bar N + `fill_delay`'s open.

Allocation math, per symbol, at the fill bar's open:
    equity        = cash + sum(shares * price)
    target_shares = int(equity * target_allocation / price)
    cash         -= (target_shares - shares) * fill_price
Targets above 1.0 (leverage) drive cash negative: that is margin debt, and
`equity = cash + shares * price` still holds.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.hmm_engine import HMMRegimeEngine, RegimeInfo, RegimeState
from core.regime_strategies import Direction, Signal
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator
from data.feature_engineering import sma
from monitoring.logger import get_logger

log = get_logger(__name__)

TRADING_DAYS = 252
BREAKER_EVENTS = ("daily_reduce", "daily_halt", "weekly_reduce", "weekly_halt", "peak_lock")

# decide(t, timestamp, equity, current_weights) -> (target weights or None, per-bar info)
DecideFn = Callable[[int, pd.Timestamp, float, Dict[str, float]], Tuple[Optional[Dict[str, float]], Dict]]


# ------------------------------------------------------------- simulation
@dataclass
class SimConfig:
    initial_capital: float = 100_000.0
    slippage: float = 0.0005            # fraction of price, against us on every fill
    commission_per_order: float = 0.0
    fill_delay: int = 1                 # bars between signal (close) and fill (open)
    cash_rate: float = 0.0              # annual, earned on positive cash
    margin_rate: float = 0.0            # annual, paid on negative cash (margin debt)

    @classmethod
    def from_settings(cls, settings: Dict) -> "SimConfig":
        bt = settings.get("backtest", {})
        delay = int(bt.get("fill_delay", 1))
        if delay < 1:
            raise ValueError("backtest.fill_delay must be >= 1 (signals use the bar's close)")
        return cls(
            initial_capital=float(bt.get("initial_capital", 100_000)),
            slippage=float(bt.get("slippage_pct", 0.0005)),
            commission_per_order=float(bt.get("commission_per_order", 0.0)),
            fill_delay=delay,
            cash_rate=float(bt.get("cash_rate", 0.0)),
            margin_rate=float(bt.get("margin_rate", 0.0)),
        )


@dataclass
class SimResult:
    equity: pd.Series                   # marked at each close
    cash: pd.Series
    shares: pd.DataFrame
    weights: pd.DataFrame               # position value / equity at each close
    fills: pd.DataFrame                 # one row per symbol filled
    rebalances: pd.DataFrame            # one row per rebalance (fill bar)
    info: pd.DataFrame                  # decide() info per bar
    initial_capital: float

    @property
    def returns(self) -> pd.Series:
        r = self.equity.pct_change()
        r.iloc[0] = self.equity.iloc[0] / self.initial_capital - 1
        return r.rename("returns")


def simulate_allocation(opens: pd.DataFrame, closes: pd.DataFrame, decide: DecideFn, cfg: SimConfig) -> SimResult:
    """Bar-by-bar allocation simulation in whole shares with next-open fills."""
    symbols = list(closes.columns)
    idx = closes.index
    O = opens.reindex(index=idx, columns=symbols).to_numpy(float)
    C = closes.to_numpy(float)
    O = np.where(np.isfinite(O), O, C)
    r_pos = (1 + cfg.cash_rate) ** (1 / TRADING_DAYS) - 1
    r_neg = (1 + cfg.margin_rate) ** (1 / TRADING_DAYS) - 1

    cash = cfg.initial_capital
    shares = np.zeros(len(symbols), dtype=np.int64)
    pending: Optional[Tuple[int, Dict[str, float], pd.Timestamp, Dict]] = None
    eq, cash_hist, sh_hist, w_hist, infos, fills, rebalances = [], [], [], [], [], [], []

    for t in range(len(idx)):
        if pending is not None and pending[0] == t:
            _, target, signal_ts, meta = pending
            pending = None
            px = O[t]
            equity_open = cash + float(shares @ px)
            prev_w = shares * px / equity_open if equity_open else np.zeros(len(symbols))
            notional = cost = 0.0
            for j, sym in enumerate(symbols):
                target_shares = int(equity_open * target.get(sym, 0.0) / px[j])
                delta = target_shares - int(shares[j])
                if delta == 0:
                    continue
                fill_px = px[j] * (1 + cfg.slippage * np.sign(delta))
                cash -= delta * fill_px + cfg.commission_per_order
                shares[j] = target_shares
                slip = abs(delta) * px[j] * cfg.slippage
                notional += abs(delta) * fill_px
                cost += slip + cfg.commission_per_order
                fills.append({"date": idx[t], "signal_date": signal_ts, "symbol": sym,
                              "side": "BUY" if delta > 0 else "SELL", "shares": abs(delta),
                              "price": fill_px, "notional": abs(delta) * fill_px, "cost": slip + cfg.commission_per_order})
            if notional:
                rebalances.append({"date": idx[t], "bar": t, "signal_date": signal_ts,
                                   "equity_before": equity_open, "equity_after": cash + float(shares @ px),
                                   "notional": notional, "cost": cost,
                                   **{f"prev_w_{s}": float(prev_w[j]) for j, s in enumerate(symbols)},
                                   **{f"target_w_{s}": float(target.get(s, 0.0)) for s in symbols}, **meta})
        # Overnight financing on the cash balance (interest on cash, or margin interest)
        cash *= 1 + (r_pos if cash >= 0 else r_neg)
        equity = cash + float(shares @ C[t])
        weights = {s: float(shares[j] * C[t, j] / equity) if equity > 0 else 0.0 for j, s in enumerate(symbols)}
        target, meta = decide(t, idx[t], equity, weights)
        if target is not None:
            pending = (t + cfg.fill_delay, target, idx[t], meta)   # a newer signal supersedes a pending one
        eq.append(equity)
        cash_hist.append(cash)
        sh_hist.append(shares.copy())
        w_hist.append(weights)
        infos.append(meta)

    return SimResult(
        equity=pd.Series(eq, index=idx, name="equity"),
        cash=pd.Series(cash_hist, index=idx, name="cash"),
        shares=pd.DataFrame(sh_hist, index=idx, columns=symbols),
        weights=pd.DataFrame(w_hist, index=idx),
        fills=pd.DataFrame(fills),
        rebalances=pd.DataFrame(rebalances),
        info=pd.DataFrame(infos, index=idx),
        initial_capital=cfg.initial_capital,
    )


def trade_log(sim: SimResult) -> pd.DataFrame:
    """One row per rebalance; P&L is the portfolio's equity change until the next rebalance."""
    reb = sim.rebalances
    if reb.empty:
        return reb
    reb = reb.copy()
    end_equity = reb["equity_before"].shift(-1)
    end_equity.iloc[-1] = sim.equity.iloc[-1]
    end_bar = reb["bar"].shift(-1).fillna(len(sim.equity)).astype(int)
    reb["exit_date"] = [sim.equity.index[min(b, len(sim.equity) - 1)] for b in end_bar]
    reb["holding_bars"] = end_bar - reb["bar"]
    reb["pnl"] = end_equity - reb["equity_after"]
    reb["pnl_pct"] = end_equity / reb["equity_after"] - 1
    return reb.set_index("date")


# ---------------------------------------------------------------- results
@dataclass
class Window:
    number: int
    train_start: int                    # positions in the feature index
    test_start: int
    test_end: int                       # exclusive
    engine: Optional[HMMRegimeEngine] = None

    def describe(self, index: pd.Index) -> Dict:
        e = self.engine
        return {"window": self.number, "train_start": index[self.train_start].date(),
                "test_start": index[self.test_start].date(), "test_end": index[self.test_end - 1].date(),
                "n_regimes": e.n_states if e else None, "bic": round(e.bic, 1) if e else None}


@dataclass
class BacktestResult:
    symbol: str
    sim: SimResult
    trades: pd.DataFrame
    regimes: pd.DataFrame               # per-bar regime + strategy decisions
    windows: List[Window]
    features_index: pd.Index
    asset_returns: pd.Series            # close-to-close returns of the traded symbol
    events: List[Dict] = field(default_factory=list)
    benchmarks: Dict[str, "BacktestResult"] = field(default_factory=dict)
    random_benchmarks: List[SimResult] = field(default_factory=list)

    @property
    def equity(self) -> pd.Series:
        return self.sim.equity

    @property
    def returns(self) -> pd.Series:
        return self.sim.returns

    @property
    def weights(self) -> pd.DataFrame:
        return self.sim.weights

    def summary(self) -> Dict:
        from backtest.performance import build_report

        return build_report(self)


# ------------------------------------------------------------- backtester
class WalkForwardBacktester:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.uni = settings.get("universe", {})
        self.bt = settings.get("backtest", {})
        wf = self.bt.get("walk_forward", {})
        self.in_sample = int(wf.get("in_sample", 252))
        self.out_of_sample = int(wf.get("out_of_sample", 126))
        self.step = int(wf.get("step", self.out_of_sample))
        self.sim_cfg = SimConfig.from_settings(settings)
        # Each window trains on exactly `in_sample` bars, with full BIC selection
        self.hmm_settings = copy.deepcopy(settings)
        self.hmm_settings["hmm"]["min_train_size"] = self.in_sample
        self.hmm_settings["hmm"]["model_dir"] = None
        configured_min = settings.get("hmm", {}).get("min_train_size", self.in_sample)
        if self.in_sample < configured_min:
            log.warning("walk-forward in-sample shorter than hmm.min_train_size",
                        extra={"in_sample": self.in_sample, "min_train_size": configured_min})

    @property
    def signal_symbol(self) -> str:
        return self.uni.get("signal_symbol", "SPY")

    @property
    def traded(self) -> List[str]:
        risk, safe = self.uni.get("risk_symbol", "SPY"), self.uni.get("safe_symbol")
        return [risk, safe] if safe and safe != risk else [risk]

    # -------------------------------------------------------------- windows
    def features(self, bars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        feats = SignalGenerator(self.settings).features(bars[self.signal_symbol]).dropna()
        end = self.bt.get("end")
        return feats.loc[: pd.Timestamp(end)] if end else feats

    def plan_windows(self, index: pd.Index) -> List[Window]:
        start = self.bt.get("start")
        first = max(self.in_sample, int(index.searchsorted(pd.Timestamp(start))) if start else 0)
        if first >= len(index) - 1:
            raise ValueError(f"Not enough data: need {self.in_sample} in-sample bars before the first test bar")
        windows, k = [], 0
        for test_start in range(first, len(index), self.step):
            test_end = min(test_start + self.out_of_sample, len(index))
            windows.append(Window(k, test_start - self.in_sample, test_start, test_end))
            k += 1
        return windows

    def train(self, feats: pd.DataFrame, windows: Sequence[Window]) -> None:
        for w in windows:
            w.engine = HMMRegimeEngine(self.hmm_settings).fit(feats.iloc[w.train_start : w.test_start])
            log.info("walk-forward window", extra={k: str(v) for k, v in w.describe(feats.index).items()})

    @staticmethod
    def regimes(
        feats: pd.DataFrame, windows: Sequence[Window]
    ) -> Tuple[List[RegimeState], List[List[RegimeInfo]], List[int]]:
        """Filtered regimes for every test bar, from each window's own trained model.

        Filtering starts at the window's train start so the posterior is warmed
        up; row t depends only on rows <= t.
        """
        states, infos, owner = [], [], []
        for w in windows:
            n = w.test_end - w.test_start
            states += w.engine.regime_states(feats.iloc[w.train_start : w.test_end])[-n:]
            infos += [w.engine.regime_info] * n
            owner += [w.number] * n
        return states, infos, owner

    # ------------------------------------------------------------------ run
    def run(self, bars: Dict[str, pd.DataFrame]) -> BacktestResult:
        feats = self.features(bars)
        windows = self.plan_windows(feats.index)
        self.train(feats, windows)
        return self.replay(bars, windows, feats)

    def _prices(self, bars: Dict[str, pd.DataFrame], index: pd.Index) -> Tuple[pd.DataFrame, pd.DataFrame]:
        opens = pd.DataFrame({s: bars[s]["open"] for s in self.traded}).reindex(index).ffill()
        closes = pd.DataFrame({s: bars[s]["close"] for s in self.traded}).reindex(index).ffill()
        return opens, closes

    def replay(
        self,
        bars: Dict[str, pd.DataFrame],
        windows: Sequence[Window],
        feats: Optional[pd.DataFrame] = None,
        state_transform: Optional[Callable[[List[RegimeState], List[int]], List[RegimeState]]] = None,
    ) -> BacktestResult:
        """Simulate the strategy with already-trained windows (also used by stress tests)."""
        feats = self.features(bars) if feats is None else feats
        states, infos, owner = self.regimes(feats, windows)
        if state_transform is not None:
            states = state_transform(states, owner)
        index = feats.index[windows[0].test_start : windows[-1].test_end]
        opens, closes = self._prices(bars, index)

        gen = SignalGenerator(self.settings, backtest=True)
        risk_sym = gen.risk_symbol
        risk_bars = bars[risk_sym]
        bar_end = risk_bars.index.searchsorted(index, side="right")   # risk bars up to and incl. each bar
        lookback = gen.strategy.lookback
        active: Dict[str, Optional[List[RegimeInfo]]] = {"infos": None}
        events: List[Dict] = []

        def decide(t: int, ts: pd.Timestamp, equity: float, weights: Dict[str, float]):
            if infos[t] is not active["infos"]:
                active["infos"] = infos[t]
                gen.strategy.update_regime_infos(infos[t])
            state = states[t]
            hist = risk_bars.iloc[max(0, bar_end[t] - lookback) : bar_end[t]]
            sig = gen.from_regime(state, hist, equity, current_weights=weights)
            events.extend({"date": ts, "event": e} for e in sig.events)
            meta = {
                "window": owner[t], "regime": state.label, "regime_id": state.state_id, "vol_rank": state.vol_rank,
                "n_regimes": state.n_states, "confidence": state.probability, "confirmed": state.is_confirmed,
                "flickering": state.is_flickering, "strategy": sig.trade.strategy_name if sig.trade else None,
                "raw_allocation": sig.raw_weight, "target_allocation": sig.target_weights.get(risk_sym, 0.0),
                "signal_size": sig.trade.position_size_pct if sig.trade else None,
                "signal_leverage": sig.trade.leverage if sig.trade else None,
                "stop_distance": (1 - sig.trade.stop_loss / sig.trade.entry_price) if sig.trade else None,
                "risk_approved": sig.risk_decision.approved if sig.risk_decision else None,
                "risk_rejection": sig.risk_decision.rejection_reason if sig.risk_decision else None,
                "risk_modifications": "; ".join(sig.risk_decision.modifications) if sig.risk_decision else "",
                "breaker": sig.breaker, "drawdown": sig.drawdown, "halted": sig.halted, "events": ",".join(sig.events),
            }
            return (sig.target_weights if sig.rebalance else None), meta

        sim = simulate_allocation(opens, closes, decide, self.sim_cfg)
        regimes = sim.info.join(sim.weights.add_prefix("actual_w_"))
        result = BacktestResult(
            symbol=risk_sym, sim=sim, trades=trade_log(sim), regimes=regimes, windows=list(windows),
            features_index=feats.index, asset_returns=closes[risk_sym].pct_change().fillna(0.0), events=events,
        )
        log.info("backtest complete", extra={"symbol": risk_sym, "start": str(index[0].date()),
                                             "end": str(index[-1].date()), "windows": len(windows),
                                             "final_equity": round(float(sim.equity.iloc[-1]), 2),
                                             "rebalances": len(result.trades)})
        return result

    # ----------------------------------------------------------- benchmarks
    def run_benchmarks(self, bars: Dict[str, pd.DataFrame], base: BacktestResult, n_random: int = 100,
                       seed: int = 0) -> BacktestResult:
        """Attach buy-and-hold, 200-SMA trend and random-allocation benchmarks to `base`."""
        sym = base.symbol
        index = base.equity.index
        opens, closes = self._prices(bars, index)
        o, c = opens[[sym]], closes[[sym]]

        def wrap(sim: SimResult) -> BacktestResult:
            return BacktestResult(symbol=sym, sim=sim, trades=trade_log(sim), regimes=pd.DataFrame(),
                                  windows=[], features_index=index, asset_returns=base.asset_returns)

        def buy_and_hold(t, ts, equity, weights):
            return ({sym: 1.0} if t == 0 else None), {}

        trend = (bars[sym]["close"] > sma(bars[sym]["close"], 200)).reindex(index).fillna(False).to_numpy()

        def sma_trend(t, ts, equity, weights):
            changed = t == 0 or trend[t] != trend[t - 1]
            return ({sym: 1.0 if trend[t] else 0.0} if changed else None), {}

        base.benchmarks = {
            "Buy & Hold": wrap(simulate_allocation(o, c, buy_and_hold, self.sim_cfg)),
            "200 SMA Trend": wrap(simulate_allocation(o, c, sma_trend, self.sim_cfg)),
        }
        base.random_benchmarks = self._random_benchmarks(bars, base, n_random, seed)
        return base

    def _random_benchmarks(self, bars, base: BacktestResult, n: int, seed: int) -> List[SimResult]:
        """Random allocation changes at the strategy's rebalance frequency. Each change draws
        one of the strategy's own signals (size, leverage, strategy, stop distance) and goes
        through the same risk manager and rebalance band, so only the timing is random."""
        if base.trades.empty or n <= 0:
            return []
        index = base.equity.index
        opens, closes = self._prices(bars, index)
        cols = ["signal_size", "signal_leverage", "strategy", "stop_distance"]
        pool = base.regimes.loc[base.trades["signal_date"], cols].dropna().to_numpy()
        if not len(pool):
            return []
        p_change = len(base.trades) / len(index)
        threshold = self.settings.get("strategy", {}).get("rebalance_threshold", 0.10)
        risk_sym, safe_sym = base.symbol, self.uni.get("safe_symbol")
        px = closes[risk_sym].to_numpy()
        out = []
        for k in range(n):
            rng = np.random.RandomState(seed + k)
            change = rng.rand(len(index)) < p_change
            change[0] = True
            pick = np.where(change, rng.randint(0, len(pool), len(index)), -1)
            pick = pd.Series(pick).replace(-1, np.nan).ffill().astype(int).to_numpy()
            risk = RiskManager(self.settings, backtest=True)

            def decide(t, ts, equity, weights, pick=pick, risk=risk):
                size, lev, strat, stop_dist = pool[pick[t]]
                sig = Signal(risk_sym, Direction.LONG, 1.0, px[t], px[t] * (1 - stop_dist), None, size, lev,
                             -1, "RANDOM", 1.0, ts, "random benchmark", strat)
                o = risk.process(sig, equity, weights, risk_sym, safe_sym, ts)
                drift = max(abs(o.weights.get(s, 0.0) - weights.get(s, 0.0)) for s in o.weights)
                return (o.weights if o.events or drift > threshold else None), {}

            out.append(simulate_allocation(opens, closes, decide, self.sim_cfg))
        return out
