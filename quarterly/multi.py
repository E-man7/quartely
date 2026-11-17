"""Several strategies in one bot (config `strategies: [...]`), e.g. the silver bullet models and NY open v2.

Each strategy keeps its own engine, trade counts and order prefix. They share the
account, so an index holds at most one position: a strategy cannot enter an index
another strategy is already in, and when two signal the same index on the same
minute the one listed first wins. Opt-outs (NY open) only close the trade of the
strategy that opened it; every strategy flattens at its own `flat_at`.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional

import pandas as pd

from monitoring.logger import get_logger
from quarterly.config import QuarterlyConfig
from quarterly.signals import Decision

log = get_logger(__name__)


def sub_config(cfg: QuarterlyConfig, strategy: str) -> QuarterlyConfig:
    return dataclasses.replace(cfg, strategy=strategy, strategies=(strategy,))


class MultiEngine:
    def __init__(self, cfg: QuarterlyConfig, verbose: bool = True):
        from quarterly.triad import make_engine

        self.cfg = cfg
        self.strategies = cfg.strategies
        self.engines = {s: make_engine(sub_config(cfg, s), verbose=verbose) for s in self.strategies}
        self.owner: Dict[str, str] = {}               # symbol -> strategy whose trade is open / was last entered
        self._signalled: Dict[str, str] = {}          # symbol -> strategy of the latest entry decision

    # The live loop and backtest read these from "the" engine
    @property
    def primary(self):
        return self.engines[self.strategies[0]]

    @property
    def day(self):
        return self.primary.day

    @property
    def days(self):
        return self.primary.days

    @property
    def verbose(self) -> bool:
        return self.primary.verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        for e in self.engines.values():
            e.verbose = value

    def _new_day(self, day) -> None:
        for e in self.engines.values():
            e._new_day(day)

    @property
    def flatten_at(self) -> Optional[pd.Timestamp]:
        times = [e.flatten_at for e in self.engines.values() if e.flatten_at is not None]
        return min(times) if times else None

    # ---------------------------------------------------------- trade counts
    def trades(self, symbol: str, strategy: str) -> int:
        d = self.engines[strategy].days.get(symbol)
        return d.trades if d is not None else 0

    def set_trades(self, symbol: str, n: int, strategy: Optional[str] = None) -> None:
        for s in ([strategy] if strategy else self.strategies):
            self.engines[s].set_trades(symbol, n)

    def record_trade(self, symbol: str, strategy: Optional[str] = None) -> None:
        s = strategy or self._signalled.get(symbol) or self.strategies[0]
        self.engines[s].record_trade(symbol)
        self.owner[symbol] = s

    # ------------------------------------------------------------------ step
    def on_minute(self, ts, bars, has_position: Callable[[str], bool] = lambda s: False) -> List[Decision]:
        taken: set = set()
        out: List[Decision] = []
        for s, e in self.engines.items():
            decs = e.on_minute(ts, bars, has_position=lambda sym: has_position(sym) or sym in taken)
            for d in decs:
                d.strategy = s
                if d.action:
                    if d.symbol in taken:          # same index, same minute: the earlier strategy wins
                        d.reasons.append("another strategy is entering this index on this minute")
                        d.action = d.entry = d.stop = d.target = d.risk_per_share = None
                        continue
                    taken.add(d.symbol)
                    self._signalled[d.symbol] = s
            out += decs
        return out

    def exit_signal(self, symbol: str, side: str) -> Optional[str]:
        e = self.engines.get(self.owner.get(symbol, ""))
        fn = getattr(e, "exit_signal", None) if e is not None else None
        return fn(symbol, side) if fn is not None else None

    def trade_context(self, dec: Decision) -> Dict:
        return {**self.engines[dec.strategy].trade_context(dec), "strategy": dec.strategy}

    def state_summary(self) -> Dict:
        return {s: e.state_summary() for s, e in self.engines.items()}
