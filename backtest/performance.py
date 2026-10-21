"""Performance analytics for allocation backtests.

A "trade" is one rebalance; its P&L is the portfolio's equity change until the
next rebalance. Regime and confidence attributions use the regime known when
each bar (or trade) was entered, never the regime detected afterwards.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest.backtester import BacktestResult

TRADING_DAYS = 252
CONFIDENCE_BINS = [0.0, 0.5, 0.6, 0.7, 1.0 + 1e-9]
CONFIDENCE_LABELS = ["< 50%", "50-60%", "60-70%", "70%+"]


# ------------------------------------------------------------ return stats
def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    growth = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else float("nan")


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - ((1 + rf) ** (1 / TRADING_DAYS) - 1)
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if len(excess) > 1 and sd > 0 else float("nan")


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - ((1 + rf) ** (1 / TRADING_DAYS) - 1)
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else float("nan")


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Each stretch below a prior high: depth and duration in bars (peak to recovery, or to the end)."""
    dd = drawdown_series(returns).to_numpy()
    rows, start = [], None
    for t, v in enumerate(dd):
        if v < 0 and start is None:
            start = t
        elif v >= 0 and start is not None:
            rows.append((start, t, True))
            start = None
    if start is not None:
        rows.append((start, len(dd), False))
    return pd.DataFrame(
        [{"start": returns.index[s - 1] if s else returns.index[0], "depth": float(dd[s:e].min()),
          "duration_bars": e - s, "recovered": rec} for s, e, rec in rows],
        columns=["start", "depth", "duration_bars", "recovered"],
    )


def max_drawdown_duration(returns: pd.Series) -> int:
    """Bars spent underwater in the deepest drawdown."""
    ep = drawdown_episodes(returns)
    return int(ep.loc[ep["depth"].idxmin(), "duration_bars"]) if not ep.empty else 0


def longest_underwater(returns: pd.Series) -> int:
    ep = drawdown_episodes(returns)
    return int(ep["duration_bars"].max()) if not ep.empty else 0


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return cagr(returns) / abs(mdd) if mdd < 0 else float("nan")


def max_consecutive_losses(returns: pd.Series) -> int:
    longest = run = 0
    for r in returns:
        run = run + 1 if r < 0 else 0
        longest = max(longest, run)
    return longest


def worst_period(returns: pd.Series, freq: str) -> float:
    """Worst compounded return over calendar periods (`W` week, `M` month)."""
    if returns.empty:
        return float("nan")
    return float((1 + returns).groupby(returns.index.to_period(freq)).prod().min() - 1)


def core_metrics(returns: pd.Series) -> Dict[str, float]:
    returns = returns.dropna()
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns),
        "ann_vol": ann_vol(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "calmar": calmar(returns),
        "max_drawdown": max_drawdown(returns),
        "max_dd_duration_bars": max_drawdown_duration(returns),
    }


def worst_case(returns: pd.Series) -> Dict[str, float]:
    return {
        "worst_day": float(returns.min()),
        "worst_week": worst_period(returns, "W"),
        "worst_month": worst_period(returns, "M"),
        "max_consecutive_losses": max_consecutive_losses(returns),
        "longest_underwater_bars": longest_underwater(returns),
    }


# -------------------------------------------------------------- trade stats
def trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"total_trades": 0, "win_rate": float("nan"), "avg_win": float("nan"), "avg_loss": float("nan"),
                "profit_factor": float("nan"), "avg_holding_bars": float("nan")}
    pnl = trades["pnl_pct"]
    gains, losses = trades.loc[trades["pnl"] > 0, "pnl"].sum(), -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    return {
        "total_trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_win": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else float("nan"),
        "avg_loss": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else float("nan"),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "avg_holding_bars": float(trades["holding_bars"].mean()),
    }


def _trade_of_bar(trades: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Position (row number) of the trade active at each bar's close, -1 before the first fill."""
    starts = index.searchsorted(trades.index)
    owner = np.full(len(index), -1)
    for k, s in enumerate(starts):
        owner[s:] = k
    return pd.Series(owner, index=index)


def _group_table(returns: pd.Series, bar_group: pd.Series, trades: pd.DataFrame, trade_group: pd.Series,
                 order: Optional[List[str]] = None) -> pd.DataFrame:
    rows = {}
    keys = order or sorted(set(bar_group.dropna()) | set(trade_group.dropna()))
    for key in keys:
        r = returns[bar_group == key]
        t = trades[trade_group == key]
        rows[key] = {
            "pct_time": len(r) / len(returns) if len(returns) else float("nan"),
            "return_contribution": total_return(r) if len(r) else 0.0,
            "trades": len(t),
            "avg_trade_pnl": float(t["pnl_pct"].mean()) if len(t) else float("nan"),
            "win_rate": float((t["pnl_pct"] > 0).mean()) if len(t) else float("nan"),
            "sharpe": sharpe(r),
        }
    return pd.DataFrame(rows).T


def regime_table(result: BacktestResult, column: str = "regime") -> pd.DataFrame:
    """Per-regime (or per-strategy) attribution. Bar t's return is earned under the regime known at t-1."""
    known = result.regimes[column].shift(1)
    trade_key = result.regimes[column].reindex(result.trades["signal_date"]).to_numpy() if not result.trades.empty else []
    return _group_table(result.returns, known, result.trades, pd.Series(trade_key, index=result.trades.index))


def confidence_table(result: BacktestResult) -> pd.DataFrame:
    trades = result.trades
    if trades.empty:
        return pd.DataFrame()
    conf = result.regimes["confidence"].reindex(trades["signal_date"]).to_numpy()
    bucket = pd.Series(pd.cut(conf, CONFIDENCE_BINS, labels=CONFIDENCE_LABELS, right=False), index=trades.index)
    owner = _trade_of_bar(trades, result.returns.index)
    bar_bucket = pd.Series([bucket.iat[k] if k >= 0 else None for k in owner], index=result.returns.index)
    table = _group_table(result.returns, bar_bucket, trades, bucket, order=CONFIDENCE_LABELS)
    return table[["trades", "sharpe", "win_rate", "avg_trade_pnl"]]


# --------------------------------------------------------------- benchmarks
def _row(name: str, returns: pd.Series, trades: pd.DataFrame) -> Dict:
    return {"strategy": name, **core_metrics(returns), "trades": len(trades)}


def benchmark_table(result: BacktestResult) -> pd.DataFrame:
    rows = [_row(f"HMM regime ({result.symbol})", result.returns, result.trades)]
    rows += [_row(name, b.returns, b.trades) for name, b in result.benchmarks.items()]
    if result.random_benchmarks:
        from backtest.backtester import trade_log

        rnd = pd.DataFrame([_row("random", s.returns, trade_log(s)) for s in result.random_benchmarks])
        num = rnd.drop(columns="strategy")
        rows.append({"strategy": f"Random x{len(rnd)} (mean)", **num.mean().to_dict()})
        rows.append({"strategy": f"Random x{len(rnd)} (std)", **num.std().to_dict()})
    return pd.DataFrame(rows).set_index("strategy")


# ------------------------------------------------------------------ report
def build_report(result: BacktestResult) -> Dict:
    report: Dict = {
        "symbol": result.symbol,
        "period": (result.returns.index[0].date(), result.returns.index[-1].date()),
        "core": core_metrics(result.returns),
        "trades": trade_stats(result.trades),
        "worst_case": worst_case(result.returns),
        "regimes": regime_table(result, "regime"),
        "strategies": regime_table(result, "strategy"),
        "confidence": confidence_table(result),
        "windows": pd.DataFrame([w.describe(result.features_index) for w in result.windows]),
        "events": pd.Series([e["event"] for e in result.events], dtype=object).value_counts().to_dict(),
    }
    if result.benchmarks or result.random_benchmarks:
        report["benchmarks"] = benchmark_table(result)
    return report


_PCT = {"total_return", "cagr", "ann_vol", "max_drawdown", "win_rate", "avg_win", "avg_loss", "worst_day",
        "worst_week", "worst_month", "pct_time", "return_contribution", "avg_trade_pnl"}


_INT = {"trades", "total_trades", "max_dd_duration_bars", "max_consecutive_losses", "longest_underwater_bars"}


def _fmt(key: str, value) -> str:
    if key in _INT and isinstance(value, (float, np.floating)) and np.isfinite(value):
        value = int(round(value))
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—" if np.isnan(value) else "∞"
        return f"{value:.2%}" if key in _PCT else f"{value:,.2f}"
    return str(value)


def _frame_table(title: str, df: pd.DataFrame, index_name: str):
    from rich.table import Table

    table = Table(title=title, title_justify="left", header_style="bold")
    table.add_column(index_name)
    for col in df.columns:
        table.add_column(col.replace("_", " ").title(), justify="right")
    for key, row in df.iterrows():
        table.add_row(str(key), *[_fmt(c, row[c]) for c in df.columns])
    return table


def render_report(report: Dict, console=None) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    start, end = report["period"]
    console.rule(f"[bold]{report['symbol']} walk-forward backtest  {start} → {end}")
    sections = ("core", "trades", "worst_case")
    kv = Table.grid(padding=(0, 3))
    for _ in sections:
        kv.add_column()
    kv.add_row(*[_kv(report[s], s) for s in sections])
    console.print(kv)
    console.print(_frame_table("By regime (labels are return-sorted)", report["regimes"], "Regime"))
    console.print(_frame_table("By strategy (volatility environment)", report["strategies"], "Strategy"))
    if not report["confidence"].empty:
        console.print(_frame_table("By regime confidence at entry", report["confidence"], "Confidence"))
    if "benchmarks" in report:
        console.print(_frame_table("Benchmark comparison", report["benchmarks"], "Strategy"))
    if report["events"]:
        console.print("Risk events: " + ", ".join(f"{k} ×{v}" for k, v in report["events"].items()))


def _kv(values: Dict, title: str):
    from rich.table import Table

    t = Table(title=title.replace("_", " ").title(), title_justify="left", show_header=False, box=None)
    t.add_column(style="dim")
    t.add_column(justify="right")
    for k, v in values.items():
        t.add_row(k.replace("_", " "), _fmt(k, v))
    return t


def save_report(result: BacktestResult, report: Dict, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = result.sim
    curve = pd.concat([sim.equity, sim.cash, result.returns, result.asset_returns.rename("asset_return"),
                       sim.shares.add_prefix("shares_")], axis=1)
    files = {
        "equity_curve.csv": curve,
        "trade_log.csv": result.trades,
        "fills.csv": sim.fills,
        "regime_history.csv": result.regimes,
        "walk_forward_windows.csv": report["windows"],
    }
    if "benchmarks" in report:
        files["benchmark_comparison.csv"] = report["benchmarks"]
    paths = []
    for name, df in files.items():
        path = out_dir / name
        df.to_csv(path, index=name not in {"fills.csv", "walk_forward_windows.csv"})
        paths.append(path)
    return paths
