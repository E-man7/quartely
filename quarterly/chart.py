"""Interactive chart of backtest trades: one self-contained HTML file.

For each trade: candles of the symbol and its SMT partner (triad930: the
leader whose 8am sweep triggered it) for the trade day
(00:00-17:00 NY, extended to the exit for trades held overnight), lines for
entry / stop / target / True Day Open / NY Session True Open, markers for the
SMT bar, the swing pivot, the signal, the fill and the exit, and the entry
window shaded. Bars are embedded once per symbol and day, so only the days the
trades need are stored.

    python -m quarterly backtest ...        # writes chart_<tag>.html next to the trades CSV
    python -m quarterly chart --trades results/quarterly/trades_<tag>.csv
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quarterly.config import QuarterlyConfig
from quarterly.timemodel import NY, clock, trading_day
from quarterly.multi import sub_config
from quarterly.triad import chart_spans

TEMPLATE = Path(__file__).with_name("web") / "chart.html"
DAY_FROM, DAY_TO = (0, 0), (17, 0)          # NY wall-clock span shown for each day
MAX_EXTRA_DAYS = 3                           # trades held overnight: show at most this many more days


def chart_time(ts) -> int:
    """NY wall-clock as epoch seconds: the chart library displays UTC, so this shows NY times."""
    t = pd.Timestamp(ts).tz_convert(NY).tz_localize(None)
    return int((t - pd.Timestamp("1970-01-01")).total_seconds())


def _num(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    return round(float(x), 4)


def _key(symbol: str, d: date) -> str:
    return f"{symbol}|{d}"


def _days(trade: pd.Series) -> List[date]:
    start = trading_day(trade["signal_ts"])
    end = trading_day(trade["exit_ts"]) if pd.notna(trade.get("exit_ts")) else start
    days, d = [], start
    while d <= end and len(days) <= MAX_EXTRA_DAYS:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def build_chart_data(trades: pd.DataFrame, bars: pd.DataFrame, cfg: QuarterlyConfig,
                     report: Optional[pd.DataFrame] = None, title: str = "") -> Dict[str, Any]:
    trades = trades.copy()
    for col in ("signal_ts", "entry_ts", "exit_ts", "smt_ts", "pivot_ts"):
        if col in trades:
            trades[col] = pd.to_datetime(trades[col], utc=True).dt.tz_convert(NY)
    bars = bars.copy()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True).dt.tz_convert(NY)
    bars["date"] = bars["ts"].map(lambda t: t.date())
    wanted = {}
    for _, t in trades.iterrows():
        for d in _days(t):
            for sym in (t["symbol"], t.get("partner")):
                if isinstance(sym, str):
                    wanted[_key(sym, d)] = (sym, d)

    series: Dict[str, List[List[float]]] = {}
    grouped = {k: g for k, g in bars.groupby(["symbol", "date"])}
    for key, (sym, d) in wanted.items():
        g = grouped.get((sym, d))
        if g is None:
            continue
        lo = pd.Timestamp(d.year, d.month, d.day, *DAY_FROM, tz=NY)
        hi = pd.Timestamp(d.year, d.month, d.day, *DAY_TO, tz=NY)
        g = g[(g["ts"] >= lo) & (g["ts"] < hi)]
        series[key] = [[chart_time(r.ts), round(r.open, 4), round(r.high, 4), round(r.low, 4), round(r.close, 4)]
                       for r in g.itertuples(index=False)]

    out_trades = []
    for i, t in enumerate(trades.sort_values("signal_ts").itertuples(index=False)):
        tt = t._asdict()
        strat = tt.get("strategy")
        tcfg = sub_config(cfg, strat) if isinstance(strat, str) and strat in cfg.strategies else cfg
        (w0, w1), (s0, s1) = chart_spans(tcfg, trading_day(tt["signal_ts"]))
        days = _days(pd.Series(tt))
        out_trades.append({
            "id": i, "symbol": tt["symbol"], "partner": tt.get("partner"), "side": tt["side"],
            "day": str(days[0]), "keys": [_key(tt["symbol"], d) for d in days],
            "partner_keys": [_key(tt["partner"], d) for d in days] if isinstance(tt.get("partner"), str) else [],
            "signal": chart_time(tt["signal_ts"]), "signal_label": tt["signal_ts"].strftime("%Y-%m-%d %H:%M"),
            "fill": chart_time(tt["entry_ts"]) if pd.notna(tt.get("entry_ts")) else None,
            "exit": chart_time(tt["exit_ts"]) if pd.notna(tt.get("exit_ts")) else None,
            "exit_label": tt["exit_ts"].strftime("%Y-%m-%d %H:%M") if pd.notna(tt.get("exit_ts")) else "",
            "smt": chart_time(tt["smt_ts"]) if pd.notna(tt.get("smt_ts")) else None,
            "pivot": chart_time(tt["pivot_ts"]) if pd.notna(tt.get("pivot_ts")) else None,
            "pivot_price": _num(tt.get("pivot_price")),
            "entry": _num(tt.get("entry")), "stop": _num(tt.get("stop")), "target": _num(tt.get("target")),
            "exit_price": _num(tt.get("exit")), "exit_reason": tt.get("exit_reason"),
            "tdo": _num(tt.get("tdo")), "tdo_source": tt.get("tdo_source") if isinstance(tt.get("tdo_source"), str) else None,
            "session_open": _num(tt.get("session_open")),
            "qty": int(tt["qty"]), "risk": _num(tt.get("risk_per_share")), "r": _num(tt.get("r")),
            "pnl": _num(tt.get("pnl")),
            "window": [chart_time(w0), chart_time(w1)],
            "session": [chart_time(s0), chart_time(s1)],
            "range": ([chart_time(clock(days[0], cfg.triad.range_start)), chart_time(clock(days[0], cfg.triad.range_end))]
                      if cfg.strategy == "triad930" else None),
            "range_high": _num(tt.get("range_high")), "range_low": _num(tt.get("range_low")),
        })
    rep = [] if report is None or report.empty else [
        {"symbol": idx, **{k: _num(v) if not isinstance(v, (int,)) else v for k, v in row.items()}}
        for idx, row in report.iterrows()]
    return {"title": title, "strategy": cfg.strategy, "rr": cfg.rr, "session_start_hour": cfg.session_start_hour, "trades": out_trades,
            "bars": series, "report": rep}


def write_chart(path: Path, trades: pd.DataFrame, bars: pd.DataFrame, cfg: QuarterlyConfig,
                report: Optional[pd.DataFrame] = None, title: str = "") -> Path:
    data = build_chart_data(trades, bars, cfg, report, title)
    payload = json.dumps(data, separators=(",", ":"), default=str).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path
