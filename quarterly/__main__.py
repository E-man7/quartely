"""Quarterly Theory strategies on the index triad (QQQ / DIA / SPY).

strategy in config/quarterly.yaml (or --strategy): triad930 = 9:30 manipulation
(default); q3 = the Q3-of-Q3 model (dormant).

    python -m quarterly live                    # paper account (LIVE_TRADING=true for live)
    python -m quarterly live --dry-run          # full pipeline, orders only logged
    python -m quarterly backtest --start 2026-01-02 --end 2026-10-20 [--rr 4] [--session-start 6] [--feed sip]
    python -m quarterly coverage --start 2026-10-01 --end 2026-10-20   # bars per symbol in the entry window
    python -m quarterly chart --trades results/quarterly/trades_<tag>.csv   # rebuild the trade chart page

Separate from the HMM bot (main.py). Both use the same Alpaca account: run one at a time.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, load_credentials, load_settings
from monitoring.logger import get_logger, setup_logging
from quarterly.config import STRATEGIES, load_config

log = get_logger("quarterly")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m quarterly", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    lv = sub.add_parser("live", help="run the live loop (paper unless LIVE_TRADING=true)")
    lv.add_argument("--dry-run", action="store_true", help="log orders instead of sending them")
    for name in ("backtest", "coverage"):
        b = sub.add_parser(name)
        b.add_argument("--start", required=True, type=date.fromisoformat)
        b.add_argument("--end", required=True, type=date.fromisoformat)
        b.add_argument("--feed", choices=["iex", "sip"], help="day feed for history (default: backtest_day_feed)")
        b.add_argument("--session-start", type=int, choices=[6, 7])
        b.add_argument("--rr", type=float)
        b.add_argument("--equity", type=float, default=100_000.0)
        b.add_argument("--no-cache", action="store_true")
    ch = sub.add_parser("chart", help="interactive HTML chart of a backtest's trades")
    ch.add_argument("--trades", required=True, type=Path, help="trades_<tag>.csv written by the backtest")
    ch.add_argument("--feed", choices=["iex", "sip"], help="day feed the backtest used (default: backtest_day_feed)")
    p.add_argument("--config", help="path to quarterly.yaml")
    p.add_argument("--strategy", choices=list(STRATEGIES), help="override the strategy in quarterly.yaml")
    args = p.parse_args(argv)

    settings = load_settings()
    setup_logging(settings.get("monitoring"), PROJECT_ROOT)
    creds = load_credentials()
    cfg = load_config(Path(args.config) if args.config else None)
    if getattr(args, "dry_run", False):
        cfg.dry_run = True
    overrides = {k: v for k, v in (("session_start_hour", getattr(args, "session_start", None)),
                                   ("rr", getattr(args, "rr", None)),
                                   ("strategy", args.strategy),
                                   ("strategies", (args.strategy,) if args.strategy else None)) if v is not None}
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    if args.command == "live":
        from quarterly.live import QuarterlyBot

        QuarterlyBot(cfg, creds).run()
        return 0

    from quarterly.data import MinuteBars, coverage
    from quarterly.triad import strategy_label

    name = strategy_label(cfg)

    if args.command == "chart":
        from quarterly.chart import write_chart
        from quarterly.timemodel import trading_day

        trades = pd.read_csv(args.trades)
        if trades.empty:
            print("No trades in that file.")
            return 1
        days = pd.to_datetime(trades["signal_ts"], utc=True).map(trading_day)
        exits = pd.to_datetime(trades["exit_ts"], utc=True).map(trading_day)
        bars = MinuteBars(cfg, credentials=creds).history(cfg.symbols, min(days), max(exits), args.feed)
        report_file = args.trades.with_name(args.trades.name.replace("trades_", "report_"))
        rep = pd.read_csv(report_file, index_col=0) if report_file.exists() else None
        path = write_chart(args.trades.with_name(args.trades.name.replace("trades_", "chart_").replace(".csv", ".html")),
                           trades, bars, cfg, rep, title=f"Quarterly {name} trades: {args.trades.stem[7:]}")
        print(f"Chart: {path}")
        return 0

    bars = MinuteBars(cfg, credentials=creds).history(cfg.symbols, args.start, args.end, args.feed,
                                                      use_cache=not args.no_cache)
    if bars.empty:
        print("No bars for that range.")
        return 1
    if args.command == "coverage":
        cov = coverage(bars, cfg)
        print(cov.groupby("symbol").agg(days=("day", "count"), avg_window_bars=("window_bars", "mean"),
                                        avg_8am_range_bars=("range_bars", "mean"),
                                        days_with_midnight_bar=("midnight_bar", "sum")).to_string())
        return 0

    from quarterly.backtest import Backtester, report

    bt = Backtester(cfg, args.equity)
    trades = bt.run(bars)
    out = cfg.results_dir
    out.mkdir(parents=True, exist_ok=True)
    tag = (f"{cfg.strategy}_{args.start}_{args.end}_rr{cfg.rr:g}" if cfg.strategy != "q3"
           else f"{args.start}_{args.end}_s{cfg.session_start_hour}_rr{cfg.rr:g}")
    pd.DataFrame([dataclasses.asdict(t) for t in trades]).to_csv(out / f"trades_{tag}.csv", index=False)
    rep = report(trades, args.equity)
    if cfg.strategy == "q3":
        when = f"session {cfg.session_start_hour}:00"
    elif cfg.strategy == "silverbullet":
        when = "sessions " + ", ".join("{:02d}:{:02d}".format(*t) for t in cfg.silverbullet.sessions) + \
            " (" + "+".join(cfg.silverbullet.models) + ")"
    elif cfg.strategy == "nyopen":
        when = "entries " + ", ".join("{:02d}:{:02d}".format(*t) for t in cfg.nyopen.entry_times)
    else:
        when = "window {:02d}:{:02d}-{:02d}:{:02d}".format(*cfg.triad.window_start, *cfg.triad.window_end)
    print(f"Quarterly {name} backtest {args.start} .. {args.end}  {when}  "
          f"RR {cfg.rr:g}  risk {cfg.risk_pct}%  symbols {', '.join(cfg.symbols)}")
    print(rep.to_string(float_format=lambda x: f"{x:,.3f}") if not rep.empty else "No trades.")
    print(f"Final equity ${bt.equity:,.2f}  |  trades saved to {out / f'trades_{tag}.csv'}")
    rep.to_csv(out / f"report_{tag}.csv")
    if trades:
        from quarterly.chart import write_chart

        chart = write_chart(out / f"chart_{tag}.html", pd.DataFrame([dataclasses.asdict(t) for t in trades]), bars,
                            cfg, rep, title=f"Quarterly {name} trades {args.start} .. {args.end}  ({when}, RR {cfg.rr:g})")
        print(f"Chart of every trade: {chart}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
