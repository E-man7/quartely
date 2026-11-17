"""Quartely-trader entry point.

    python main.py                       # live trading loop (paper by default)
    python main.py --dry-run             # full live pipeline, no orders sent
    python main.py --backtest [--symbols SPY] [--start 2019-01-01] [--end 2024-12-31]
    python main.py --compare             # backtest + buy-and-hold / 200 SMA / random benchmarks
    python main.py --stress-test         # backtest + crash / gap / misclassification stress tests
    python main.py --train-only          # train the HMM on the latest data, save it, exit
    python main.py --dashboard           # attach a dashboard to a running instance
    python main.py --web-dashboard       # same, in the browser at http://127.0.0.1:8050
    python main.py regimes               # today's regime and signal (no trading)

`python main.py backtest --symbols SPY ...` (subcommand form) also works.
"""
from __future__ import annotations

import argparse
import json
import signal as signal_mod
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

from config import PROJECT_ROOT, load_credentials, load_settings
from monitoring.logger import get_logger, setup_logging

log = get_logger("main")


def _load_bars(settings: Dict, creds: Dict) -> Dict[str, pd.DataFrame]:
    """Bars for the signal, risk and safe symbols, with enough history for features + first training window."""
    from data.feature_engineering import feature_warmup
    from data.market_data import MarketData

    uni, bt = settings["universe"], settings["backtest"]
    symbols = sorted({s for s in (uni["signal_symbol"], uni["risk_symbol"], uni.get("safe_symbol")) if s})
    md = MarketData(settings, creds)
    start = bt.get("start")
    if start:
        bars_needed = bt.get("walk_forward", {}).get("in_sample", 252) + feature_warmup(settings.get("features"))
        # ~1.45 calendar days per trading day
        start = (pd.Timestamp(start) - pd.Timedelta(days=int(bars_needed * 1.45) + 10)).strftime("%Y-%m-%d")
    return {s: md.history(s, start=start, end=bt.get("end")) for s in symbols}


# ---------------------------------------------------------------- commands
def cmd_backtest(settings: Dict, creds: Dict, args) -> None:
    import copy

    from rich.console import Console

    from backtest import StressTester, WalkForwardBacktester, render_report, render_stress, save_report, stress_summary

    if args.start:
        settings["backtest"]["start"] = args.start
    if args.end:
        settings["backtest"]["end"] = args.end
    console = Console()
    symbols = args.symbols or [settings["universe"]["risk_symbol"]]
    for symbol in symbols:
        s = copy.deepcopy(settings)
        # Each symbol is modelled and traded on its own
        s["universe"]["signal_symbol"] = s["universe"]["risk_symbol"] = symbol
        bars = _load_bars(s, creds)
        bt = WalkForwardBacktester(s)
        result = bt.run(bars)
        if args.compare:
            bt.run_benchmarks(bars, result, n_random=s["backtest"].get("random_seeds", 100))
        report = result.summary()
        render_report(report, console)
        out = Path(args.out) / symbol if len(symbols) > 1 else Path(args.out)
        if args.stress_test:
            tester = StressTester(s)
            stress = tester.run(result, bars)
            summary = stress_summary(stress, result, tester.containment_limit)
            render_stress(summary, console)
            out.mkdir(parents=True, exist_ok=True)
            for name, df in stress.items():
                df.to_csv(out / f"stress_{name}.csv", index=False)
        save_report(result, report, out)
        console.print(f"Saved results to {out.resolve()}")


def cmd_regimes(settings: Dict, creds: Dict, args) -> None:
    from core.signal_generator import SignalGenerator
    from data.market_data import MarketData

    md = MarketData(settings, creds)
    sig = SignalGenerator(settings)
    bars = md.history(settings["universe"]["signal_symbol"])
    risk_bars = md.history(sig.risk_symbol) if sig.risk_symbol != settings["universe"]["signal_symbol"] else None
    signal = sig.generate(bars, risk_bars=risk_bars)
    print("BIC by state count:", {k: round(v, 1) for k, v in sig.hmm.bic_scores.items()})
    print(sig.hmm.describe().to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nTransition matrix:")
    print(sig.hmm.get_transition_matrix().to_string(float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(signal.as_dict(), indent=2))


def cmd_train(settings: Dict, creds: Dict, args) -> None:
    """Train the HMM on the latest signal-timeframe history, save it with metadata, and exit."""
    from core.trading_system import TradingSystem
    from data.market_data import MarketData

    system = TradingSystem(settings, creds, market_data=MarketData(settings, creds))
    bars, _ = system.signal_history()
    sig = system.sig
    sig.ensure_fitted(sig.features(bars).dropna(), max_age_days=-1)    # always retrain
    print("BIC by state count:", {k: round(v, 1) for k, v in sig.hmm.bic_scores.items()})
    print(sig.hmm.describe().to_string(float_format=lambda x: f"{x:.4f}"))
    print(json.dumps(sig.hmm.metadata(), indent=2, default=str))


def cmd_dashboard(settings: Dict, creds: Dict, args) -> None:
    from monitoring.dashboard import run_viewer

    mon = settings.get("monitoring", {})
    run_viewer(PROJECT_ROOT / mon.get("status_file", "logs/status.json"), mon.get("dashboard_refresh_seconds", 5))


def cmd_web_dashboard(settings: Dict, creds: Dict, args) -> None:
    from monitoring.web_dashboard import serve

    mon = settings.get("monitoring", {})
    web = mon.get("web_dashboard", {})
    serve(PROJECT_ROOT / mon.get("status_file", "logs/status.json"), host=args.host or web.get("host", "127.0.0.1"),
          port=args.port or web.get("port", 8050), refresh_seconds=mon.get("dashboard_refresh_seconds", 5))


def cmd_live(settings: Dict, creds: Dict, args) -> None:
    from core.trading_system import TradingSystem

    system = TradingSystem(settings, creds, dry_run=args.dry_run)

    def request_stop(signum, frame) -> None:
        log.warning("shutdown requested", extra={"signal": signal_mod.Signals(signum).name})
        system.stop_event.set()

    signal_mod.signal(signal_mod.SIGINT, request_stop)
    signal_mod.signal(signal_mod.SIGTERM, request_stop)
    started = False
    try:
        started = system.startup()
        if not started:
            return
        if args.once:
            system.decide(system.now())
        else:
            system.run()
    except Exception as exc:
        if system.alerts is not None:
            system.on_unhandled(exc)
        raise
    finally:
        if started or system.tracker is not None:
            system.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HMM regime-based allocation trader")
    parser.add_argument("command", nargs="?", choices=["live", "backtest", "regimes"], default=None,
                        help="optional subcommand (default: live trading)")
    parser.add_argument("--config", help="path to settings.yaml")
    parser.add_argument("--source", choices=["alpaca", "yfinance", "csv"], help="override data.source")
    mode = parser.add_argument_group("modes")
    mode.add_argument("--dry-run", action="store_true", help="full live pipeline, no orders")
    mode.add_argument("--backtest", action="store_true", help="walk-forward backtest")
    mode.add_argument("--train-only", action="store_true", help="train the HMM and exit")
    mode.add_argument("--stress-test", action="store_true", help="run stress tests (implies --backtest)")
    mode.add_argument("--compare", action="store_true", help="benchmark comparisons (implies --backtest)")
    mode.add_argument("--dashboard", action="store_true", help="show the dashboard of a running instance")
    mode.add_argument("--web-dashboard", action="store_true", help="serve the dashboard on localhost")
    web = parser.add_argument_group("web dashboard options")
    web.add_argument("--port", type=int, help="web dashboard port (default 8050)")
    web.add_argument("--host", help="web dashboard bind address (default 127.0.0.1, local only)")
    bt = parser.add_argument_group("backtest options")
    bt.add_argument("--symbols", nargs="+", help="symbols to backtest independently (default: universe.risk_symbol)")
    bt.add_argument("--start", help="first out-of-sample date (YYYY-MM-DD)")
    bt.add_argument("--end", help="last date (YYYY-MM-DD)")
    bt.add_argument("--out", default=str(PROJECT_ROOT / "results"), help="directory for CSV results")
    lv = parser.add_argument_group("live options")
    lv.add_argument("--once", action="store_true", help="start up, make one decision, shut down")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    if args.source:
        settings["data"]["source"] = args.source
    setup_logging(settings.get("monitoring"), PROJECT_ROOT)
    creds = load_credentials()

    if args.dashboard:
        cmd_dashboard(settings, creds, args)
    elif args.web_dashboard:
        cmd_web_dashboard(settings, creds, args)
    elif args.train_only:
        cmd_train(settings, creds, args)
    elif args.command == "backtest" or args.backtest or args.stress_test or args.compare:
        cmd_backtest(settings, creds, args)
    elif args.command == "regimes":
        cmd_regimes(settings, creds, args)
    else:
        cmd_live(settings, creds, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
