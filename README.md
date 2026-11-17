# Quartely-trader

A regime-switching allocation trader. A Gaussian HMM classifies the market into
volatility regimes, each regime's volatility rank maps to a long-only exposure
in a risk asset (SPY by default), and the rest goes to a defensive asset (SHY)
or cash. A risk
manager applies vol targeting, leverage caps and drawdown circuit breakers.
Orders go through Alpaca (paper by default).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your Alpaca paper keys (.env is git-ignored)
```

Keys come only from `.env` or the environment, never from config files. Paper
trading is the default; `broker.paper_trading: false` asks you to type a
confirmation phrase at the terminal before connecting to the live API.

## Usage

```bash
python main.py --source yfinance backtest --symbols SPY --start 2019-01-01 --end 2024-12-31   # no keys needed
python main.py backtest --symbols SPY --start 2019-01-01 --end 2024-12-31 --compare   # + benchmarks
python main.py backtest --stress-test  # crash / gap / regime-misclassification stress tests
python main.py --train-only             # train + save the HMM, print the regime table
python main.py regimes                  # today's regime and signal, no trading
python main.py --dry-run                # full live loop, no orders sent
python main.py                          # live loop on the paper account (Ctrl-C to stop)
python main.py --dashboard              # in another terminal: attach to the running instance
python main.py --web-dashboard          # or in the browser: http://127.0.0.1:8050 (local only)
pytest tests/                           # offline tests (synthetic data, fake broker)
```

All parameters are in [config/settings.yaml](config/settings.yaml).

## How it works

| Stage | Module | Notes |
|---|---|---|
| Data | `data/market_data.py` | Alpaca, yfinance or CSV; split/dividend-adjusted daily bars, cached to `data/cache/` |
| Features | `data/feature_engineering.py` | 14 observables (returns, vol, volume, ADX/SMA trend, RSI, SMA200 distance, ROC, ATR), each a **rolling 252-bar z-score** |
| Regimes | `core/hmm_engine.py` | BIC picks 3–7 states; labels sorted by return (readability only); **forward-filtered** probabilities; 3-bar confirmation + flicker filter; models pickled to `models/` |
| Allocation | `core/regime_strategies.py` | ranks states by **volatility** (labels ignored): low → 95% @1.25x, mid → 95%/60% by 50 EMA trend, high → 60%; always long; uncertainty halves size at 1.0x; ATR/EMA stops (reported, not yet enforced) |
| Risk | `core/risk_manager.py` | independent of the HMM, veto over every signal/order: daily (2%/3%), weekly (5%/7%) and peak (10%, lock file) circuit breakers on actual P&L; stop-based sizing with gap-risk cap; leverage rules; correlation check; order validation (buying power, tradable, spread, duplicates) |
| Signals | `core/signal_generator.py` | rebalance schedule; trades only when target weight moves > 10% |
| Execution | `broker/` | limit orders ±0.1% (market retry after 30s), bracket/OTO entries, GTC protective stops that only tighten, trade_id journal (signal → risk → order → fill), websocket fills, startup reconciliation |
| Live loop | `core/trading_system.py` | startup checks, 5-min bar loop (breakers, trailing stops, feed/API health), signals on completed daily bars, snapshot recovery, graceful shutdown |
| Monitoring | `monitoring/` | JSON logs (`main`/`trades`/`alerts`/`regime`, 10 MB, 30 days), rate-limited alerts, dashboard |
| Backtest | `backtest/` | rolling walk-forward (252-bar train / 126-bar test), whole-share allocation math with margin, next-open fills, 0.05% slippage; regime/confidence attribution; buy-and-hold, 200 SMA and random benchmarks; crash, gap and misclassification stress tests |

### Look-ahead safeguards

- hmmlearn's `predict` (Viterbi) and `predict_proba` (forward-backward) use
  future data. The engine implements its own forward recursion instead;
  `tests/test_look_ahead.py` checks that appending data never changes past
  regimes, probabilities, stability flags or features.
- Each walk-forward window trains only on the 252 bars before its test period;
  `test_backtest_has_no_look_ahead` checks that truncating the data never
  changes earlier results.
- Signals use bar N's close and fill at bar N+1's open (`backtest.fill_delay`).

Backtest output (default `results/`): `equity_curve.csv`, `trade_log.csv`
(one row per rebalance, with P&L until the next one), `fills.csv`,
`regime_history.csv`, `walk_forward_windows.csv`, and with `--compare`
`benchmark_comparison.csv`; `--stress-test` adds `stress_*.csv`.

### Live trading notes

- The main loop wakes on every 5-minute bar. Signals use completed **daily** bars
  (`execution.signal_timeframe: 1Day`, as backtested) and run once a day at
  `execution.run_time` (ET). Setting it to `5Min` runs the HMM intraday, which has
  not been backtested.
- On SIGINT/SIGTERM the system closes its feeds and saves
  `logs/state_snapshot.json`, but leaves positions open under their stops. The
  next start recovers risk state from the snapshot.
- Live positions carry broker-side stop orders; backtests don't simulate stops,
  so live results can differ.
- If peak drawdown exceeds 10%, the risk manager closes everything and writes
  `trading_halted.lock`. Trading stays halted until you review and delete the
  file by hand. Backtests simulate that review with `backtest.lock_cooldown_bars`.
- Risk limits in `settings.yaml` are sized for one ETF at 60–119% of equity; the
  stricter multi-stock values from the spec are noted beside each one.
- Set `broker.paper: false` only after paper trading it for a while.

This is research software, not investment advice. Backtest results do not
guarantee future performance.

## Second strategy: Quarterly Theory on the index triad

A separate intraday strategy in `quarterly/` (config: `config/quarterly.yaml`).
It shares the Alpaca account with the HMM bot, so **run only one of them at a
time**. `strategy:` in the config picks the model; the other stays dormant.

**`triad930` (active): 9:30 manipulation** on QQQ / DIA / SPY (Nasdaq 100, Dow 30,
S&P 500), a port of the Pine Script "QT Triad 9:30 Manipulation":

- Per asset: True Day Open (00:00 NY bar, via Alpaca's overnight `boats` feed),
  the 8am range (08:00–09:00 high/low), day high/low from midnight.
- A *leader* takes exactly one side of its 8am range and trades beyond its TDO
  (8am high taken and above the TDO = bearish; 8am low taken and below the TDO =
  bullish). A leader that took both sides is no leader.
- In the 9:30–9:33 window, sell (bearish leader) or buy (bullish leader) every
  other member that still has not taken its 8am range (`strict_lag`).
- Stop at the day high/low (or window high/low) + 5 cents, target RR × risk,
  one trade per symbol per day, open trades closed at 15:55. Orders are tagged
  `qt930-…`.
- Options in `triad930:`: window length, `sweep_in_window`, `strict_lag`,
  `need_premium`, long/short switches, `stop_mode`, `stop_buffer`, `flat_at`.

**`q3` (dormant): "Q3 of Q3"**, entries only in the third quarter of Q3
(10:45:00–11:07:30 NY at session start 7), bias from TDO and NY Session True
Open, SMT divergence across the triad, market structure shift on confirmed
swing pivots. Enable with `strategy: q3` or `--strategy q3`.

Both use bracket orders (market entry + stop + target), whole shares, GTC legs,
1% risk per trade, and skip symbols that can't be shorted.

```bash
scripts/run_quarterly.sh                    # paper, kept awake (caffeinate), restarted after a crash
kill $(cat logs/quarterly_runner.pid)       # stop it cleanly
python -m quarterly live --dry-run          # full pipeline, orders only logged
python -m quarterly live                    # paper account
LIVE_TRADING=true python -m quarterly live  # real money (asks for a typed confirmation)
python -m quarterly backtest --start 2026-01-02 --end 2026-10-20 [--rr 4] [--feed sip] [--strategy q3]
python -m quarterly coverage --start 2026-10-01 --end 2026-10-20   # data coverage per symbol
python -m quarterly chart --trades results/quarterly/trades_<tag>.csv   # rebuild a trade chart page
```

Each backtest also writes `results/quarterly/chart_<tag>.html`: open it in a
browser to step through every trade on 1-minute candles (symbol + the leader or
SMT partner) with entry, stop, target, exit, True Day Open, the 8am range (triad)
or session open / SMT / pivot (q3) and the entry window marked. Needs internet for the chart library (lightweight-charts).

Safeguards: bad prints (a bar's high/low more than `max_wick_pct` beyond its
open/close) are clipped; a backup stop closes a position at market if a bar
closes beyond its bracket stop and the broker's stop leg has not filled.

Decisions and their reasons are logged to `logs/quarterly.log`; backtest trades
and reports go to `results/quarterly/`.
