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
