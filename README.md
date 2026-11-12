# Reproducible Financial Analytics and Reporting

This repository preserves the original five-phase forecasting research while adding a supported, leakage-safe reporting path designed for reviewable financial analytics work. It is not a live-trading system, investment advice, or evidence of guaranteed forecasting performance.

## Verified workflow

The supported workflow uses actual daily OHLCV records for AAPL, NVDA and SPY from Yahoo Finance through `yfinance`. The default request window is 2020-01-01 through 2025-12-30 (the yfinance `end` argument is exclusive).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q tests
python main.py --start 2020-01-01 --end 2025-12-31
```

This writes a self-contained run under `outputs/verified/`:

- `FINANCIAL_ANALYTICS_REPORT.md` — business objective, data controls, held-out forecasting results, strategy results, risks and conclusion.
- `data_quality_summary.csv` — source, ticker, requested range, actual coverage, records, null and duplicate checks.
- `forecast_metrics.csv` and `strategy_metrics.csv` — deliberately separate model accuracy from illustrative strategy metrics.
- `*_input_snapshot.csv`, `*_held_out_predictions.csv` and `*_backtest_ledger.csv` — date-level audit trails.
- `manifest.json` — input and output SHA-256 hashes, source semantics, split/model configuration and cost settings.
- `forecast_rmse_by_ticker.png` — report chart.

The pipeline fails rather than fabricating data when a download is unavailable or invalid. Raw quotes are intentionally not versioned: Yahoo Finance may revise its history, so the quality summary documents the exact run coverage.

## Method and safeguards

- Market dates remain the index end-to-end. Current single-ticker yfinance `MultiIndex` responses are normalised explicitly.
- Required OHLCV fields, timestamp ordering, duplicated dates, missing values, minimum coverage and invalid prices are checked before modelling.
- Splits are contiguous: 60% train, 20% validation, 20% held-out test. Any scaler parameters are fitted on training data alone.
- A persistence forecast is the primary historical baseline. A 20-day moving average and a small ridge autoregression are compared on exactly the same test dates. Ridge regularisation is selected on validation only, then refit on train + validation before one final held-out test evaluation.
- Forecasts use prices available through the preceding close to predict the next close. No test target is used in a feature or model selection decision.
- The illustrative long/flat strategy makes a position decision from the prior-close forecast, fills at the next available opening price, earns open-to-close returns, and reports 0/5/10/20 basis-point turnover-cost sensitivity. Sharpe uses all daily strategy returns; drawdown and trade count are derived from the exported ledger.

## Audit outcome of legacy research files

The original `phase1_*` through `phase5_*`, `ensemble.py`, `walk_forward.py`, outputs, and deep-learning definitions are retained as research artifacts. They are not inputs to the verified report. Audit findings include discarded market dates, synthetic Prophet dates, mislabeled/misaligned horizons, target construction incompatibilities in the deep-learning phase, zero-valued fallback metrics, and randomly generated Phase 4 results. Consequently, historical `outputs/` performance figures must not be used as validated results.

The verified pipeline intentionally does not retrain legacy LSTM/Transformer/Prophet models. Their architecture and target construction need a separate, documented redevelopment before comparison with the baselines would be defensible.

## Tests and CI

`tests/test_verified_pipeline.py` uses deterministic fixtures, never market downloads. It covers input validation, current yfinance format normalisation, chronological splits, train-only preprocessing, held-out forecasting, backtest costs/metrics, CSV export and report rendering. GitHub Actions installs the lightweight supported dependencies and runs the test suite plus an explicit fixture integration smoke test.

## Limitations

Public source data may change; results are sample-specific. The strategy excludes taxes, bid/ask spread beyond its simple turnover charge, market impact, intraday execution, financing/borrow cost, survivorship checks and corporate-action reconciliation. No result should be interpreted as a production deployment or as evidence that a model will forecast future prices successfully.
