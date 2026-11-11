"""Supported entry point for the reproducible financial analytics pipeline.

The original five research phases remain in their directories for historical
reference but are not used for verified reporting because their targets and
backtests are not reliable enough to support business claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from financial_pipeline.core import (backtest_long_flat, chronological_split, download_market_data,
    export_csvs, fit_standardizer, forecast_models, forecasting_metrics, validate_market_data)
from financial_pipeline.report import generate_report


def run_verified(start: str = "2020-01-01", end: str = "2025-12-31", output_dir: str = "outputs/verified") -> dict:
    """Run a bounded real-data analysis. ``end`` is exclusive, per yfinance."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    quality_rows, forecast_rows, strategy_rows, selection_rows = [], [], [], []
    for ticker in ("AAPL", "NVDA", "SPY"):
        data = download_market_data(ticker, start, end)
        split = chronological_split(data)
        quality = validate_market_data(data)
        quality.update({"ticker": ticker, "source": "Yahoo Finance via yfinance", "requested_start": start,
                        "requested_end_exclusive": end, "train_scaler_close": fit_standardizer(split.train)["Close"]["mean"]})
        predictions, selection = forecast_models(split)
        metrics = forecasting_metrics(predictions); metrics.insert(0, "ticker", ticker)
        selection.insert(0, "ticker", ticker)
        quality_rows.append(quality); forecast_rows.append(metrics); selection_rows.append(selection)
        for model in predictions.columns.drop("actual_close"):
            ledger, strategy = backtest_long_flat(predictions, model)
            strategy["ticker"] = ticker
            strategy_rows.append(strategy)
            ledger.to_csv(destination / f"{ticker}_{model}_backtest_ledger.csv", index_label="Date")
        predictions.to_csv(destination / f"{ticker}_held_out_predictions.csv", index_label="Date")
    quality, forecasts = pd.DataFrame(quality_rows), pd.concat(forecast_rows, ignore_index=True)
    strategies, selections = pd.DataFrame(strategy_rows), pd.concat(selection_rows, ignore_index=True)
    export_csvs(destination, quality, forecasts, strategies, selections)
    report = generate_report(destination, quality, forecasts, strategies)
    return {"quality": quality, "forecasts": forecasts, "strategies": strategies, "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-safe real-data financial analytics report")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31", help="exclusive yfinance end date")
    parser.add_argument("--output-dir", default="outputs/verified")
    args = parser.parse_args()
    result = run_verified(args.start, args.end, args.output_dir)
    print(f"Verified report written to {result['report']}")
