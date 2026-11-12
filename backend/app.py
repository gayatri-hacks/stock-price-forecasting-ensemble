"""FastAPI facade over completed, verified pipeline runs; it never retrains models."""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from financial_pipeline.core import verify_manifest

app = FastAPI(title="Financial Forecast Intelligence API", version="1.0.0")
TICKERS = {"AAPL", "NVDA", "SPY"}
MODELS = {"persistence", "moving_average_20", "ridge_autoregression"}
COSTS = {0, 5, 10, 20}
APPROVED_FILES = {"FINANCIAL_ANALYTICS_REPORT.md", "forecast_metrics.csv", "strategy_metrics.csv", "model_selection.csv", "manifest.json"}


def output_dir() -> Path:
    return Path(os.environ.get("VERIFIED_OUTPUT_DIR", "outputs/verified"))


def unavailable(message: str = "No verified genuine-data run is available. Load a provenance-marked snapshot or run the pipeline when Yahoo Finance is available.") -> HTTPException:
    return HTTPException(status_code=503, detail=message)


def ensure_scalar(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return ensure_scalar(value.item())
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return [{key: ensure_scalar(value) for key, value in row.items()} for row in frame.replace({np.nan: None}).to_dict("records")]


@lru_cache(maxsize=8)
def read_csv(name: str, identity: str) -> pd.DataFrame:
    path = output_dir() / name
    if not path.is_file():
        raise unavailable()
    return pd.read_csv(path)


def identity() -> str:
    manifest = output_dir() / "manifest.json"
    if not manifest.is_file():
        raise unavailable("No verified manifest is available; refusing to serve unprovenanced results.")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def validated(ticker: str | None = None, model: str | None = None, cost: int | None = None) -> None:
    if ticker is not None and ticker not in TICKERS: raise HTTPException(422, "Unsupported ticker")
    if model is not None and model not in MODELS: raise HTTPException(422, "Unsupported model")
    if cost is not None and cost not in COSTS: raise HTTPException(422, "Unsupported transaction cost")


def filter_dates(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if start_date and end_date and start_date > end_date: raise HTTPException(422, "start_date must not be after end_date")
    if "Date" not in frame: return frame
    if start_date: frame = frame[frame["Date"] >= start_date]
    if end_date: frame = frame[frame["Date"] <= end_date]
    return frame.head(2_000)


@app.get("/health")
def health() -> dict: return {"status": "ok"}

@app.get("/ready")
def ready() -> dict:
    try:
        path = output_dir() / "manifest.json"
        return {"ready": verify_manifest(path), "dataset_available": True, "run_identity": identity()}
    except HTTPException:
        return {"ready": False, "dataset_available": False}

@app.get("/api/datasets")
def datasets() -> dict:
    quality = read_csv("data_quality_summary.csv", identity())
    return {"datasets": records(quality)}

@app.get("/api/overview")
def overview(ticker: Literal["AAPL", "NVDA", "SPY"], model: str = "persistence") -> dict:
    validated(ticker, model)
    metrics = read_csv("forecast_metrics.csv", identity())
    row = metrics[(metrics.ticker == ticker) & (metrics.model == model)]
    if row.empty: raise unavailable("Verified metrics do not contain this selection")
    quality = read_csv("data_quality_summary.csv", identity())
    return {"metric": records(row)[0], "quality": records(quality[quality.ticker == ticker])[0], "historical_held_out": True}

@app.get("/api/forecast-metrics")
def forecast_metrics(ticker: str | None = None) -> dict:
    validated(ticker); frame = read_csv("forecast_metrics.csv", identity())
    return {"metrics": records(frame if ticker is None else frame[frame.ticker == ticker])}

@app.get("/api/model-comparison")
def model_comparison(ticker: Literal["AAPL", "NVDA", "SPY"]) -> dict:
    return forecast_metrics(ticker)

@app.get("/api/predictions")
def predictions(ticker: Literal["AAPL", "NVDA", "SPY"], model: str = "persistence", start_date: str | None = None, end_date: str | None = None) -> dict:
    validated(ticker, model); frame = read_csv(f"{ticker}_held_out_predictions.csv", identity())
    if "Date" not in frame: raise unavailable("Invalid prediction ledger")
    frame = filter_dates(frame, start_date, end_date)
    return {"ticker": ticker, "model": model, "forecast_horizon": "one trading session", "predictions": records(frame[["Date", "actual_close", "previous_close", model]].rename(columns={model: "forecast"}))}

def ledger(ticker: str, model: str, transaction_cost_bps: int) -> pd.DataFrame:
    validated(ticker, model, transaction_cost_bps)
    return read_csv(f"{ticker}_{model}_{transaction_cost_bps}bps_backtest_ledger.csv", identity())

@app.get("/api/backtest")
def backtest(ticker: Literal["AAPL", "NVDA", "SPY"], model: str = "persistence", transaction_cost_bps: int = 5) -> dict:
    strategies = read_csv("strategy_metrics.csv", identity())
    row = strategies[(strategies.ticker == ticker) & (strategies.model == model) & (strategies.cost_bps_per_side == transaction_cost_bps)]
    if row.empty: raise unavailable("Verified backtest is unavailable for this selection")
    return {"metrics": records(row)[0], "assumption": "prior-close signal; next-open entry; same-session close exit; flat overnight"}

@app.get("/api/equity-curve")
def equity_curve(ticker: Literal["AAPL", "NVDA", "SPY"], model: str = "persistence", transaction_cost_bps: int = 5) -> dict:
    frame = ledger(ticker, model, transaction_cost_bps)
    return {"points": records(frame[["Date", "equity", "gross_return", "net_return"]].head(2_000))}

@app.get("/api/drawdown")
def drawdown(ticker: Literal["AAPL", "NVDA", "SPY"], model: str = "persistence", transaction_cost_bps: int = 5) -> dict:
    frame = ledger(ticker, model, transaction_cost_bps)
    return {"points": records(frame[["Date", "drawdown"]].head(2_000))}

@app.get("/api/cost-sensitivity")
def cost_sensitivity(ticker: Literal["AAPL", "NVDA", "SPY"], model: str = "persistence") -> dict:
    validated(ticker, model); frame = read_csv("strategy_metrics.csv", identity())
    return {"results": records(frame[(frame.ticker == ticker) & (frame.model == model)].sort_values("cost_bps_per_side"))}

@app.get("/api/data-quality")
def data_quality(ticker: str | None = None) -> dict: return datasets() if ticker is None else {"datasets": [x for x in datasets()["datasets"] if x["ticker"] == ticker]}

@app.get("/api/provenance")
def provenance() -> dict:
    manifest = json.loads((output_dir() / "manifest.json").read_text())
    return {"input_mode": manifest.get("input_mode"), "snapshots": manifest.get("input_snapshot_provenance", []), "warning": "A declared provider and matching hash establish integrity, not independent origin verification."}

@app.get("/api/manifest")
def manifest() -> dict:
    path = output_dir() / "manifest.json"
    if not path.is_file(): raise unavailable()
    return {"manifest": json.loads(path.read_text()), "verified": verify_manifest(path)}

@app.get("/api/reports/{filename}")
def report(filename: str):
    if filename not in APPROVED_FILES: raise HTTPException(404, "File is not approved for download")
    path = output_dir() / filename
    if not path.is_file(): raise unavailable("Approved output is not available")
    return FileResponse(path, filename=filename)
