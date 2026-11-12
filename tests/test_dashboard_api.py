"""Isolated API tests use labelled fixtures, never genuine market data."""
import os
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import app, read_csv
from financial_pipeline.core import write_manifest


def fixture_run(path: Path) -> None:
    quality = pd.DataFrame([{"ticker": ticker, "records": 120, "start_date": "2024-01-01", "end_date": "2024-06-01"} for ticker in ("AAPL", "NVDA", "SPY")])
    metrics = pd.DataFrame([{"ticker": ticker, "model": model, "mae": 1., "rmse": 2., "nrmse_pct": 1., "mape_pct": 1., "directional_accuracy": .5, "skill_vs_persistence_mae": 0., "observations": 20} for ticker in ("AAPL", "NVDA", "SPY") for model in ("persistence", "moving_average_20", "ridge_autoregression")])
    strategies = pd.DataFrame([{"ticker": ticker, "model": model, "cost_bps_per_side": cost, "total_return_pct": 1., "annualized_sharpe": .1, "max_drawdown_pct": -2., "trade_count": 4, "turnover": 8} for ticker in ("AAPL", "NVDA", "SPY") for model in ("persistence", "moving_average_20", "ridge_autoregression") for cost in (0, 5, 10, 20)])
    quality.to_csv(path / "data_quality_summary.csv", index=False); metrics.to_csv(path / "forecast_metrics.csv", index=False); strategies.to_csv(path / "strategy_metrics.csv", index=False)
    pd.DataFrame([{"model": "ridge_autoregression", "selected_alpha": 1.}]).to_csv(path / "model_selection.csv", index=False)
    for ticker in ("AAPL", "NVDA", "SPY"):
        pd.DataFrame({"Date": ["2024-05-01"], "actual_close": [100.], "previous_close": [99.], "persistence": [99.], "moving_average_20": [98.], "ridge_autoregression": [101.]}).to_csv(path / f"{ticker}_held_out_predictions.csv", index=False)
        for model in ("persistence", "moving_average_20", "ridge_autoregression"):
            for cost in (0, 5, 10, 20): pd.DataFrame({"Date": ["2024-05-01"], "equity": [1.01], "gross_return": [.01], "net_return": [.009], "drawdown": [-.01]}).to_csv(path / f"{ticker}_{model}_{cost}bps_backtest_ledger.csv", index=False)
    (path / "FINANCIAL_ANALYTICS_REPORT.md").write_text("fixture report, not market data")
    write_manifest(path, {"input_mode": "test fixture, not market data", "input_snapshot_provenance": []})


def client(tmp_path, monkeypatch):
    fixture_run(tmp_path); monkeypatch.setenv("VERIFIED_OUTPUT_DIR", str(tmp_path)); read_csv.cache_clear(); return TestClient(app)


def test_health_and_unavailable_state(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIFIED_OUTPUT_DIR", str(tmp_path)); read_csv.cache_clear(); c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/api/datasets").status_code == 503


def test_verified_artifact_endpoints_and_validation(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.get("/ready").json()["ready"] is True
    assert c.get("/api/overview", params={"ticker": "AAPL", "model": "persistence"}).status_code == 200
    assert c.get("/api/predictions", params={"ticker": "AAPL"}).json()["predictions"][0]["forecast"] == 99.
    assert c.get("/api/backtest", params={"ticker": "AAPL", "transaction_cost_bps": 5}).status_code == 200
    assert len(c.get("/api/cost-sensitivity", params={"ticker": "AAPL"}).json()["results"]) == 4
    assert c.get("/api/predictions", params={"ticker": "BAD"}).status_code == 422
    assert c.get("/api/predictions", params={"ticker": "AAPL", "start_date": "2025-01-01", "end_date": "2024-01-01"}).status_code == 422
    assert c.get("/api/reports/../../README.md").status_code == 404


def test_tampered_manifest_artifacts_are_not_served(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    (tmp_path / "forecast_metrics.csv").write_text("tampered\n")
    read_csv.cache_clear()
    assert c.get("/api/forecast-metrics").status_code == 503
