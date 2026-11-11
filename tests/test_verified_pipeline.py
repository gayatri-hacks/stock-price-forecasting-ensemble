import numpy as np
import pandas as pd
import pytest

from financial_pipeline.core import (backtest_long_flat, chronological_split, export_csvs, fit_standardizer,
    forecast_models, forecasting_metrics, normalise_yfinance_frame, validate_market_data)
from financial_pipeline.report import generate_report


def fixture_frame(rows=180):
    index = pd.bdate_range("2020-01-01", periods=rows)
    close = pd.Series(np.linspace(100, 150, rows) + np.sin(np.arange(rows)), index=index)
    return pd.DataFrame({"Open": close - .5, "High": close + 1, "Low": close - 1, "Close": close,
                         "Volume": 1_000_000}, index=index)


def test_validation_rejects_duplicate_and_missing_data():
    frame = fixture_frame()
    assert validate_market_data(frame)["records"] == 180
    with pytest.raises(ValueError, match="Duplicate"):
        validate_market_data(pd.concat([frame, frame.iloc[[-1]]]))
    frame.loc[frame.index[4], "Close"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        validate_market_data(frame)


def test_current_yfinance_multiindex_format_is_normalised():
    frame = fixture_frame(120)
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["AAPL"]])
    normalized = normalise_yfinance_frame(frame)
    assert list(normalized.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert normalized.index.tz is None


def test_chronological_split_and_scaler_do_not_use_future():
    frame = fixture_frame()
    split = chronological_split(frame)
    assert split.train.index.max() < split.validation.index.min() < split.test.index.min()
    mean_before = fit_standardizer(split.train)["Close"]["mean"]
    changed = frame.copy(); changed.loc[split.test.index, "Close"] *= 100
    assert fit_standardizer(chronological_split(changed).train)["Close"]["mean"] == mean_before


def test_forecasts_are_held_out_and_metrics_are_finite():
    frame = fixture_frame(240); split = chronological_split(frame)
    predictions, selection = forecast_models(split)
    assert predictions.index.min() >= split.test.index.min()
    assert selection.loc[0, "selected_alpha"] in {0.01, .1, 1., 10.}
    assert np.isfinite(forecasting_metrics(predictions)[["mae", "rmse"]].to_numpy()).all()


def test_backtest_charges_turnover_and_uses_no_future_actual_for_signal():
    dates = pd.bdate_range("2024-01-01", periods=4)
    predictions = pd.DataFrame({"actual_close": [100, 110, 100, 110], "model": [100, 101, 109, 99]}, index=dates)
    ledger, metrics = backtest_long_flat(predictions, "model", cost_bps=10)
    assert metrics["trade_count"] == int(ledger["turnover"].sum())
    assert (ledger["net_return"] <= ledger["gross_return"]).all()
    assert metrics["max_drawdown_pct"] <= 0


def test_csv_exports_and_report_generation(tmp_path):
    quality = pd.DataFrame([{"ticker": "AAPL", "records": 100}])
    forecasts = pd.DataFrame([{"ticker": "AAPL", "model": "persistence", "rmse": 1., "mae": 1., "mape_pct": 1., "directional_accuracy": .5}])
    strategy = pd.DataFrame([{"ticker": "AAPL", "model": "persistence", "total_return_pct": 1., "annualized_sharpe": .2, "max_drawdown_pct": -1., "trade_count": 2}])
    export_csvs(tmp_path, quality, forecasts, strategy, pd.DataFrame([{"ticker": "AAPL", "model": "ridge_autoregression"}]))
    report = generate_report(tmp_path, quality, forecasts, strategy)
    assert (tmp_path / "forecast_metrics.csv").exists()
    assert (tmp_path / "forecast_rmse_by_ticker.png").exists()
    assert "not investment advice" in report.read_text()
