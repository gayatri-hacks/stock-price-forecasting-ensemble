import numpy as np
import pandas as pd
import pytest

from financial_pipeline.core import (backtest_long_flat, build_lagged_features, chronological_split, export_csvs, fit_standardizer,
    forecast_models, forecasting_metrics, load_market_snapshot, normalise_yfinance_frame, validate_market_data, verify_manifest, write_manifest)
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


def test_validation_rejects_nonfinite_and_inconsistent_ohlc():
    frame = fixture_frame(); frame.iloc[3, frame.columns.get_loc("Close")] = np.inf
    with pytest.raises(ValueError, match="Missing or nonnumeric|Non-finite"):
        validate_market_data(frame)
    frame = fixture_frame(); frame.loc[frame.index[3], "High"] = frame.loc[frame.index[3], "Low"] - 1
    with pytest.raises(ValueError, match="OHLC"):
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
    assert predictions.iloc[0].previous_close == frame.loc[split.validation.index[-1], "Close"]


def test_features_and_earlier_forecasts_do_not_depend_on_future_data():
    frame = fixture_frame(240); split = chronological_split(frame)
    before, _ = forecast_models(split)
    changed = frame.copy(); changed.loc[changed.index[-1], "Close"] *= 100
    after, _ = forecast_models(chronological_split(changed))
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
    features, _ = build_lagged_features(frame.Close)
    assert features.iloc[0].lag_1 == frame.Close.iloc[19]


def test_intraday_backtest_has_hand_calculated_round_trip_costs_and_initial_equity():
    dates = pd.bdate_range("2024-01-01", periods=4)
    predictions = pd.DataFrame({"actual_open": [100, 100, 100, 100], "actual_close": [110, 90, 120, 105], "previous_close": [99, 100, 100, 100], "model": [101, 101, 99, 101]}, index=dates)
    ledger, metrics = backtest_long_flat(predictions, "model", cost_bps=10)
    # Long, long, flat, long: each long session is independently bought and sold.
    assert ledger["position"].tolist() == [1., 1., 0., 1.]
    assert ledger["turnover"].tolist() == [2., 2., 0., 2.]
    assert ledger["transaction_cost"].tolist() == pytest.approx([.002, .002, 0., .002])
    assert ledger["gross_return"].tolist() == pytest.approx([.10, -.10, 0., .05])
    assert ledger["net_return"].tolist() == pytest.approx([.098, -.102, 0., .048])
    assert ledger.iloc[0]["equity"] == pytest.approx(1.098)  # measured from initial capital of 1.0
    assert ledger.iloc[1]["drawdown"] == pytest.approx((1.098 * .898) / 1.098 - 1)
    assert metrics["trade_count"] == 3
    assert metrics["turnover"] == 6
    assert (ledger["net_return"] <= ledger["gross_return"]).all()
    assert metrics["max_drawdown_pct"] <= 0


def test_intraday_cost_sensitivity_is_two_sides_per_long_session():
    dates = pd.bdate_range("2024-01-01", periods=1)
    predictions = pd.DataFrame({"actual_open": [100], "actual_close": [101], "previous_close": [99], "model": [100]}, index=dates)
    returns = [backtest_long_flat(predictions, "model", cost)[0].iloc[0].net_return for cost in (0, 5, 10, 20)]
    assert returns == pytest.approx([.01, .009, .008, .006])


def test_offline_snapshot_rejects_unprovenanced_or_fixture_data(tmp_path):
    fixture_frame(120).to_csv(tmp_path / "AAPL_input_snapshot.csv", index_label="Date")
    (tmp_path / "AAPL_input_snapshot.metadata.json").write_text('{"ticker":"AAPL", "provider":"test fixture, not market data"}')
    with pytest.raises(ValueError, match="genuine Yahoo Finance"):
        load_market_snapshot(tmp_path, "AAPL")


def test_csv_exports_and_report_generation(tmp_path):
    quality = pd.DataFrame([{"ticker": "AAPL", "records": 100}])
    forecasts = pd.DataFrame([{"ticker": "AAPL", "model": "persistence", "rmse": 1., "mae": 1., "mape_pct": 1., "directional_accuracy": .5}])
    strategy = pd.DataFrame([{"ticker": "AAPL", "model": "persistence", "total_return_pct": 1., "annualized_sharpe": .2, "max_drawdown_pct": -1., "trade_count": 2}])
    export_csvs(tmp_path, quality, forecasts, strategy, pd.DataFrame([{"ticker": "AAPL", "model": "ridge_autoregression"}]))
    report = generate_report(tmp_path, quality, forecasts, strategy)
    assert (tmp_path / "forecast_metrics.csv").exists()
    assert (tmp_path / "forecast_rmse_by_ticker.png").exists()
    assert "not investment advice" in report.read_text()
    assert report.name == "FINANCIAL_ANALYTICS_REPORT.md"


def test_manifest_hashes_outputs(tmp_path):
    artifact = tmp_path / "a.csv"; artifact.write_text("x\n1\n")
    manifest = write_manifest(tmp_path, {"schema_version": 1})
    contents = __import__("json").loads(manifest.read_text())
    assert contents["output_sha256"]["a.csv"]
    assert verify_manifest(manifest)
    artifact.write_text("changed\n")
    assert not verify_manifest(manifest)
