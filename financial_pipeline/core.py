"""Leakage-safe market-data, forecasting, and backtesting primitives.

This module is deliberately independent of the legacy five-phase research code.
It is the supported, reproducible path used by ``main.py --verified``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def normalise_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, single-ticker OHLCV frame from current yfinance formats."""
    if frame.empty:
        raise ValueError("Market-data response is empty")
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        # yf.download may return (Price, Ticker) even for one ticker.
        result.columns = result.columns.get_level_values(0)
    result.index = pd.to_datetime(result.index, utc=True).tz_convert(None).normalize()
    result.index.name = "Date"
    return result.loc[:, ~result.columns.duplicated()].sort_index()


def validate_market_data(frame: pd.DataFrame, min_rows: int = 100) -> dict:
    """Validate and summarise market data without silently repairing it."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Market-data index must be timestamps")
    duplicate_dates = int(frame.index.duplicated().sum())
    missing_cells = int(frame.loc[:, REQUIRED_COLUMNS].isna().sum().sum())
    if duplicate_dates:
        raise ValueError(f"Duplicate market dates: {duplicate_dates}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Market dates are not chronological")
    if missing_cells:
        raise ValueError(f"Missing OHLCV values: {missing_cells}")
    if len(frame) < min_rows:
        raise ValueError(f"Only {len(frame)} observations; require at least {min_rows}")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any() or (frame["Volume"] < 0).any():
        raise ValueError("Non-positive prices or negative volume detected")
    return {
        "records": len(frame), "start_date": frame.index.min().date().isoformat(),
        "end_date": frame.index.max().date().isoformat(), "missing_ohlcv_cells": missing_cells,
        "duplicate_dates": duplicate_dates,
    }


def download_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download actual daily bars; no synthetic fallback is permitted."""
    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("Install yfinance to run a real-data pipeline") from error
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    data = normalise_yfinance_frame(raw)
    validate_market_data(data)
    return data


def chronological_split(frame: pd.DataFrame, train_ratio: float = .60, validation_ratio: float = .20) -> Split:
    """Contiguous chronological split; never shuffle time-series observations."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must be positive and leave a test partition")
    train_end, validation_end = int(len(frame) * train_ratio), int(len(frame) * (train_ratio + validation_ratio))
    split = Split(frame.iloc[:train_end].copy(), frame.iloc[train_end:validation_end].copy(), frame.iloc[validation_end:].copy())
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("A split partition is empty")
    return split


def fit_standardizer(train: pd.DataFrame, columns: Iterable[str] = ("Close",)) -> dict:
    """Fit preprocessing parameters on training data only for auditability."""
    values = train.loc[:, list(columns)].astype(float)
    return {column: {"mean": float(values[column].mean()), "std": float(values[column].std(ddof=0) or 1.0)} for column in values}


def build_lagged_features(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """At date t, use only prices through t-1 to forecast Close[t]."""
    table = pd.DataFrame(index=close.index)
    table["lag_1"] = close.shift(1)
    table["mean_5"] = close.shift(1).rolling(5).mean()
    table["mean_20"] = close.shift(1).rolling(20).mean()
    table["return_1"] = close.pct_change().shift(1)
    return table.dropna(), close.loc[table.dropna().index]


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1
    xs = (x - mean) / std
    design = np.c_[np.ones(len(xs)), xs]
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y), (mean, std)


def _predict_ridge(x: np.ndarray, coefficient: np.ndarray, scale: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = scale
    return np.c_[np.ones(len(x)), (x - mean) / std] @ coefficient


def forecast_models(split: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate persistence, moving-average, and validation-selected ridge on held-out test dates."""
    close = pd.concat([split.train["Close"], split.validation["Close"], split.test["Close"]])
    features, target = build_lagged_features(close)
    train_dates, validation_dates, test_dates = split.train.index, split.validation.index, split.test.index
    x_train, y_train = features.loc[features.index.intersection(train_dates)], target.loc[features.index.intersection(train_dates)]
    x_val, y_val = features.loc[features.index.intersection(validation_dates)], target.loc[features.index.intersection(validation_dates)]
    x_test, y_test = features.loc[features.index.intersection(test_dates)], target.loc[features.index.intersection(test_dates)]
    if min(len(x_train), len(x_val), len(x_test)) < 2:
        raise ValueError("Insufficient lagged observations in a partition")
    candidates = [0.01, 0.1, 1.0, 10.0]
    selected_alpha = min(candidates, key=lambda alpha: np.mean((y_val.to_numpy() - _predict_ridge(x_val.to_numpy(), *_fit_ridge(x_train.to_numpy(), y_train.to_numpy(), alpha))) ** 2))
    x_refit, y_refit = pd.concat([x_train, x_val]), pd.concat([y_train, y_val])
    coefficient, scale = _fit_ridge(x_refit.to_numpy(), y_refit.to_numpy(), selected_alpha)
    predictions = pd.DataFrame({
        "actual_close": y_test,
        "persistence": x_test["lag_1"],
        "moving_average_20": x_test["mean_20"],
        "ridge_autoregression": _predict_ridge(x_test.to_numpy(), coefficient, scale),
    }, index=x_test.index)
    predictions.index.name = "Date"
    selection = pd.DataFrame([{"model": "ridge_autoregression", "selected_alpha": selected_alpha, "validation_observations": len(x_val)}])
    return predictions, selection


def forecasting_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    actual = predictions["actual_close"].to_numpy(dtype=float)
    rows = []
    for model in predictions.columns.drop("actual_close"):
        predicted = predictions[model].to_numpy(dtype=float)
        error = actual - predicted
        # Direction compares the forecast's next-day move with the realised move from t-1.
        prior = predictions["actual_close"].shift(1).bfill().to_numpy(dtype=float)
        rows.append({"model": model, "observations": len(actual), "mae": float(np.mean(np.abs(error))),
                     "rmse": float(np.sqrt(np.mean(error ** 2))), "mape_pct": float(np.mean(np.abs(error / actual)) * 100),
                     "directional_accuracy": float(np.mean((predicted - prior > 0) == (actual - prior > 0)))})
    return pd.DataFrame(rows)


def backtest_long_flat(predictions: pd.DataFrame, model: str, cost_bps: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Close-to-close, one-day long/flat strategy with cost charged on every position change."""
    actual, forecast = predictions["actual_close"], predictions[model]
    prior_close = actual.shift(1)
    position = (forecast > prior_close).astype(float).fillna(0.0)
    gross_return = position * actual.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position)
    net_return = gross_return - turnover * (cost_bps / 10_000)
    equity = (1 + net_return).cumprod()
    drawdown = equity / equity.cummax() - 1
    volatility = net_return.std(ddof=1)
    sharpe = float(np.sqrt(252) * net_return.mean() / volatility) if volatility and not np.isnan(volatility) else 0.0
    ledger = pd.DataFrame({"actual_close": actual, "forecast": forecast, "position": position, "turnover": turnover,
                           "gross_return": gross_return, "net_return": net_return, "equity": equity, "drawdown": drawdown})
    metrics = {"model": model, "total_return_pct": float((equity.iloc[-1] - 1) * 100), "annualized_sharpe": sharpe,
               "max_drawdown_pct": float(drawdown.min() * 100), "trade_count": int(turnover.sum()),
               "cost_bps_per_turnover": cost_bps, "observations": len(ledger)}
    return ledger, metrics


def export_csvs(output_dir: Path, quality: pd.DataFrame, forecasts: pd.DataFrame, strategies: pd.DataFrame, selections: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in {"data_quality_summary.csv": quality, "forecast_metrics.csv": forecasts,
                        "strategy_metrics.csv": strategies, "model_selection.csv": selections}.items():
        frame.to_csv(output_dir / name, index=False)
