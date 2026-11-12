"""Executive report generation for verified pipeline outputs."""
from pathlib import Path
import matplotlib

# CI and headless workstations must never attempt to initialise a GUI backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _markdown_table(frame: pd.DataFrame, floatfmt: str | None = None) -> str:
    """Small dependency-free Markdown renderer so reports work offline."""
    headers = [str(column) for column in frame.columns]
    def value(item):
        return format(item, floatfmt) if floatfmt and isinstance(item, float) else str(item)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    rows.extend("| " + " | ".join(value(item) for item in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def generate_report(output_dir: Path, quality: pd.DataFrame, forecasts: pd.DataFrame, strategies: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart = output_dir / "forecast_rmse_by_ticker.png"
    pivot = forecasts.pivot(index="ticker", columns="model", values="rmse")
    pivot.plot(kind="bar", figsize=(10, 5), title="Held-out forecast RMSE (lower is better)")
    plt.ylabel("USD per share")
    plt.tight_layout(); plt.savefig(chart, dpi=150); plt.close()
    report = output_dir / "FINANCIAL_ANALYTICS_REPORT.md"
    report.write_text("""# Financial Analytics Executive Report

## Business objective

Provide a reproducible, historical analysis of daily AAPL, NVDA and SPY prices. This is research and reporting, not investment advice or a live-trading system.

## Controls and methodology

Daily OHLCV is retrieved from Yahoo Finance through yfinance. Dates are preserved; duplicate dates, missing OHLCV fields, invalid prices, and chronology failures stop the run. Splits are chronological (60% train / 20% validation / 20% test). The validation partition selects ridge regularisation; the test partition remains held out. The persistence baseline is reported alongside a 20-day moving average and a lightweight autoregression.

## Dataset quality and coverage

""" + _markdown_table(quality) + """

## Historical forecasting performance

Metrics below use held-out test dates only. Raw-dollar RMSE is not comparable across securities with different price levels; use NRMSE/MAPE for scale-aware comparison. Skill is relative to persistence MAE. They are model-error metrics, not trading returns.

""" + _markdown_table(forecasts, ".4f") + """

## Strategy backtest (separate from model accuracy)

The illustrative long/flat policy makes its signal after the preceding close, buys at the next available open, and sells at that session's close. It is flat overnight: each long session pays both entry and exit costs, with a 0/5/10/20 basis-point per-side sensitivity. This is not a tradability claim.

""" + _markdown_table(strategies, ".4f") + """

## Risk, limitations and conclusions

- Yahoo Finance is a convenient public source and may revise historical records; reruns can differ.
- Results cover a single historical sample and do not establish forecasting skill, causality, or investability.
- The backtest excludes taxes, borrow costs, market impact, intraday execution uncertainty, and corporate-action verification.
- Input snapshots, date-level ledgers, model-selection records and a SHA-256 manifest are written with every run. Yahoo data can be revised, so separate source-data changes from code changes.
- Deep-learning legacy phases were not used in the verified report: their labels and target alignment require separate redevelopment.
- Executive conclusion: use the exported tables for transparent comparison; do not infer guaranteed returns or select a model solely from one sample.
""")
    return report
