# API contract

`/health` is process health and `/ready` reports verified-artifact availability. `/api/overview`, `/api/forecast-metrics`, `/api/predictions`, `/api/backtest`, `/api/equity-curve`, `/api/drawdown`, `/api/cost-sensitivity`, `/api/data-quality`, `/api/provenance`, and `/api/manifest` expose only verified files. Tickers are AAPL/NVDA/SPY, models are the three supported baselines, and per-side costs are 0/5/10/20 bps. Unsupported selections return 422; absent/unverified artifacts return 503. Downloads use a fixed allowlist.
