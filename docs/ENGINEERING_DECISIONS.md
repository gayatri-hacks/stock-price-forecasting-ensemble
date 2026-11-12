# Engineering decisions

The supported path intentionally uses NumPy/Pandas ridge regression rather than heavyweight legacy dependencies. It fails loudly on absent provider data and invalid OHLCV, never synthesizes prices, keeps artifacts in a caller-selected output directory, and separates test forecasting metrics from strategy illustrations.
