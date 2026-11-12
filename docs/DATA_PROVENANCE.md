# Data provenance

The provider is Yahoo Finance through `yfinance`, requested with `auto_adjust=False`; Open/High/Low/Close are unadjusted source bars. The pipeline does not reconcile splits or dividends, and Yahoo may revise history. Each run records requested and observed dates, validation summary, and hashes of local input snapshots and output artifacts. Do not commit raw snapshots without confirming redistribution rights.

For offline replay, pass `--snapshot-dir` pointing to a prior verified output directory. Each ticker must have its CSV and matching metadata sidecar with its Yahoo provider declaration and SHA-256. Replay applies the same OHLCV checks as a live response, copies the validated input into the new run, and records its provenance in `manifest.json`. Synthetic fixtures are not accepted as genuine snapshots.
