# Data provenance

The provider is Yahoo Finance through `yfinance`, requested with `auto_adjust=False`; Open/High/Low/Close are unadjusted source bars. The pipeline does not reconcile splits or dividends, and Yahoo may revise history. Each run records requested and observed dates, validation summary, and hashes of local input snapshots and output artifacts. Do not commit raw snapshots without confirming redistribution rights.
