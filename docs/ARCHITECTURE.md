# Architecture

`main.py` downloads unadjusted Yahoo Finance daily OHLCV, validates it, writes a per-run local snapshot, makes chronological 60/20/20 splits, evaluates the supported models, writes ledgers/CSVs, then emits a report and SHA-256 manifest. Legacy phase directories are historical research artifacts and are not inputs to this path.
