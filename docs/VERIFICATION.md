# Verification

Run `python -m pytest -q tests`, then execute `python main.py --start 2020-01-01 --end 2025-12-31`. Tests use deterministic synthetic fixtures only; they do not claim to be market data. Live download is deliberately not part of CI because provider availability and historical responses may change.
