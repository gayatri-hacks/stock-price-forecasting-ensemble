# Dashboard verification

Offline backend tests use fixtures explicitly labelled as non-market data. They cover unavailable state, endpoint allowlists, date validation, download safety, finite JSON output, and ledger mapping. The dashboard is designed to show an unavailable state until a provenance-marked genuine run exists. Browser QA requires a local browser environment and a genuine source snapshot; neither is implied by unit tests.
