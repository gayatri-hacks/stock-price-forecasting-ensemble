# Web architecture

The React/Vite dashboard is a read-only client of FastAPI. FastAPI reads completed verified-run CSV ledgers and the SHA-256 manifest from `VERIFIED_OUTPUT_DIR`; it never retrains models or creates prices. With no valid manifest the API returns 503 and the UI presents Dataset unavailable rather than financial KPIs.

Data flow: user control → React/TanStack Query → allowlisted FastAPI endpoint → validated snapshot-backed ledger → chart/KPI. The dashboard is historical research, not a trading platform.
