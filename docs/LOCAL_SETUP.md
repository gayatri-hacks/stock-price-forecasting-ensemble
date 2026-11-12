# Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
uvicorn backend.app:app --reload

cd dashboard
npm ci
npm run dev
```

To run FastAPI on port 8001 instead, use `uvicorn backend.app:app --reload --port 8001`, then start Vite with `VITE_API_TARGET=http://127.0.0.1:8001 npm run dev`. The default proxy target remains `http://127.0.0.1:8000`.

Set `VERIFIED_OUTPUT_DIR=/absolute/path/to/verified-run` before starting FastAPI. Create that directory only through a live verified run or `python main.py --snapshot-dir /path/to/genuine/snapshots --output-dir outputs/replay`.
