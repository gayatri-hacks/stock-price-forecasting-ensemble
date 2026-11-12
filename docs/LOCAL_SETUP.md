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

Set `VERIFIED_OUTPUT_DIR=/absolute/path/to/verified-run` before starting FastAPI. Create that directory only through a live verified run or `python main.py --snapshot-dir /path/to/genuine/snapshots --output-dir outputs/replay`.
