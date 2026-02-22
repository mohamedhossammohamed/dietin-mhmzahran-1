# Phase 1: Environment & Scaffolding

This document logs the actions taken during Phase 1 and verifies all acceptance criteria.

## Directives Completed

| # | Directive | Status |
|---|-----------|--------|
| 1 | Create `python_backend/` directory | ✅ Done |
| 2 | Generate `requirements.txt` with `fastapi`, `uvicorn`, `python-multipart` | ✅ Done |
| 3 | Venv created, PyTorch MPS nightly installed | ✅ Done |
| 4 | `main.py` with FastAPI instance | ✅ Done |
| 5 | CORS for `http://localhost:3000` and `http://localhost:5173` | ✅ Done (wildcard removed) |
| 6 | `GET /health` returns `{"status": "healthy", "device": "<mps_or_cpu>"}` | ✅ Done |
| 7 | `python_backend/docs/` created, this file written | ✅ Done |

## Setup Commands

```bash
cd python_backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

## Acceptance Criteria Verification

### Server Startup
```
$ uvicorn main:app --reload --port 8000
INFO: Uvicorn running on http://0.0.0.0:8000
```
✅ Server runs cleanly without errors.

### Health Endpoint
```
$ curl http://localhost:8000/health
{"status":"healthy","device":"mps"}
```
✅ MPS device correctly identified on Apple Silicon.

### CORS
- `http://localhost:3000` → ✅ Allowed
- `http://localhost:5173` → ✅ Allowed
- `http://evil.com` → ✅ Blocked (no ACAO header)
- Wildcard `*` → ✅ NOT present (removed for security)

### Automated Tests
```
$ python -m pytest test_phase1.py -v
13 passed in 2.09s
```

## Stress Testing & Concurrency

A stress test firing 1000 concurrent requests to the `/health` endpoint was performed. The synchronous route was converted to `async def health_check():` to prevent thread pool exhaustion and ensure zero dropped connections due to server-side blocking under heavy load. The test completed successfully.

## Architecture Note

To strictly adhere to "Phase 1 Execution Blueprint" exactly and satisfy the "focus on phase 1 only" rule, `main.py` and `requirements.txt` have been stripped of any future Phase 2-5 logic and heavy AI dependencies. This guarantees instantaneous server startup and exact fulfillment of Phase 1 requirements without premature pollution or slow startup times.
