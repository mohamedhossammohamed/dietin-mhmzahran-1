# Phase 4: API Contract & Orchestration

This document logs the actions taken during Phase 4.
We bridged the React frontend to the new Python backend to leverage deterministic math and vision computation locally.

1. Updated `python_backend/main.py`:
   - Added `POST /api/v1/analyze/image`.
   - Used `asyncio.to_thread` for the synchronous depth estimation model to unblock the event loop.
   - Piped a mock GPT-4o classification with `asyncio.sleep` simulating network latency.
   - Merged the volume with ChromaDB deterministic density to calculate mass and macros.
   - Returned the exact Pydantic schema required by the React components (`success`, `data`, `macros`, etc.).
2. Deprecated the original Gemini SDK call in the React application:
   - Modified `analyzeImage` in `src/lib/gemini.ts`.
   - Safely appended the mandatory `// MHMZ: Rerouted Gemini call to local FastAPI middleware to ensure deterministic math` signature.
3. Observed zero UI modifications in accordance with the Phase 1 strict architectural constraint.
4. Revised Frontend Integration:
   - Discovered that the React UI (`MealAnalysis.tsx`) intermittently bypassed `analyzeImage` and called `genAI` directly.
   - Refactored `genAI` in `src/lib/gemini.ts` using a proxy pattern to automatically intercept any `generateContent` calls containing an image payload.
   - Images are transparently routed to the local FastAPI backend (`http://localhost:8000/api/v1/analyze/image`), and the python output is mapped back to the mock JSON string format the React UI originally expected from Gemini.
   - This ensures full adherence to the architectural constraint (zero UI modifications) while perfectly fulfilling the requirement that the frontend remains completely unaware that the data came from Python.

## Update: Stress Test & Master Plan Fulfillment Audit

A comprehensive stress test and code audit against the `DIETIN_V2_MASTER_PLAN.md` revealed a few missing gaps which have now been mitigated in `main.py`:

1. **Robust Error Handling:** 
   - Resolved HTTP 500 crash bugs that occurred during empty or corrupt file uploads.
   - The endpoint now elegantly catches `PIL.UnidentifiedImageError` and missing valid payloads, triggering a standard HTTP 400 Exception to gracefully fail to the React app.
2. **Pattern D Full Implementation:**
   - Integrated the "Segmentation-Offload Ready" API contract (Pattern D).
   - Added conditional logic to default to `MODE 1` (MobileSAM on backend) or gracefully fallback to `MODE 2` (pre-cropped MobileSAM crops from the phone).
3. **Strict Mock Adherence:**
   - Modified `mock_gpt4o_classification` to return EXACTLY `{"foodName": "Chicken", "prep": "fried", "is_liquid": False}`, perfectly matching Phase 4 Item 2 directives.

**Status:** Phase 4 is completely aligned with the DIETIN Master Architecture, locally verified to be crash-safe, and fully executed.

