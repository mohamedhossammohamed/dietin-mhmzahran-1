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
