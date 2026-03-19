# AGENTS.md - Developer Guidelines for Dietin V2

This document serves as the primary instruction manual for AI coding agents (and human developers) operating in the Dietin V2 repository. Adherence to these rules is mandatory to maintain architectural integrity.

---

## 1. Core Architectural Constraints (DIETIN V2 Master Plan)

All development must adhere to the 3-Tier Architecture:
1.  **Tier 1 (Frontend):** React Client (Legacy & Modern components).
2.  **Tier 2 (Middleware):** Python FastAPI Microservice (`python_backend/`).
3.  **Tier 3 (Services):** Firebase, Supabase, Appwrite, and Gemini/GPT-4o APIs.

### **Critical Rules:**
- **Isolation:** All new backend logic MUST reside in `python_backend/`. Do not pollute `/src`.
- **Zero UI Changes (Phase 1):** Avoid modifying React components unless necessary for API plumbing (e.g., connecting a new endpoint).
- **Progress Documentation:** Log completed milestones in `python_backend/docs/` (e.g., `phase_1_setup.md`).
- **Hardware Target:** Python models are optimized for **Apple Silicon (M-Series MPS)**.

---

## 2. The MHMZ Signature Protocol

**Mandatory:** Every modification to an existing file (especially in `/src`) MUST be accompanied by a comment signed with `MHMZ`.
- **Format:** `// MHMZ: <Short explanation of the change>`
- **Example:** `// MHMZ: Rerouted Gemini call to local FastAPI middleware for deterministic math.`

---

## 3. Build, Lint, and Test Commands

### **Frontend (Root Directory)**
- **Development:** `npm run dev` (Starts Vite dev server)
- **Production Build:** `npm run build` (Builds for web)
- **Linting:** `npm run lint` (Uses ESLint with TypeScript support)
- **Android Build:** `npm run build:android` (Requires Capacitor)
- **Preview:** `npm run preview` (Local preview of production build)

### **Backend (`python_backend/` Directory)**
- **Virtual Environment Setup:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Development Server:** `uvicorn main:app --reload` (Runs on port 8000)
- **Run All Tests:** `pytest` (Run from within `python_backend/`)
- **Run Single Test File:** `pytest tests/test_api.py`
- **Run Specific Test:** `pytest tests/test_api.py::test_health_check`

---

## 4. Code Style & Standards

### **Frontend (TypeScript/React)**
- **Naming:** PascalCase for Components, camelCase for functions/variables.
- **Imports:** Standard ES modules. Group absolute imports first, then relative.
- **Types:** Use strict TypeScript. Define interfaces for props and API responses in `src/lib/types.ts` when shared.
- **Components:** Functional components with Hooks. Use `memo` where performance is critical.
- **Error Handling:** Use the `use-toast` hook for user-facing errors. Prefer `try/catch` for async API calls.
- **Styling:** Tailwind CSS is the standard. Use `cn()` utility (from `src/lib/utils.ts`) for conditional classes.
- **State:** Zustand for global state, React Query for server state.

### **Backend (Python/FastAPI)**
- **Style:** PEP 8 compliance. Use `black` or `ruff` for formatting if available.
- **Typing:** Use Pydantic models (v2) for all Request/Response bodies.
- **Concurrency:** Prefer `async`/`await`. Use `asyncio.gather` for parallelizing IO-bound tasks (e.g., GPT-4o API + Vision Model).
- **Vision Models:** Use `mps` device for Torch/Vision models when available on Apple Silicon.
- **Error Handling:** Raise `HTTPException` with clear detail strings.
- **Logging:** Use standard Python `logging` module for server-side logs.

---

## 5. File Structure Reference

- `/src/pages`: Main view components (Entry points for routes).
- `/src/components`: Reusable UI elements (Shadcn UI base).
- `/src/lib`: Core logic, API clients (`gemini.ts`, `firebase.ts`), and utils.
- `/src/hooks`: Custom React hooks (e.g., `use-toast.ts`).
- `/src/i18n`: Internationalization config and locale files (AR/EN).
- `/python_backend/main.py`: FastAPI entry point.
- `/python_backend/vision_engine.py`: Computer vision and depth estimation.
- `/python_backend/nutrition_engine.py`: Vector search and density mapping.
- `/python_backend/docs`: Progress logs (Phase-by-phase).
- `/python_backend/tests`: Pytest suite for backend logic.

---

## 6. Common Patterns & API Contracts

### **Deterministic Math Pattern**
LLMs classify food; Python computes physics.
1.  **GPT-4o:** Returns classification (e.g., "Fried Chicken").
2.  **Vision Engine:** Estimates volume in $cm^3$ via depth map (Depth Anything V2).
3.  **Nutrition Engine:** Fetches "Fried Chicken" density ($g/cm^3$) from ChromaDB using hybrid search (BM25 + Dense).
4.  **Backend:** Mass = Volume * Density.

### **The Liquid Fork**
If GPT returns `is_liquid: true`, ignore depth tensors and apply "Container Fill Heuristic" based on rim detection (MobileSAM).

### **API Response Schema (Standard)**
All AI analysis endpoints should return:
```json
{
  "success": true,
  "data": {
    "foodName": "Chicken",
    "calories": 250,
    "macros": { "protein": 25, "carbs": 0, "fat": 15 },
    "healthScore": 8.5,
    "confidenceScore": 0.92,
    "warnings": []
  }
}
```

---

## 7. Development Workflow

1.  **Backend First:** Implement logic in `python_backend/`.
2.  **Test:** Add unit tests in `python_backend/tests/`.
3.  **Document:** Update `python_backend/docs/` with the phase summary.
4.  **Connect:** Modify `src/lib/` to point to the new endpoint.
5.  **Sign:** Add the `MHMZ` signature to all modified files in `src/`.

---

## 8. Mobile & PWA Considerations

- **Capacitor:** Use `@capacitor-firebase/authentication` for native auth.
- **PWA:** Managed via `vite-plugin-pwa`. Ensure all new assets are added to the service worker manifest.
- **Android:** Sync changes using `npx cap sync` after a web build.

