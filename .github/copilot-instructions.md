# Dietin - Development Guide

Dietin is a cross-platform fitness app with AI-powered nutrition tracking. It uses a React + TypeScript frontend with a Python FastAPI backend for computer vision-based meal analysis.

## Build, Lint, and Test Commands

### Frontend (React + Vite)
```bash
# Development server (runs on port 3000)
npm run dev

# Production build
npm run build

# Development build (preserves dev environment variables)
npm run build:dev

# Preview production build
npm run preview

# Lint all files
npm run lint

# Build and open Android app
npm run build:android
```

### Backend (Python FastAPI)
```bash
# Navigate to backend
cd python_backend

# Install dependencies (use venv recommended)
pip install -r requirements.txt

# Start backend server (runs on port 8000)
uvicorn main:app --reload

# Run all tests
pytest

# Run specific test file
pytest tests/test_main.py

# Run tests with verbose output
pytest -v

# Health check endpoint
curl http://localhost:8000/health
```

**Important**: The backend requires the frontend to be proxied through `localhost:8000` for image analysis. The Python backend uses Apple Silicon MPS acceleration when available.

## Architecture Overview

### 3-Tier Architecture
```
React Frontend (Tier 1)
    ↓ 
Python FastAPI Middleware (Tier 2)
    ↓
Firebase/Supabase/Appwrite (Tier 3)
```

### Frontend Architecture
- **Framework**: React 18 + TypeScript + Vite
- **Routing**: React Router v6 (main routes in `src/App.tsx`)
- **State Management**: Zustand with persistence
  - `userStore`: User profile, goals, and settings
  - `mealStore`: Daily calorie entries and meal data
  - `nutritionStore`: Nutrition tracking data
  - `workoutStore`: Workout plans and logs
  - `hydrationStore`: Water intake tracking
  - `analyticsStore`: Analytics and progress data
- **UI Framework**: Radix UI + Tailwind CSS + Shadcn UI components
- **Animation**: Framer Motion for page transitions and UI animations
- **3D Graphics**: React Three Fiber for 3D elements
- **Forms**: React Hook Form + Zod validation
- **Data Fetching**: TanStack Query (React Query)

### Backend Architecture
- **Framework**: FastAPI with async/await patterns
- **Computer Vision**: Depth Anything V2 for volume estimation
- **Food Classification**: Routed to OpenAI GPT-4o (proxied securely)
- **Nutrition Database**: ChromaDB with USDA food data + semantic search (all-MiniLM-L6-v2 embeddings + BM25)
- **Multi-Modal Input**: Supports image, speech (Whisper-tiny), and nutrition label OCR
- **Device Optimization**: Automatically uses Apple MPS, CUDA, or CPU

### Backend Pipeline (Concurrent Execution)
The FastAPI backend uses `asyncio.gather` to run vision and classification concurrently:
```python
async def analyze_image_pipeline(image_bytes):
    vision_task = asyncio.to_thread(vision_engine.get_volume, image_bytes)
    gpt_task = gpt_client.get_classification(image_bytes)
    volume_data, class_data = await asyncio.gather(vision_task, gpt_task)
    return merge_and_calculate(volume_data, class_data)
```

### Key Data Flow
1. User captures food photo in `MealAnalysis.tsx`
2. Image sent to `src/lib/gemini.ts` (intercepted proxy)
3. `gemini.ts` routes to `http://localhost:8000/api/v1/analyze/image`
4. Python backend:
   - Runs MobileSAM segmentation + Depth Anything V2 (volume estimation)
   - Calls GPT-4o for food classification
   - Queries ChromaDB for density constants
   - Computes: `mass = volume × density`
   - Returns structured nutrition data
5. Frontend displays results with confidence score

### API Contract
**Frontend expects backend `confidenceScore` in 0-1 range** (frontend multiplies by 100 for UI display).

Frontend requests route to:
- `/api/v1/analyze/image` - Image analysis (vision + AI)
- `/api/v1/analyze/speech` - Audio transcription (Whisper)
- `/api/v1/analyze/label` - Nutrition label OCR
- `/api/v1/proxy/generate` - Text generation proxy

## Key Conventions

### Code Modification Protocol (MHMZ Signature)
**Critical**: When modifying existing frontend files (especially in `src/lib/`), add an explanatory comment signed with `MHMZ`:
```typescript
// MHMZ: Rerouted Gemini call to local FastAPI middleware for deterministic math and secure API gateway.
```
This tracks architectural changes and prevents confusion about deviations from standard patterns.

### Backend Development Rules
1. **Isolation**: All backend code lives in `python_backend/` directory (never pollute `src/`)
2. **Progress Documentation**: Document completed phases in `python_backend/docs/phase_N_*.md`
3. **Zero Frontend Changes**: Backend changes should not require React component modifications (except `src/lib/gemini.ts` proxy rerouting)

### State Management Patterns
- Use Zustand stores for global state (user, meals, workouts, etc.)
- Stores persist to localStorage via `zustand/middleware`
- Firebase sync happens automatically via `userStore` actions
- Always use `getLocalDateKey()` helper for date-based keys to avoid UTC shift issues

### Component Organization
- **Pages**: Top-level routes in `src/pages/`
- **Components**: Reusable components in `src/components/`
- **UI Components**: Shadcn components in `src/components/ui/`
- **Hooks**: Custom hooks in `src/hooks/`
- **Types**: Shared TypeScript types in `src/types/` and `src/lib/types.ts`

### Styling Conventions
- Use Tailwind CSS utility classes
- Use `cn()` utility from `src/lib/utils.ts` for conditional classes
- Shadcn components follow Radix UI patterns
- Custom animations via Framer Motion
- Responsive: Mobile-first design with breakpoints

### Firebase Integration
- Authentication via Firebase Auth
- Firestore for user data persistence
- Firebase Analytics for tracking
- Local emulators configured (ports: Auth 9099, Firestore 8080, Storage 9199)
- Always wrap writes with offline/online state checks

### Mobile (Capacitor)
- App ID: `com.dietin.app`
- Builds target Android primarily (`npm run build:android`)
- Uses Capacitor plugins: Camera, Geolocation, Motion, Browser
- Configure in `capacitor.config.ts`

### Path Aliases
Import from `@/` to reference `src/` directory:
```typescript
import { Button } from "@/components/ui/button"
import { useUserStore } from "@/stores/userStore"
```

### Testing
- Backend: pytest with async support (`pytest-asyncio`)
- Frontend: No test framework currently configured
- Backend tests located in `python_backend/tests/`

### Environment Variables
Required environment variables (`.env`):
- `VITE_GOOGLE_AI_KEY` - Google AI API key (currently intercepted by backend)
- Firebase configuration keys
- Supabase keys
- Additional API keys per backend requirements

### Deployment
- Frontend builds to `dist/` directory
- Backend runs as standalone FastAPI service
- Firebase hosting configured (see `firebase.json`)
- Mobile builds via Capacitor to native platforms

## Project-Specific Patterns

### Liquid/Transparent Food Handling
Backend implements a "Liquid Fork" pattern: when GPT-4o returns `is_liquid: true`, depth estimation is bypassed and container fill heuristics are used instead.

### OCR Override
If nutrition label detected, the entire vision pipeline is skipped and OCR extraction takes precedence for deterministic accuracy.

### Segmentation Offload Strategy
API designed to eventually receive pre-cropped masks from mobile device (to offload MobileSAM from backend to client), but currently backend handles all segmentation.

### Confidence Gating
Backend evaluates composite score (Volume Confidence × Classification Confidence). Low scores trigger warnings in the response payload.

### Macro Goals Calculation
User metabolism and macro goals calculated via `src/lib/calculations.ts` using standardized formulas (BMR, TDEE adjustments).

### Date Handling
Always use `getLocalDateKey(date)` helper to format dates as `yyyy-MM-dd` to avoid timezone issues with Firebase and localStorage.

## Documentation References

- Master implementation plan: `DIETIN_V2_MASTER_PLAN.md`
- Backend phase docs: `python_backend/docs/phase_*.md`
- README: `README.md`
- License: AGPL-3.0 (modifications must be open-sourced)
