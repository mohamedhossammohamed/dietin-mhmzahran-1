# Phase 2: The Nutrition Engine (Data Layer)

**Status:** ✅ Complete — all 38 stress tests passing  
**Last updated:** 2026-02-21

---

## What was built

A local ChromaDB vector database acting as a "Deterministic Physics Engine" — maps food strings to strict USDA density values via hybrid dense+sparse search, eliminating LLM hallucination.

### Files

| File | Purpose |
|------|---------|
| `nutrition_engine.py` | Core engine — `seed_database()`, `HybridSearchEngine`, `NutritionRecord` |
| `test_phase2.py` | 38-test stress suite |

## Implementation Checklist (per Master Plan)

- [x] Created `python_backend/nutrition_engine.py`
- [x] Added `chromadb`, `sentence-transformers`, `rank_bm25`, `pandas` to `requirements.txt`
- [x] `seed_database()` — initializes local ChromaDB with 15 complex food items
  - Each record has: `id`, `name`, `preparation`, `density_g_cm3`, `yield_factor`
- [x] `class HybridSearchEngine` with `search(query, prep_filter)` method
  - Dense vector search via `all-MiniLM-L6-v2` (top 10 candidates)
  - Sparse BM25 keyword re-ranking
  - Prep-filter penalty (0.2× for mismatched preparation)
  - Returns strictly typed Pydantic `NutritionRecord`
- [x] Acceptance criteria verified: "Spicy crispy chicken" + "fried" → "Fried Chicken" @ density 0.5

## Bug Fixes Applied (Stress Test Audit)

1. **Missing deps in `requirements.txt`** — Phase 2 packages were installed but not declared
2. **Return type** — `search()` → `Optional[NutritionRecord]` (was falsely typed as non-optional)
3. **Singleton ChromaDB client** — prevents multiple `PersistentClient` file-lock contention
4. **Prep-filter normalization** — `.strip().lower()` for case-insensitive matching
5. **Float coercion** — explicit `float()` cast for metadata from ChromaDB (may deserialize as strings)
6. **Edge-case guards** — empty/None query handling added

## Test Results (38/38 passing)

```
test_phase2.py::TestAcceptanceCriteria          (3 tests)  ✅
test_phase2.py::TestPrepDiscrimination          (4 tests)  ✅
test_phase2.py::TestSeedingIntegrity            (2 tests)  ✅
test_phase2.py::TestDensityIntegrity            (15 tests) ✅
test_phase2.py::TestPydanticModel               (2 tests)  ✅
test_phase2.py::TestEdgeCases                   (7 tests)  ✅
test_phase2.py::TestPrepPenalty                 (1 test)   ✅
test_phase2.py::TestIdempotency                 (1 test)   ✅
test_phase2.py::TestConcurrency                 (1 test)   ✅
test_phase2.py::TestBM25Sanity                  (2 tests)  ✅
────────────────────────────────────────────────────────────
38 passed, 0 failed in 18.30s
```
