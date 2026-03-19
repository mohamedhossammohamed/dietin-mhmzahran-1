# DIETIN V2 — Technical Debt & Runtime Warnings

This document tracks existing technical debt, runtime terminal warnings, and non-crashing bugs identified during the backend testing and verification phase (March 2026).

## Terminal Warnings (Runtime Issues)

During application startup (`uvicorn main:app`), several warnings are printed to the terminal. While they do not crash the application currently, they represent significant technical debt that must be addressed before the respective dependencies drop support entirely.

### 1. Python Version EOL Warnings
**Description:** The Google API core and OAuth2 libraries are throwing `FutureWarning` because the environment is running Python `3.9.6`.
**Log Output:**
```text
FutureWarning: You are using a non-supported Python version (3.9.6). Google will not post any further updates to google.api_core supporting this Python version. Please upgrade to the latest Python version, or at least Python 3.10, and then update google.api_core.
FutureWarning: You are using a Python version 3.9 past its end of life. Google will update google-auth with critical bug fixes on a best-effort basis, but not with any other fixes or features.
```
**Action Required:**
- Update the development and production Python environments to `Python 3.10` or higher (preferably `Python 3.12` or `3.13` for longevity).
- Update the `google-auth` and `google-api-core` pip packages afterward.

### 2. Google Generative AI Package Deprecation
**Description:** The backend currently uses `google.generativeai` for the Gemini LLM integration. This package has hit End-of-Life (EOL) and will no longer receive updates.
**Log Output:**
```text
FutureWarning: All support for the `google.generativeai` package has ended. It will no longer be receiving updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
```
**Action Required:**
- Remove `google.generativeai` from `requirements.txt`.
- Install `google.genai`.
- Refactor all functions in `main.py` calling `genai.GenerativeModel(...)` to use the new SDK nomenclature. *(A TODO comment has been added to `main.py` directly mapping this out).*

### 3. Hugging Face Image Processor Warning
**Description:** The `transformers` pipeline for `Depth-Anything-V2-Small-hf` is loading a slow image processor because the fast variant isn't explicitly requested.
**Log Output:**
```text
UserWarning: Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52...
```
**Action Required:**
- In `vision_engine.py`, add `use_fast=True` to the `pipeline()` initialization call to clear the warning and slightly improve memory/startup performance. *(A TODO comment has been added to `vision_engine.py` directly mapping this out).*

---

## Action Plan

These issues have been tagged deeply within the codebase using `TODO(TechDebt)` comments. 
No immediate code freezes are required since these are warnings (not `Exception`s or HTTP `500`s), but upgrading the Python environment and transitioning the Google SDK should be scheduled for the next minor sprint.
