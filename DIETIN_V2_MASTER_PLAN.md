# **DIETIN V2: BACKEND EXECUTION PLAN & API CONTRACT**

**Status:** Execution Ready  
**Constraint:** Strict Backend Modification Only (Zero React UI Changes for Phase 1\)  
**Target Hardware:** Apple Silicon (MacBook Air \- M-Series mps)

## **0\. Development Constraints & Rules (NEW)**

To prevent "AI Spaghetti Code" and maintain a clean project state, all development MUST adhere to these rules:

1. **Isolation:** All new backend code must be created in a strictly separate directory named python\_backend/. It must not pollute the existing React /src structure.  
2. **Progress Documentation:** A dedicated directory python\_backend/docs/ must be created. Every completed phase must be documented here (e.g., phase\_1\_summary.md) to track progress and architectural decisions.  
3. **MHMZ Signature Protocol:** While the goal is zero frontend changes, connecting the new backend *will* require modifying existing API service files (like src/lib/gemini.ts). **Any** modification to an existing file MUST be accompanied by an explanatory comment signed with MHMZ.  
   * *Example:* // MHMZ: Rerouted Gemini call to local FastAPI middleware to ensure deterministic math and secure key gateway.

## **1\. Executive Alignment & The 3-Tier Architecture**

To achieve the CEO's objectives without altering the frontend or Firebase infrastructure today, we are inserting a **Python FastAPI Middleware** (Tier 2\) between the React Client (Tier 1\) and Firebase (Tier 3).  
The frontend's src/lib/gemini.ts will simply be re-pointed from https://generativelanguage.googleapis.com to http://localhost:8000/api/v1/analyze.

### **CEO Objectives Mapped to Backend Mechanics**

* **Secure API Gateway (No Client-Side Keys):** The React mobile app will **never** hold the OpenAI API key. The Python backend securely injects the key and proxies the call to GPT-4o, preventing malicious key extraction.  
* **Deterministic Math:** Python computes Mass \= Volume × Density. GPT-4o only outputs classification strings (e.g., "fried chicken").  
* **Prep-Aware Density & Semantic Mapping:** Python queries a local ChromaDB instance (USDA data) using all-MiniLM-L6-v2 embeddings combined with BM25 keyword filtering to fetch density constants.  
* **Future-Proofed Segmentation (MobileSAM):** While the frontend currently sends full images, the backend will utilize MobileSAM or SAM 2 Tiny. The API is structured to accept pre-cropped masks in the future, smoothly migrating the compute cost to the user's device without rewriting the backend.  
* **Confidence Gating:** Python evaluates a composite score (Volume Confidence × Class Confidence). If low, Python injects a warning string into the JSON payload.

## **2\. Core Backend Patterns**

### **Pattern A: The Async Race (asyncio.gather)**

To beat the 4-second latency target, the FastAPI endpoint will execute the heavy local vision models and the external GPT-4o API call concurrently.  
\# Conceptual Implementation  
async def analyze\_image\_pipeline(image\_bytes):  
    vision\_task \= asyncio.to\_thread(vision\_engine.get\_volume, image\_bytes)  
    gpt\_task \= gpt\_client.get\_classification(image\_bytes)  
      
    volume\_data, class\_data \= await asyncio.gather(vision\_task, gpt\_task)  
    return merge\_and\_calculate(volume\_data, class\_data)

### **Pattern B: The Liquid Fork (Soup Problem)**

Depth models fail on transparent/flat liquids. The Python backend will implement a routing switch.

* If GPT-4o returns is\_liquid: true, the backend discards the Depth Anything V2 tensor and applies a mathematical "Container Fill Heuristic" based on MobileSAM rim detection.

### **Pattern C: Source of Truth (OCR Override)**

If the user captures a nutrition label, the backend skips the Vision \+ Density pipeline entirely. It extracts the table using OCR and returns the deterministic macros, pausing the pipeline if a serving size multiplier is required from the user.

### **Pattern D: The "Segmentation-Offload Ready" API**

To satisfy the CEO's directive to eventually run MobileSAM on the phone, the API contract supports two modes. Currently, it defaults to Mode 1\.  
@app.post("/api/v1/analyze/image")  
async def analyze(  
    full\_image: UploadFile \= File(None),   
    context\_image: UploadFile \= File(None),   
    crops: List\[UploadFile\] \= File(None)  
):  
    if full\_image:  
        \# MODE 1 (Current): Backend runs MobileSAM  
        masks \= vision\_engine.run\_mobile\_sam(full\_image)  
    elif context\_image and crops:  
        \# MODE 2 (Future): Phone ran MobileSAM, backend skips segmentation  
        masks \= crops

## **3\. Phase-by-Phase Execution Blueprints (For AI Agents)**

*The following phases are formatted as strict "Agent Directives". When starting a new phase, copy the block and provide it to your AI coding assistant (e.g., Cursor) to guarantee precise, context-aware code generation.*

### **Phase 1: Environment & Scaffolding Blueprint**

\*\*Role:\*\* Expert Python DevOps Engineer  
\*\*Context:\*\* We are building a standalone FastAPI microservice (\`python\_backend/\`) for an existing React application. The target hardware is an Apple Silicon Mac (M-Series).  
\*\*Task Directive:\*\*  
1\. Create a directory named \`python\_backend\`.  
2\. Generate a \`requirements.txt\` containing: \`fastapi\`, \`uvicorn\`, \`python-multipart\`.  
3\. Provide the exact bash commands to create a virtual environment (\`venv\`) and install the PyTorch nightly build specifically optimized for Apple Metal Performance Shaders (MPS): \`pip install \--pre torch torchvision torchaudio \--extra-index-url https://download.pytorch.org/whl/nightly/cpu\`.  
4\. Create \`main.py\` containing a FastAPI instance.  
5\. Configure CORS middleware in \`main.py\` to allow origins \`http://localhost:3000\` and \`http://localhost:5173\`.  
6\. Implement a \`GET /health\` endpoint that returns \`{"status": "healthy", "device": "\<mps\_or\_cpu\>"}\`.  
7\. Create a folder \`python\_backend/docs/\` and generate a \`phase\_1\_setup.md\` logging these actions.  
\*\*Acceptance Criteria:\*\* \`uvicorn main:app \--reload\` runs cleanly on port 8000 without errors, and the health check correctly identifies the MPS device if available.

### **Phase 2: The Nutrition Engine (Data Layer) Blueprint**

\*\*Role:\*\* Senior AI Data Engineer  
\*\*Context:\*\* We need a local vector database to act as the "Deterministic Physics Engine" for a nutrition app. We are avoiding LLM hallucinations by mapping food strings to strict USDA density values.  
\*\*Task Directive:\*\*  
1\. Create \`python\_backend/nutrition\_engine.py\`.  
2\. Add \`chromadb\`, \`sentence-transformers\`, \`rank\_bm25\`, and \`pandas\` to \`requirements.txt\`.  
3\. Implement a \`seed\_database()\` function that initializes a local ChromaDB collection. Populate it with a hardcoded mock JSON dataset of 15 complex food items (e.g., "fried chicken", "raw chicken", "white rice"). Each record MUST have: \`id\`, \`name\`, \`preparation\`, \`density\_g\_cm3\` (float), and \`yield\_factor\` (float).  
4\. Implement \`class HybridSearchEngine:\`.  
5\. Write the \`search(query: str, prep\_filter: str)\` method. It must:  
   \- Perform a dense vector search using \`all-MiniLM-L6-v2\` to get the top 10 candidates.  
   \- Apply a sparse BM25 keyword penalty. If \`prep\_filter\`="fried" but the candidate is "raw", penalize the score heavily.  
   \- Return the highest-scoring match as a strictly typed Pydantic model (\`NutritionRecord\`).  
6\. Log completion in \`python\_backend/docs/phase\_2\_nutrition\_engine.md\`.  
\*\*Acceptance Criteria:\*\* Querying "Spicy crispy chicken" with filter "fried" must reliably return the "Fried Chicken" record and its exact \`density\_g\_cm3\` without hallucination.

### **Phase 3: The Vision Engine (The Eye) Blueprint**

\*\*Role:\*\* Principal Computer Vision Researcher  
\*\*Context:\*\* We are processing food images locally on Apple Silicon to calculate physical volume in cubic centimeters ($cm^3$).   
\*\*Task Directive:\*\*  
1\. Create \`python\_backend/vision\_engine.py\`.  
2\. Add \`transformers\` and \`pillow\` to \`requirements.txt\`.  
3\. Implement a singleton class \`VisionPipeline\` that loads \`depth-anything/Depth-Anything-V2-Small\` onto the \`mps\` device.  
4\. Implement \`def estimate\_volume(image\_bytes: bytes) \-\> float:\`  
   \- Convert bytes to PIL Image.  
   \- Run inference to generate a relative depth map.  
   \- IMPLEMENT THE MATH: Apply a standard mobile camera intrinsic matrix assumption (e.g., FOV 70 degrees, distance 30cm) to integrate the depth pixels into an estimated volume in $cm^3$. Add thorough code comments explaining the geometric assumptions.  
5\. Create a stub function \`def run\_mobile\_sam(image\_bytes: bytes)\` returning a dummy bounding box for now (placeholder for Phase 4 integration).  
6\. Log completion in \`python\_backend/docs/phase\_3\_vision\_ssengine.md\`.  
\*\*Acceptance Criteria:\*\* The script accepts an image and returns a deterministic float representing volume. The math must be logically sound based on pinhole camera geometry.

### **Phase 4: API Contract & Orchestration Blueprint**

\*\*Role:\*\* Senior Full-Stack Architect  
\*\*Context:\*\* We are bridging the new Python backend to the existing React frontend. The frontend expects a specific JSON schema previously generated by the Gemini SDK.  
\*\*Task Directive:\*\*  
1\. In \`python\_backend/main.py\`, implement \`POST /api/v1/analyze/image\`.  
2\. Utilize Python's \`asyncio.gather\` to concurrently execute:  
   \- \`vision\_engine.estimate\_volume()\`  
   \- A mock API call to OpenAI (use a dummy sleep and return \`{"foodName": "Chicken", "prep": "fried", "is\_liquid": False}\`).  
3\. Merge the volume and the database density (from \`nutrition\_engine.py\`) to calculate \`mass \= volume \* density\`. Calculate calories deterministically.  
4\. The endpoint MUST return this exact Pydantic schema:   
   \`{"success": bool, "data": {"foodName": str, "calories": int, "macros": {"protein": int, "carbs": int, "fat": int}, "healthScore": float, "confidenceScore": float, "warnings": list}}\`.  
5\. FRONTEND MODIFICATION: Open \`src/lib/gemini.ts\`. Deprecate the existing \`analyzeImage\` logic. Reroute the \`fetch\` call to \`http://localhost:8000/api/v1/analyze/image\`.  
6\. STRICT RULE: You must add the comment \`// MHMZ: Rerouted Gemini call to local FastAPI middleware to ensure deterministic math\` immediately above the modified fetch call in \`gemini.ts\`.  
\*\*Acceptance Criteria:\*\* The React frontend successfully displays nutrition data when a photo is uploaded, completely unaware that the data came from Python instead of the Gemini SDK.

### **Phase 5: Advanced Multi-Modal Inputs Blueprint**

\*\*Role:\*\* AI Integration Engineer  
\*\*Context:\*\* We are adding speech and OCR capabilities to the backend to handle alternate data entry modes.  
\*\*Task Directive:\*\*  
1\. Add \`openai-whisper\` to \`requirements.txt\`.  
2\. In \`main.py\`, create \`POST /api/v1/analyze/speech\`. Load the \`whisper-tiny\` model locally. Accept an audio file, transcribe it, and pass the resulting text to the \`nutrition\_engine\`.  
3\. Create \`POST /api/v1/analyze/label\`. This endpoint is for OCR of Nutrition Facts panels.  
   \- Accept an image.  
   \- Construct a prompt for GPT-4o-mini to extract EXACT table values (Calories per 100g, Serving Size).  
   \- Enforce a strict JSON output schema. Do NOT apply depth estimation or density physics to this endpoint.  
4\. Log completion in \`python\_backend/docs/phase\_5\_multimodal.md\`.  
\*\*Acceptance Criteria:\*\* The backend can transcribe an audio file saying "I ate a banana" and return the deterministic macros for a banana via the API contract.  
