# **Dietin v2: Clinical-Grade Architecture Master Plan**

## **1\. Executive Summary**

**Objective:** Transition Dietin from a "Subjective AI Wrapper" (Client-Heavy) to a **"Deterministic Clinical Instrument"** (Hybrid Architecture).  
**Core Philosophy:** Eliminate LLM hallucination for nutritional math. Use Computer Vision for **Volume** (![][image1]), Database for **Density** (![][image2]), and LLM strictly for **Classification**.  
**Environment:** Prototype Phase on Local Apple Silicon (MacBook Air) with "Zero-CapEx" design, scalable to Cloud GPU (NVIDIA L4).

## **2\. Architectural Pivot**

We are moving from a **Serverless (React \+ Firebase)** stack to a **Hybrid** (React \+ Python **Middleware)** stack.

### **The New "Brain" (Backend)**

* **Runtime:** Python 3.10+ (FastAPI)  
* **Host:** Localhost (Port 8000\) for Dev \-\> Cloud GPU (RunPod/AWS) for Prod.  
* **Vector Store:** ChromaDB (Local persistence).  
* **AI Models:**  
  * **Vision:** SAM2 (Segmentation) \+ Depth Anything V2 (Metric Depth).  
  * **Logic:** GPT-4o (via OpenAI API) for Semantic Classification only.  
  * **Search:** all-MiniLM-L6-v2 \+ BM25 (Hybrid Search).

### **The New "Bridge" (Frontend)**

* **Role:** Reduced to UI rendering and Camera Capture.  
* **Change Scope:** strictly limited to src/lib/ service connectors and MealAnalysis.tsx data hooks. No visual redesigns.

## **3\. Core Logic & Constraints**

1. **The Async Race:** Vision Inference (Local) and Semantic Classification (GPT-4o) run in parallel.  
2. **Volume Injection:** GPT-4o **never** estimates calories. It receives volume data from the Vision Engine or outputs classification for the database to resolve.  
3. **The Liquid Fork:**  
   * *Solids:* Depth Anything V2 point cloud integration.  
   * *Liquids:* "Container Heuristic" (Rim detection \+ Fill level) or DKT (Future).  
4. **Hybrid Search:** Vector candidates (Top 50\) \-\> Keyword Filtering (e.g., must contain "Fried") \-\> Re-ranking.

## **4\. Implementation Phases**

### **Phase 1: Infrastructure & Foundation**

**Objective:** Establish the Python environment that bridges the React app to local AI models.

* **Task 1.1:** Initialize python\_backend/ root directory.  
* **Task 1.2:** Define requirements.txt.  
  * *Core:* fastapi, uvicorn, python-multipart.  
  * *AI:* torch, torchvision, transformers, pillow, numpy, openai-whisper (for speech).  
  * *Data:* chromadb, sentence-transformers, rank\_bm25, pandas.  
* **Task 1.3:** Create main.py scaffolding.  
  * Implement Device Detection (mps vs cpu).  
  * Setup CORS for localhost:3000.  
  * Create Health Check endpoint GET /health.  
* **Task 1.4:** Documentation Update.  
  * Create python\_backend/README.md with setup instructions.

### **Phase 2: The Nutrition Engine (Data Layer)**

**Objective:** Build the deterministic "Source of Truth" for food density and macros.

* **Task 2.1:** Create python\_backend/nutrition\_engine.py.  
* **Task 2.2:** Implement Data Seeding (seed\_database).  
  * Ingest a "Mock Clinical Dataset" (JSON) of \~50 complex items with **Yield Factors** and **Density Coefficients**.  
* **Task 2.3:** Implement Hybrid Search Logic.  
  * *Step 1:* Dense Retrieval (ChromaDB).  
  * *Step 2:* Sparse Filtering (BM25 for critical modifiers like "Fried", "Raw").  
  * *Step 3:* Re-ranking logic.  
* **Deliverable:** A script that takes "Fried Chicken" and returns { density: 0.65, calories: 240, method: 'hybrid' }.

### **Phase 3: The Vision Engine (The Eye)**

**Objective:** Implement "Zero-Cost" local inference for segmentation and depth.

* **Task 3.1:** Create python\_backend/vision\_engine.py.  
* **Task 3.2:** Implement Model Loading Strategy (Singleton Pattern).  
  * Load Depth-Anything-V2-Small (Local weights).  
  * Load SAM2 (or robust fallback like sam-vit-base).  
* **Task 3.3:** Implement process\_image(image\_bytes).  
  * Run Segmentation.  
  * Run Depth Estimation.  
  * **Crucial:** Implement the "Volume Calculation" math (converting pixel depth to ![][image1] using camera intrinsics).  
* **Task 3.4:** Implement the "Liquid Router" stub (detect if label is 'soup' \-\> apply flat-surface logic).

### **Phase 4: The Orchestrator (The Brain)**

**Objective:** Wire the Eye (Vision) and Data (Nutrition) together via FastAPI.

* **Task 4.1:** Implement POST /analyze endpoint in main.py.  
* **Task 4.2:** Implement the **Async Parallel Flow**.  
  * Thread 1: Call vision\_engine.process\_image.  
  * Thread 2: Call OpenAI API (GPT-4o) for classification & preparation\_state.  
* **Task 4.3:** Merge Logic.  
  * Combine volume\_cm3 (from Thread 1\) with density\_g\_cm3 (from Thread 2 \-\> Nutrition Engine Lookup).  
  * Calculate final Macros: Mass \= Volume \* Density.  
* **Task 4.4:** Return structured JSON matching the frontend's expected schema.

### **Phase 5: Frontend Integration (The Bridge)**

**Objective:** Connect the existing React UI to the new Local Backend.

* **Task 5.1:** Create src/lib/ai\_bridge.ts.  
  * Define the interface for the Local Backend.  
  * Implement error handling (Fallback to Cloud if Local is offline?).  
* **Task 5.2:** Refactor src/lib/gemini.ts.  
  * Deprecate direct Gemini logic for the "Analyze" feature.  
  * Redirect calls to ai\_bridge.ts.  
* **Task 5.3:** Update MealAnalysis.tsx.  
  * Add a "Server Status" indicator (Green/Red dot) to show if Python backend is connected.  
  * Handle the new "Confidence Score" and "Processing Steps" UI feedback.

### **Phase 6: Validation & Optimization**

**Objective:** Verify "Clinical-Grade" claims.

* **Task 6.1:** Run Latency Benchmarks on M4 chip.  
* **Task 6.2:** Accuracy Audit.  
  * Test against known volumes (e.g., a 330ml can of soda).  
  * Calibrate the pixel\_to\_cm conversion factor.  
* **Task 6.3:** Final architecture.md update.

### **Phase 7: Advanced Inputs & UI Binding**

**Objective:** Enable multi-modal inputs (Speech, Text, Barcode, Nutrition Labels) and bind them to the Backend.

* **Task 7.1: Text & Speech Pipeline (No Vision)**  
  * **Strategy:** While Whisper can run on-device (mobile edge), for this **Prototype Phase**, we will host the openai-whisper base model on the Python Middleware (Mac) to avoid complex native plugin compilation. Future phases will move this to the client.  
  * **Backend:** Create endpoint POST /analyze/speech.  
  * **Backend:** Create endpoint POST /analyze/text.  
  * **Logic:** Route transcribed/typed text directly to **Nutrition Engine (Phase 2\)** for Hybrid Search (skips Depth/SAM2).  
  * **Frontend:** Bind Microphone button to /speech endpoint.  
* **Task 7.2: Barcode Intelligence**  
  * **Backend:** Create POST /analyze/barcode.  
  * **Logic:** Integrate with OpenFoodFacts API (or cached DB).  
  * **Frontend:** Bind Scanner UI to backend. If found, populate macros directly. If not, fallback to Photo Mode.  
* **Task 7.3: Nutrition Label "Source of Truth" Mode**  
  * **Backend:** Create POST /analyze/label.  
  * **Logic:** Use AI strictly for OCR/Parsing. Extract Calories Per 100g or Calories Per Serving table.  
  * **Constraint:** **Skip** Density Database and Depth Estimation. Treat the table as Absolute Truth.  
  * **Response Schema:** { "mode": "label", "per\_100g": { ... }, "per\_serving": { ... }, "serving\_size": "string" }.  
* **Task 7.4: Interactive "Label" UI Flow**  
  * **Frontend:** Update MealAnalysis.tsx to handle the mode: "label" response.  
  * **Interaction:** When label data is detected, **pause** and ask user: *"Nutrition Table Detected. How much did you eat?"* (e.g., "1 serving", "half the bag", "200 grams").  
  * **Frontend Logic:** Calculate final macros based on User Input \+ Backend Table Data.

## **5\. Restrictions & Guidelines**

1. **Frontend Immutable:** Do not modify src/pages/, src/stores/, or src/components/ui/ unless absolutely necessary for data binding.  
2. **No Hallucinations:** Any prompt to GPT-4o MUST include system\_prompt: "Do not estimate calories. Output classification only."  
3. **Local-First:** All heavy dependencies (torch, transformers, whisper) must reside in python\_backend/venv, **never** in the Node package.json.  
4. **Type Safety:** All data exchanges between Python and TypeScript must be strictly typed (Pydantic models \<-\> TypeScript Interfaces).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACEAAAAYCAYAAAB0kZQKAAABdUlEQVR4Xu2VsStGURjGX2SSDMqAlDKIxWShpAiDTxYZKMwmySJJ/AOSoigZJYP/QMoik0k2oyKDsiiex3kvr/e71zXdm9xfPd1znuece9/vnHvuJ/LPOIQeoXOo0mWZsAYda3sSejNZZgxDZ9qel5yKsDxBE97MigZoE7r1QR7US07bcQqNmD6L6DH9MsagE2jWeF3QLlSj/VpoBVqAKqJBYAk6gJqMR/jQDW3XaT8WVsqwX/vb0BVULeFslzRfl/AgsqpeB3QDVUGN6jXrGML34Rna0azNZJ8MSghbjMf+pXwdrRn1eOYjuCL07oxH6C06LxVOundeq1657IS/1C/jXIzH1aDX6fwf6ZYwadQHDo65cB5XwBfBlfJeKvvyu0kcwy+f96Ltst6D81KZluQipvSa9L2n1xfjDWj7xQZpcGK78/iPN6TtaykvIu596DXesiScgiR4rF4l3IA6+h5/ZFvO25NQqIdbwXuM+6CgoOBP8g6/+FVF9cy4fQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAYCAYAAABa1LWYAAACU0lEQVR4Xu2WTYiNURzG/2hsJAsThZRYiJSFWFA3ISzQbDSLGWGnJsnXRpKGrBFRlFBKspilzTRNzWbSkJXslEQxGPkoX8/j/I8588x5rzfuvd2p+6une87v/55773k/znnNWjQNN5G3yCAyXWo1Y6mKOnIaueftTuRnUqsZq5ADKuvINmTA2z1Wp0m9VNFA3iG7VdaCDyoawDzkPPJMC7VgP7JJZQOZa3W4/X6oaAB9yPakz0mtT/pZ1iF3kIoWMjxWIexC7iP7ErcauYrM8v5s5CRyGJkWDwLHkRvIwsQRTuKMt+d4vyrvkYvevoZ8t+JBF5AFKh2eSY7b6P1LyEOkzcLestPrvRb+ODnlbgXyFJlh4fvpFvkxhM/TR+SK15YltUl8QV6I46Db4iJFk91iobY4cewP2/hSvNcd95wIrxjd88QRuqPiSnHQwuD2xPFMxTOn8LLfUulwzGtxS/yTtxnhldCTwkVHXfwPK8WXggP1C49lXIRnnD+orLUwZocWBB4zJI5XSH+PV1JdaTjwk7gx9zmK/HUrrqXwGL4ZqIu3Z+reiCsNB/dnXHzHSllj4Srm6LbiSXX5Z9H7Gl0l4zZ7+3NaKANXofSHuP+wn7uXX6kQOG65OL5Rb/X2E5s8qdzztCFxJ+wvq1wRR5BR5JxVf564nFaDy/A3C+OZuxPLv2tx24hw++DEFd56/I4OLfwL/OO5Se2xsGQ3Pf3IV3Gc0CFxhGd5SsAJPPJ23AB1uY2MqGhW+A52GXmAnEVmTiz/gRvnfJVTnYqKFi3+j1/4G4wCaVE/8QAAAABJRU5ErkJggg==>