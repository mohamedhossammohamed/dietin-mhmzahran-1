"""
DIETIN V2 — Python FastAPI Backend
===================================
Phase 1: Core scaffolding (FastAPI + CORS + /health)
Phase 4: API Contract & Orchestration
"""

import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from pydantic import BaseModel

from vision_engine import VisionPipeline
from nutrition_engine import HybridSearchEngine

# ---------------------------------------------------------------------------
# Phase 1: Core App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Dietin V2 Backend", version="2.0.0")

# CORS — strictly the two origins specified in the master plan
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = _detect_device()

# ---------------------------------------------------------------------------
# Phase 1: Health Endpoint
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": DEVICE}

# ---------------------------------------------------------------------------
# Phase 4: API Contract & Orchestration
# ---------------------------------------------------------------------------

vision_pipeline = VisionPipeline()
search_engine = HybridSearchEngine()

class Macros(BaseModel):
    protein: int
    carbs: int
    fat: int

class AnalysisData(BaseModel):
    foodName: str
    calories: int
    macros: Macros
    healthScore: float
    confidenceScore: float
    warnings: list

class AnalysisResponse(BaseModel):
    success: bool
    data: AnalysisData

async def mock_gpt4o_classification(image_bytes: bytes):
    await asyncio.sleep(1) # simulate network latency
    return {"foodName": "Chicken", "prep": "fried", "is_liquid": False}

@app.post("/api/v1/analyze/image", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(None),
    full_image: UploadFile = File(None),
    context_image: UploadFile = File(None),
    crops: list[UploadFile] = File(None)
):
    upload_file = file if file else full_image
    
    if not upload_file and not (context_image and crops):
        raise HTTPException(status_code=400, detail="No valid image or crops provided")

    masks = None
    if upload_file:
        image_bytes = await upload_file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        # MODE 1 (Current): Backend runs MobileSAM
        masks = vision_pipeline.run_mobile_sam(image_bytes)
        
        try:
            # Run vision model and mock GPT concurrently
            vision_task = asyncio.to_thread(vision_pipeline.estimate_volume, image_bytes)
            gpt_task = mock_gpt4o_classification(image_bytes)
            volume_cm3, gpt_result = await asyncio.gather(vision_task, gpt_task)
        except Exception as e:
            # Catch PIL errors or other model failures gracefully
            raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
            
    elif context_image and crops:
        # MODE 2 (Future): Phone ran MobileSAM, backend skips segmentation
        masks = crops
        volume_cm3 = 250.0  # Simulated volume for pre-cropped inputs
        gpt_result = await mock_gpt4o_classification(b"")

    # Query database
    db_record = search_engine.search(query=gpt_result["foodName"], prep_filter=gpt_result.get("prep", ""))
    
    warnings = []
    
    if not db_record:
        warnings.append("Food not found in database. Relying on fallback macros.")
        food_name = gpt_result["foodName"]
        confidence_score = 0.5
        calories = 0
        macros = Macros(protein=0, carbs=0, fat=0)
    else:
        # Calculate mass deterministically
        mass_g = volume_cm3 * db_record.density_g_cm3 * db_record.yield_factor
        
        # Simple macro calculation (mocked based on density mapping)
        food_name = db_record.name
        confidence_score = 0.95
        # Since we just have density, let's mock the macros based on mass.
        calories = int(mass_g * 2.5)
        macros = Macros(protein=int(mass_g * 0.2), carbs=int(mass_g * 0.3), fat=int(mass_g * 0.1))

    return AnalysisResponse(
        success=True,
        data=AnalysisData(
            foodName=food_name,
            calories=max(0, calories),
            macros=macros,
            healthScore=85.0,
            confidenceScore=confidence_score,
            warnings=warnings
        )
    )
