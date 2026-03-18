"""
DIETIN V2 — Python FastAPI Backend
===================================
Phase 1: Core scaffolding (FastAPI + CORS + /health)
Phase 4: API Contract & Orchestration
"""

import asyncio
import os
import tempfile
import base64
import json
import whisper
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import torch
from pydantic import BaseModel
from PIL import UnidentifiedImageError

from vision_engine import VisionPipeline
from nutrition_engine import HybridSearchEngine
from liquid_logic import apply_liquid_fallback

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

whisper_model = whisper.load_model("tiny", device=DEVICE if DEVICE != "mps" else "cpu")
genai.configure(api_key=os.getenv("VITE_GOOGLE_AI_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "sk-mock"))

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

# VRAM Protection Semaphore to ensure thread-safe Apple Silicon GPU inference
vision_semaphore = asyncio.Semaphore(1)

# Helper function for Payload Cap Bounds
async def validate_file_size(file: UploadFile, max_size_mb: int):
    # Determine the payload size limit in bytes
    max_size_bytes = max_size_mb * 1024 * 1024
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    
    if file_size > max_size_bytes:
        raise HTTPException(status_code=413, detail="Payload Too Large")
        
    file.file.seek(0)
    
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

def clean_json(text: str) -> str:
    """Removes markdown backticks from a JSON string."""
    import re
    if "```" in text:
        text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL).strip()
    return text

async def gemini_classification(image_bytes: bytes) -> dict:
    if not image_bytes:
        return {"foodName": "Unknown", "prep": "raw", "is_liquid": False}
        
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config={
            "response_mime_type": "application/json",
            "max_output_tokens": 200,
        },
        system_instruction="You are a food classifier. You MUST return a JSON object exactly matching this schema: {\"foodName\": \"string\", \"prep\": \"string\", \"is_liquid\": boolean}."
    )
    
    # Gemini takes raw bytes via blob logic
    blob = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }
    
    # Run async formulation via to_thread because python genai SDK doesn't natively expose robust async generate_content yet 
    # (or we can just use the sync method inside a thread)
    def call_gemini():
        return model.generate_content([
            "Classify this food.",
            blob
        ])
        
    try:
        response = await asyncio.to_thread(call_gemini)
        cleaned_text = clean_json(response.text)
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Gemini Classification Error: {e}")
        return {"foodName": "Unknown", "prep": "raw", "is_liquid": False}

@app.post("/api/v1/analyze/image", response_model=AnalysisResponse)
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    full_image: Optional[UploadFile] = File(None),
    context_image: Optional[UploadFile] = File(None),
    crops: Optional[List[UploadFile]] = File(None)
):
    upload_file = file if file else full_image
    
    if not upload_file and not (context_image and crops):
        raise HTTPException(status_code=400, detail="No valid image or crops provided")

    masks = None
    if upload_file:
        try:
            await validate_file_size(upload_file, 20)
            image_bytes = await upload_file.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
                
            # Run vision model, MobileSAM, and GPT concurrently
            async def run_vision_inference():
                async with vision_semaphore:
                    sam_task = asyncio.to_thread(vision_pipeline.run_mobile_sam, image_bytes)
                    vision_task = asyncio.to_thread(vision_pipeline.estimate_volume, image_bytes)
                    return await asyncio.gather(sam_task, vision_task)
            
            gpt_task = gemini_classification(image_bytes)
            (masks, volume_cm3), gpt_result = await asyncio.gather(run_vision_inference(), gpt_task)
        except HTTPException:
            raise
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image format.")
        except Exception as e:
            # Catch other model failures gracefully
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")
            
    elif context_image and crops:
        try:
            await validate_file_size(context_image, 20)
            
            # MODE 2 (Future): Phone ran MobileSAM, backend skips segmentation
            masks = crops
            volume_cm3 = 0.0  # Simulated volume for pre-cropped inputs
            
            ctx_bytes = await context_image.read() if context_image else b""
            gpt_result = await gemini_classification(ctx_bytes)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mode 2 processing failed: {str(e)}")

    # Query database
    db_record = await asyncio.to_thread(
        search_engine.search,
        query=gpt_result["foodName"],
        prep_filter=gpt_result.get("prep", "")
    )
    
    warnings = []
    
    if not upload_file and context_image and crops:
        warnings.append("Mode 2 (Crops Only) active. Deterministic metric volume is impossible without a full-frame depth map.")

    if not db_record:
        warnings.append("Food not found in database. Relying on fallback macros.")
        food_name = gpt_result["foodName"]
        confidence_score = 0.5
        calories = 0
        macros = Macros(protein=0, carbs=0, fat=0)
    else:
        confidence_score = 0.95
        
        # Liquid Fork Logic
        if gpt_result.get("is_liquid"):
            volume_cm3, confidence_score = apply_liquid_fallback(volume_cm3, confidence_score)
            warnings.append("Liquid detected. Depth was down-weighted and a conservative container-fill fallback was applied.")
            
        # (Confidence score is now zeroed globally for Mode 2 before returning)

        # Calculate mass deterministically
        mass_g = volume_cm3 * db_record.density_g_cm3 * db_record.yield_factor
        
        # Simple macro calculation (mocked based on density mapping)
        food_name = db_record.name
        # Since we just have density, let's mock the macros based on mass.
        calories = int(mass_g * 2.5)
        macros = Macros(protein=int(mass_g * 0.2), carbs=int(mass_g * 0.3), fat=int(mass_g * 0.1))

    if not upload_file and context_image and crops:
        confidence_score = 0.0

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

@app.post("/api/v1/analyze/speech", response_model=AnalysisResponse)
async def analyze_speech(file: UploadFile = File(...)):
    await validate_file_size(file, 50)
    
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            audio_bytes = await file.read()
            await asyncio.to_thread(temp_file.write, audio_bytes)
            temp_file_path = temp_file.name
        
        # Run transcription in thread to avoid blocking main event loop
        result = await asyncio.to_thread(whisper_model.transcribe, temp_file_path)
        transcription = result["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {str(e)}")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    if not transcription:
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")

    # Route text to search engine
    db_record = await asyncio.to_thread(
        search_engine.search,
        query=transcription,
        prep_filter=""
    )
    
    warnings = ["Speech input: Volume scaled to standard 150.0 cm3 assumption."]
    volume_cm3 = 150.0
    
    if not db_record:
        food_name = transcription
        confidence_score = 0.5
        calories = 0
        macros = Macros(protein=0, carbs=0, fat=0)
    else:
        mass_g = volume_cm3 * db_record.density_g_cm3 * db_record.yield_factor
        food_name = db_record.name
        confidence_score = 0.85
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

class ProxyGenerateRequest(BaseModel):
    prompt: str
    system_instruction: Optional[str] = None

@app.post("/api/v1/proxy/generate")
async def proxy_generate(req: ProxyGenerateRequest):
    model_kwargs: Dict[str, Any] = {
        "model_name": "gemini-2.0-flash",
        "generation_config": {"response_mime_type": "application/json"}
    }
    if req.system_instruction:
        model_kwargs["system_instruction"] = req.system_instruction

    model = genai.GenerativeModel(**model_kwargs)
    try:
        response = await asyncio.to_thread(model.generate_content, req.prompt)
        return {"text": response.text}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze/label", response_model=AnalysisResponse)
async def analyze_label(file: UploadFile = File(...)):
    try:
        await validate_file_size(file, 20)
        
        image_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": 300,
            },
            system_instruction="You are a specialized OCR parser for Nutrition Facts labels. Your only task is to extract the food name, calories, and macros (protein, carbs, fat) from the label. You MUST return a JSON object with the exact following schema: {\"foodName\": \"string\", \"calories\": integer, \"macros\": {\"protein\": integer, \"carbs\": integer, \"fat\": integer}}."
        )
        
        blob = {
            "mime_type": mime_type,
            "data": image_bytes
        }
        
        def call_gemini_label():
            return model.generate_content([
                "Parse this nutrition label.",
                blob
            ])
            
        response = await asyncio.to_thread(call_gemini_label)
        cleaned_text = clean_json(response.text)
        parsed_data = json.loads(cleaned_text)
        
        food_name = parsed_data.get("foodName", "Nutrition Label")
        calories = int(parsed_data.get("calories", 0))
        macros_data = parsed_data.get("macros", {})
        macros = Macros(
            protein=int(macros_data.get("protein", 0)),
            carbs=int(macros_data.get("carbs", 0)),
            fat=int(macros_data.get("fat", 0))
        )
        
        return AnalysisResponse(
            success=True,
            data=AnalysisData(
                foodName=food_name,
                calories=max(0, calories),
                macros=macros,
                healthScore=85.0,
                confidenceScore=0.99,
                warnings=["Parsed directly from Nutrition Label OCR."]
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Label OCR failed: {str(e)}")
