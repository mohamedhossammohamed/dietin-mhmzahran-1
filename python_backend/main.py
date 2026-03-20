from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import torch
import time
from dotenv import load_dotenv
# Load from root directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
# Also try local in case of different deployment
load_dotenv()
from logger import logger

app = FastAPI(title="DIETIN V2 Backend", version="1.0.0")

# Request/Response Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    logger.info(f"Incoming {method} request to {path}")
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = "{0:.2f}".format(process_time)
        
        status_code = response.status_code
        logger.info(f"Completed {method} {path} - Status: {status_code} (Duration: {formatted_process_time}ms)")
        
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        formatted_process_time = "{0:.2f}".format(process_time)
        logger.exception(f"Exception during {method} {path} - Error: {str(e)} (Duration: {formatted_process_time}ms)")
        raise

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import base64
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from fastapi import File, UploadFile, HTTPException

from vision_engine import vision_engine
from nutrition_engine import engine as nutrition_engine
from llm_router import llm_router

# Response Schema Expected by React Frontend (Gemini proxy format)
class Macros(BaseModel):
    protein: float
    carbs: float
    fat: float

class NutritionData(BaseModel):
    foodName: str
    calories: int
    macros: Macros
    healthScore: float
    confidenceScore: float
    warnings: List[str]

class AnalysisResponse(BaseModel):
    success: bool
    data: NutritionData


@app.get("/health")
async def health_check():
    # Determine the available device for ML models
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return {
        "status": "healthy",
        "device": device
    }

@app.post("/api/v1/analyze/speech")
async def analyze_speech(audio_file: UploadFile = File(...)):
    """
    Phase 5: Audio Input Endpoint
    Transcribes audio using Whisper-tiny, then queries the Nutrition Engine.
    """
    import whisper
    import tempfile

    # Check if ffmpeg is installed (required by whisper) using a cross-platform check
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="FFmpeg is not installed on the host machine. Required for audio processing.")

    tmp_path = None
    try:
        # Save uploaded file to a temporary location
        suffix = os.path.splitext(audio_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load whisper-tiny (lazy load to save memory)
        model = whisper.load_model("tiny")
        result = model.transcribe(tmp_path)
        transcription = result["text"].strip()

        # Query nutrition engine directly using transcribed text
        # (Assuming the user just speaks the food name)
        db_record = nutrition_engine.search(query=transcription)

        if not db_record:
            return {
                "success": False,
                "message": f"Transcribed '{transcription}', but could not map it to a food item.",
                "data": None
            }

        return {
            "success": True,
            "transcription": transcription,
            "data": {
                "foodName": db_record.name,
                "calories": db_record.calories_per_100g, # returning per 100g base for speech as volume isn't known
                "macros": {
                    "protein": db_record.protein_per_100g,
                    "carbs": db_record.carbs_per_100g,
                    "fat": db_record.fat_per_100g
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/v1/analyze/label")
async def analyze_label(label_image: UploadFile = File(...)):
    """
    Phase 5: OCR Override Endpoint
    Skips Vision/Physics engine. Extracts strict macros from a nutrition label via LLM.
    """
    try:
        image_bytes = await label_image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Use LLM Router to extract OCR data
        ocr_data = await llm_router.get_ocr(base64_image)

        return {
            "success": True,
            "data": {
                "foodName": "Nutrition Label",
                "calories": ocr_data.get("calories", 0),
                "macros": {
                    "protein": ocr_data.get("protein", 0.0),
                    "carbs": ocr_data.get("carbs", 0.0),
                    "fat": ocr_data.get("fat", 0.0)
                },
                "healthScore": 85.0,
                "confidenceScore": 95.0,
                "warnings": ["Macros extracted directly from label (Per 100g)."]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ProxyRequest(BaseModel):
    prompt: str
    system_instruction: Optional[str] = None

@app.post("/api/v1/proxy/generate")
async def proxy_generate(request: ProxyRequest):
    """
    Phase 4: Text Proxy Endpoint
    Proxies text-only generation requests to the LLM router to secure API keys.
    """
    try:
        # MHMZ: Use the centralized LLM Router for text proxying, ensuring correct key and model usage.
        text_response = await llm_router.get_text(
            prompt=request.prompt, 
            system_instruction=request.system_instruction
        )
        return {"text": text_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AnalyzerTextRequest(BaseModel):
    text: str

@app.post("/api/v1/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: AnalyzerTextRequest):
    """
    Phase 4+: Text Analysis Endpoint
    Executes LLM Classification (API) to extract food details, 
    then uses deterministic physics (Nutrition Engine) for exact macros.
    """
    start_time = time.time()
    logger.info("Main API: Received text analysis request")

    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text description is required")

    try:
        # LLM extracts { foodName, prep, mass_g }
        llm_classification = await llm_router.get_text_classification(request.text)
        logger.info(f"Main API: Text classification extracted: {llm_classification}")
    except Exception as e:
        logger.exception("Main API FAILURE: LLM router failed for text analysis")
        raise HTTPException(status_code=500, detail=f"LLM extraction error: {str(e)}")

    food_name = llm_classification.get("foodName", request.text)
    prep = llm_classification.get("prep", "")
    mass_g = float(llm_classification.get("mass_g", 100.0))
    warnings = []

    # Semantic Database Lookup
    try:
        db_record = nutrition_engine.search(query=food_name, prep_filter=prep)
    except Exception as e:
        logger.exception("Main API FAILURE: Nutrition database search failed")
        db_record = None

    if db_record:
        logger.info(f"Main API: Database match found: {db_record.name}")
        multiplier = mass_g / 100.0
        final_calories = int(db_record.calories_per_100g * multiplier)
        final_protein = round(db_record.protein_per_100g * multiplier, 1)
        final_carbs = round(db_record.carbs_per_100g * multiplier, 1)
        final_fat = round(db_record.fat_per_100g * multiplier, 1)
        resolved_food_name = db_record.name
        
        # Check liquid heuristic
        if db_record.is_liquid:
            warnings.append("Liquid detected. Macros based on entered/estimated mass.")
    else:
        logger.warning(f"Main API: No database match for '{food_name}'. Using generic fallback.")
        warnings.append("Semantic mapping failed: Falling back to generic constants.")
        multiplier = mass_g / 100.0
        final_calories = int(150 * multiplier)
        final_protein = round(5.0 * multiplier, 1)
        final_carbs = round(20.0 * multiplier, 1)
        final_fat = round(5.0 * multiplier, 1)
        resolved_food_name = food_name

    duration = (time.time() - start_time) * 1000
    logger.info(f"Main API: Text analysis complete for '{resolved_food_name}' in {duration:.2f}ms")

    return AnalysisResponse(
        success=True,
        data=NutritionData(
            foodName=resolved_food_name,
            calories=final_calories,
            macros=Macros(protein=final_protein, carbs=final_carbs, fat=final_fat),
            healthScore=85.0,
            confidenceScore=0.9, # High confidence as it's directly typed
            warnings=warnings
        )
    )

@app.post("/api/v1/analyze/image", response_model=AnalysisResponse)

async def analyze_image(
    full_image: UploadFile = File(None),
    context_image: UploadFile = File(None),
    crops: List[UploadFile] = File(None)
):
    """
    Phase 4: Orchestration Endpoint
    Executes Vision Engine (local) and LLM Classification (API) concurrently.
    Merges deterministic volume and USDA density to calculate exact macros.
    """
    start_time = time.time()
    logger.info("Main API: Received image analysis request")

    if not full_image and not crops:
        logger.warning("Main API: No image provided in request")
        raise HTTPException(status_code=400, detail="No image provided")

    # Read image bytes
    if full_image:
        try:
            image_bytes = await full_image.read()
            logger.debug(f"Main API: Read {len(image_bytes)} bytes from upload")
        except Exception as e:
            logger.exception("Main API FAILURE: Failed to read uploaded file")
            raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")

        # Convert to base64 for LLM
        try:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.exception("Main API FAILURE: Failed to encode image to base64")
            raise HTTPException(status_code=500, detail="Failed to process image data")

        # 1. Pattern A: The Async Race (asyncio.gather)
        logger.info("Main API: Dispatching Vision Engine and LLM Router tasks...")
        vision_task = asyncio.to_thread(vision_engine.estimate_volume, image_bytes)
        gpt_task = llm_router.get_classification(base64_image)

        try:
            volume_cm3, llm_classification = await asyncio.gather(vision_task, gpt_task)
            logger.info(f"Main API: Async tasks completed. Volume: {volume_cm3:.2f}cm3, Food: {llm_classification.get('foodName')}")
        except Exception as e:
            logger.exception("Main API FAILURE: Pipeline async tasks failed")
            raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")

        food_name = llm_classification.get("foodName", "Unknown Food")
        prep = llm_classification.get("prep", "")
        is_liquid = llm_classification.get("is_liquid", False)
        class_confidence = llm_classification.get("confidence", 0.5)

        # 2. Semantic Database Lookup (Deterministic Physics)
        try:
            logger.debug(f"Main API: Searching nutrition database for '{food_name}' (prep: {prep})")
            db_record = nutrition_engine.search(query=food_name, prep_filter=prep)
        except Exception as e:
            logger.exception(f"Main API FAILURE: Nutrition database search failed for '{food_name}'")
            db_record = None

        warnings = []

        if db_record:
            logger.info(f"Main API: Database match found: {db_record.name}")
            density = db_record.density_g_cm3
            yield_factor = db_record.yield_factor

            # Pattern B: The Liquid Fork (Soup Problem)
            if is_liquid or db_record.is_liquid:
                logger.info("Main API: Liquid detected. Applying container fill heuristic.")
                warnings.append("Liquid detected: Container Fill Heuristic applied (Depth map overridden).")
                volume_cm3 = 250.0  # 1 cup default for liquids
                mass_g = volume_cm3 * density
            else:
                mass_g = volume_cm3 * density * yield_factor
                logger.debug(f"Main API: Calculated mass: {mass_g:.2f}g (Density: {density}, Yield: {yield_factor})")

            # Calculate final macros
            multiplier = mass_g / 100.0
            final_calories = int(db_record.calories_per_100g * multiplier)
            final_protein = round(db_record.protein_per_100g * multiplier, 1)
            final_carbs = round(db_record.carbs_per_100g * multiplier, 1)
            final_fat = round(db_record.fat_per_100g * multiplier, 1)
            resolved_food_name = db_record.name
        else:
            logger.warning(f"Main API: No database match for '{food_name}'. Using generic fallback.")
            warnings.append("Semantic mapping failed: Falling back to generic constants.")
            mass_g = volume_cm3 * 1.0
            multiplier = mass_g / 100.0
            final_calories = int(150 * multiplier)
            final_protein = round(5.0 * multiplier, 1)
            final_carbs = round(20.0 * multiplier, 1)
            final_fat = round(5.0 * multiplier, 1)
            resolved_food_name = food_name

        if class_confidence < 0.7:
            warnings.append(f"Low confidence ({class_confidence * 100:.1f}%): Values may be inaccurate.")

        duration = (time.time() - start_time) * 1000
        logger.info(f"Main API: Analysis complete for '{resolved_food_name}' in {duration:.2f}ms")

        return AnalysisResponse(
            success=True,
            data=NutritionData(
                foodName=resolved_food_name,
                calories=final_calories,
                macros=Macros(protein=final_protein, carbs=final_carbs, fat=final_fat),
                healthScore=85.0,
                confidenceScore=class_confidence,
                warnings=warnings
            )
        )

    else:
        logger.warning("Main API: Crops-only analysis requested but not implemented")
        raise HTTPException(status_code=400, detail="Crop-based analysis is not yet supported. Please provide a full_image.")

