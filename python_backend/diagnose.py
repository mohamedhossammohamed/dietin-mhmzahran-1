import time
print("Starting diagnostics...")
import torch
print(f"Torch imported. Version: {torch.__version__}")
import whisper
print("Whisper imported.")
start_time = time.time()
print("Loading Whisper 'tiny' model...")
model = whisper.load_model("tiny")
print(f"Whisper model loaded in {time.time() - start_time:.2f} seconds.")
import google.generativeai as genai
print("Google Generative AI imported.")
from vision_engine import VisionPipeline
print("VisionPipeline imported.")
vp = VisionPipeline()
print("VisionPipeline initialized.")
from nutrition_engine import HybridSearchEngine
print("HybridSearchEngine imported.")
hse = HybridSearchEngine()
print("HybridSearchEngine initialized.")
print("All systems go.")
