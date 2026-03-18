import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("VITE_GOOGLE_AI_KEY")
genai.configure(api_key=api_key)

def test_models():
    models = ["gemini-1.5-flash", "gemini-2.0-flash"]
    for model_name in models:
        print(f"Testing model: {model_name}")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say hi")
            print(f"  Success: {response.text}")
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    test_models()
