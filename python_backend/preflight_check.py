import os
import asyncio
import base64
from dotenv import load_dotenv

# Load .env from root
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(root_dir, ".env")
load_dotenv(env_path)

# Bridge VITE_GOOGLE_AI_KEY to GEMINI_API_KEY
if not os.getenv("GEMINI_API_KEY") and os.getenv("VITE_GOOGLE_AI_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["VITE_GOOGLE_AI_KEY"]

from llm_router import LLMRouter
from logger import logger

async def test_and_inform(router):
    print(f"--- AI Preflight Check: Testing \033[96m{router.provider.upper()}\033[0m ---")
    
    # We use a real JPEG from the root directory to avoid format mismatch errors
    # (The previous 1x1 GIF caused Anthropic 400 ERROR)
    apple_path = os.path.join(root_dir, "test_apple.jpg")
    try:
        with open(apple_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Preflight: Could not read {apple_path}: {e}")
        # dummy fallback if file is missing
        base64_image = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

    try:
        # Use simple get_classification instead of internal test_connection to fully simulate frontend path
        result = await asyncio.wait_for(router.get_classification(base64_image), timeout=25.0)
        if result.get("foodName") != "Unknown":
             print(f"\033[92mSUCCESS: {router.provider.upper()} is active and confirmed working.\033[0m")
             return True
        else:
             print(f"\033[93mWARNING: {router.provider.upper()} returned an empty/unknown result.\033[0m")
             return False
    except Exception as e:
        print(f"\033[91mFAILURE: {router.provider.upper()} failed with error: {str(e)}\033[0m")
        return False

async def run_preflight():
    # Attempt 1: Default provider (usually Gemini)
    router = LLMRouter()
    if router.provider == "mock":
        print("\033[91mCRITICAL ERROR: No AI providers configured in .env!\033[0m")
        return False

    if await test_and_inform(router):
        return True

    # Attempt 2: Fallback to Anthropic if Gemini (current provider) was exhausted/failed
    if router.provider == "gemini" and os.getenv("ANTHROPIC_API_KEY"):
         print("\033[94mINFO: Primary provider failed. Attempting Anthropic fallback...\033[0m")
         os.environ["GEMINI_API_KEY"] = "" # Hide gemini to force Anthropic selection
         router_fallback = LLMRouter()
         if await test_and_inform(router_fallback):
             # Success! Update the current environment so the backend process uses this one.
             # Note: This is just for this preflight script, we need the backend to also respect this or we update LLMRouter.py.
             return True

    print("\033[91mABORTING: No working AI provider found. Please check your API keys and quotas.\033[0m")
    return False

if __name__ == "__main__":
    if asyncio.run(run_preflight()):
        print("\033[1;32mPREFLIGHT PASSED. Initiating Backend and Frontend...\033[0m")
        exit(0)
    else:
        exit(1)
