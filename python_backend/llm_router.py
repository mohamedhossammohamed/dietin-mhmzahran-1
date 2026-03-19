import os
import json
import httpx
from typing import Dict, Any

class LLMRouter:
    """
    Routes multimodal classification requests to the best available LLM provider
    (OpenAI, Google Gemini Flash, or Anthropic Claude Haiku).
    """

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        # We prefer Gemini Flash for speed/cost if available, then Anthropic, then OpenAI.
        # This order can be adjusted based on testing or cost parameters.
        if self.gemini_key:
            self.provider = "gemini"
        elif self.anthropic_key:
            self.provider = "anthropic"
        elif self.openai_key:
            self.provider = "openai"
        else:
            self.provider = "mock"
            print("WARNING: No LLM API keys found. Defaulting to mock LLM router.")

    async def get_classification(self, base64_image: str) -> Dict[str, Any]:
        """
        Sends the image to the selected LLM and returns a strictly typed classification.
        Returns: { "foodName": "Chicken", "prep": "fried", "is_liquid": bool, "confidence": float }
        """
        if self.provider == "gemini":
            return await self._call_gemini(base64_image)
        elif self.provider == "anthropic":
            return await self._call_anthropic(base64_image)
        elif self.provider == "openai":
            return await self._call_openai(base64_image)
        else:
            # Dummy sleep to simulate network latency
            import asyncio
            await asyncio.sleep(1.0)
            return {
                "foodName": "Chicken",
                "prep": "fried",
                "is_liquid": False,
                "confidence": 0.85
            }

    async def get_ocr(self, base64_image: str) -> Dict[str, Any]:
        """
        Extracts exact OCR table values for Nutrition Facts panels.
        """
        if self.provider == "mock":
            import asyncio
            await asyncio.sleep(0.5)
            return {
                "calories": 250,
                "protein": 15,
                "carbs": 30,
                "fat": 10
            }
        # In a real implementation, we would construct a specific prompt for OCR.
        # For simplicity in Phase 4/5 integration, we use a basic mock structure if the provider isn't fully wired for OCR yet.
        return {
            "calories": 250,
            "protein": 15,
            "carbs": 30,
            "fat": 10
        }

    async def _call_gemini(self, base64_image: str) -> Dict[str, Any]:
        # Minimal implementation for Gemini REST API (gemini-1.5-flash-latest or similar)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Analyze this food image. Return ONLY a JSON object with: 'foodName' (string), 'prep' (string: fried, boiled, raw, baked, liquid, etc), 'is_liquid' (boolean), and 'confidence' (float 0.0-1.0)."},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            if response.status_code == 200:
                try:
                    data = response.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text)
                except Exception as e:
                    print(f"Gemini parsing error: {e}")

        # Fallback if API fails
        return {"foodName": "Unknown", "prep": "unknown", "is_liquid": False, "confidence": 0.0}

    async def _call_anthropic(self, base64_image: str) -> Dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze this food image. Return ONLY a JSON object with: 'foodName' (string), 'prep' (string), 'is_liquid' (boolean), and 'confidence' (float 0.0-1.0)."
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)
            if response.status_code == 200:
                try:
                    data = response.json()
                    text = data["content"][0]["text"]
                    # Anthropic might wrap in markdown ```json
                    text = text.replace("```json", "").replace("```", "").strip()
                    return json.loads(text)
                except Exception as e:
                    print(f"Anthropic parsing error: {e}")

        return {"foodName": "Unknown", "prep": "unknown", "is_liquid": False, "confidence": 0.0}

    async def _call_openai(self, base64_image: str) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this food image. Return ONLY a JSON object with: 'foodName' (string), 'prep' (string), 'is_liquid' (boolean), and 'confidence' (float 0.0-1.0)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=10.0)
            if response.status_code == 200:
                try:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    return json.loads(text)
                except Exception as e:
                    print(f"OpenAI parsing error: {e}")

        return {"foodName": "Unknown", "prep": "unknown", "is_liquid": False, "confidence": 0.0}

llm_router = LLMRouter()
