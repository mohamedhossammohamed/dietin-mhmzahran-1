import os
import json
import httpx
from typing import Dict, Any
import re
from logger import logger

_OCR_EMPTY_RESULT: Dict[str, Any] = {"calories": 0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}

class LLMRouter:
    """
    Routes multimodal classification requests to the best available LLM provider
    (OpenAI, Google Gemini Flash, or Anthropic Claude Haiku).
    """

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("VITE_GOOGLE_AI_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        # We prefer Gemini Flash for speed/cost if available, then Anthropic, then OpenAI.
        # This order can be adjusted based on testing or cost parameters.
        if self.gemini_key:
            self.provider = "gemini"
            # MHMZ: Updated model to gemini-3.1-flash-lite-preview as requested.
            self.gemini_model = "gemini-3.1-flash-lite-preview"
            logger.info(f"LLM Router: Using Google Gemini ({self.gemini_model}) as provider.")
        elif self.anthropic_key:
            self.provider = "anthropic"
            logger.info("LLM Router: Using Anthropic Claude as provider.")
        elif self.openai_key:
            self.provider = "openai"
            logger.info("LLM Router: Using OpenAI as provider.")
        else:
            self.provider = "mock"
            logger.warning("No LLM API keys found. Defaulting to mock LLM router.")

    async def test_connection(self) -> bool:
        """
        Tests if the current AI provider is responsive.
        """
        if self.provider == "mock":
            return False
            
        minimal_image_b64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        try:
            result = await self.get_classification(minimal_image_b64)
            return result.get("foodName") != "Unknown"
        except Exception as e:
            logger.error(f"LLM Router: Connection test failed for {self.provider}: {e}")
            return False

    async def get_classification(self, base64_image: str) -> Dict[str, Any]:
        """
        Sends the image to the selected LLM and returns a strictly typed classification.
        Returns: { "foodName": "Chicken", "prep": "fried", "is_liquid": bool, "confidence": float }
        """
        logger.info(f"LLM Router: Requesting classification using {self.provider} provider.")
        if self.provider == "gemini":
            return await self._call_gemini(base64_image)
        elif self.provider == "anthropic":
            return await self._call_anthropic(base64_image)
        elif self.provider == "openai":
            return await self._call_openai(base64_image)
        else:
            logger.error("LLM Router: No valid LLM provider configured for classification.")
            raise ValueError("No valid LLM API key configured.")

    async def get_text(self, prompt: str, system_instruction: str = None) -> str:
        """
        Generic text proxy to bypass library limitations and secure keys.
        """
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=15.0)
                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    logger.error("="*50)
                    logger.error(f"GEMINI TEXT API ERROR: {response.status_code}")
                    logger.error(f"Response Body: {response.text}")
                    logger.error("="*50)
                    raise ValueError(f"Gemini API Error: {response.status_code} - {response.text}")
        elif self.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
            messages = [{"role": "user", "content": prompt}]
            if system_instruction:
                messages.insert(0, {"role": "system", "content": system_instruction})

            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json={"model": "gpt-4o-mini", "messages": messages}, timeout=15.0)
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.error("="*50)
                    logger.error(f"OPENAI TEXT API ERROR: {response.status_code}")
                    logger.error(f"Response Body: {response.text}")
                    logger.error("="*50)
                    raise ValueError(f"OpenAI API Error: {response.status_code} - {response.text}")
        
        elif self.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }
            if system_instruction:
                payload["system"] = system_instruction
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=15.0)
                if response.status_code == 200:
                    data = response.json()
                    return data["content"][0]["text"]
                else:
                    logger.error("="*50)
                    logger.error(f"ANTHROPIC TEXT API ERROR: {response.status_code}")
                    logger.error(f"Response Body: {response.text}")
                    logger.error("="*50)
                    raise ValueError(f"Anthropic API Error: {response.status_code} - {response.text}")

        raise ValueError("No valid LLM provider configured.")

    async def get_ocr(self, base64_image: str) -> Dict[str, Any]:
        """
        Extracts exact OCR table values for Nutrition Facts panels.
        Sends the image to the configured LLM provider with a dedicated OCR prompt.
        """
        logger.info(f"LLM Router: Requesting OCR extraction using {self.provider} provider.")
        ocr_prompt = (
            "This image shows a nutrition facts label. Extract ONLY the following fields "
            "and return a JSON object with keys: 'calories' (integer), 'protein' (float in grams), "
            "'carbs' (float in grams), 'fat' (float in grams). "
            "Values should be per serving as shown on the label. Return ONLY the JSON object."
        )

        if self.provider == "gemini":
            return await self._call_gemini_ocr(base64_image, ocr_prompt)
        elif self.provider == "anthropic":
            return await self._call_anthropic_ocr(base64_image, ocr_prompt)
        elif self.provider == "openai":
            return await self._call_openai_ocr(base64_image, ocr_prompt)
        else:
            logger.error("LLM Router: No valid LLM provider configured for OCR.")
            raise ValueError("No valid LLM API key configured.")

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Robustly extracts JSON from an AI response allowing conversational filler."""
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
        return json.loads(text)

    async def _call_gemini_ocr(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
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
                    return self._extract_json(text)
                except Exception as e:
                    logger.error(f"Gemini OCR parsing error: {e}")
            else:
                logger.error("="*50)
                logger.error(f"GEMINI OCR API ERROR: {response.status_code}")
                logger.error(f"Response Body: {response.text}")
                logger.error("="*50)
        return _OCR_EMPTY_RESULT

    async def _call_anthropic_ocr(self, base64_image: str, prompt: str) -> Dict[str, Any]:
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
                        {"type": "text", "text": prompt}
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
                    return self._extract_json(text)
                except Exception as e:
                    logger.error(f"Anthropic OCR parsing error: {e}")
            else:
                logger.error("="*50)
                logger.error(f"ANTHROPIC OCR API ERROR: {response.status_code}")
                logger.error(f"Response Body: {response.text}")
                logger.error("="*50)
        return _OCR_EMPTY_RESULT

    async def _call_openai_ocr(self, base64_image: str, prompt: str) -> Dict[str, Any]:
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
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
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
                    return self._extract_json(text)
                except Exception as e:
                    logger.error(f"OpenAI OCR parsing error: {e}")
            else:
                logger.error("="*50)
                logger.error(f"OPENAI OCR API ERROR: {response.status_code}")
                logger.error(f"Response Body: {response.text}")
                logger.error("="*50)
        return _OCR_EMPTY_RESULT

    async def _call_gemini(self, base64_image: str) -> Dict[str, Any]:
        # Minimal implementation for Gemini REST API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"

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
                    return self._extract_json(text)
                except Exception as e:
                    logger.error(f"Gemini parsing error: {e}")
            else:
                logger.error("="*50)
                logger.error(f"GEMINI CLASSIFICATION API ERROR: {response.status_code}")
                logger.error(f"Response Body: {response.text}")
                logger.error("="*50)

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
                    return self._extract_json(text)
                except Exception as e:
                    logger.error(f"Anthropic parsing error: {e}. Raw text: {text}")
            else:
                logger.error("="*50)
                logger.error(f"ANTHROPIC CLASSIFICATION API ERROR: {response.status_code}")
                logger.error(f"Response Body: {response.text}")
                logger.error("="*50)

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
                    return self._extract_json(text)
                except Exception as e:
                    logger.error(f"OpenAI parsing error: {e}")
            else:
                logger.error("="*50)
                logger.error(f"OPENAI CLASSIFICATION API ERROR: {response.status_code}")
                logger.error(f"Response Body: {response.text}")
                logger.error("="*50)

        return {"foodName": "Unknown", "prep": "unknown", "is_liquid": False, "confidence": 0.0}

llm_router = LLMRouter()
