import base64

import httpx

from .base import ProviderType, TextProvider, VisionProvider


class OllamaProvider(VisionProvider, TextProvider):
    provider_type = ProviderType.LOCAL_SERVICE

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def describe_frames(self, frames: list[bytes], prompt: str) -> list[str]:
        # Ollama processes one image per call reliably; loop over frames.
        descriptions = []
        for frame in frames:
            payload = {
                "model": self._model,
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [base64.standard_b64encode(frame).decode()],
                }],
                "stream": False,
            }
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            descriptions.append(response.json()["message"]["content"].strip())
        return descriptions

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=300.0,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
