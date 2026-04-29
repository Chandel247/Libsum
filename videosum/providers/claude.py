import base64
import json

from anthropic import Anthropic

from .base import ProviderType, TextProvider, VisionProvider

_VISION_SYSTEM = (
    "You are a video analysis assistant. Describe video frames concisely and accurately, "
    "focusing on scene content, visible text, and notable actions."
)


class ClaudeProvider(VisionProvider, TextProvider):
    provider_type = ProviderType.API

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def describe_frames(self, frames: list[bytes], prompt: str) -> list[str]:
        content: list[dict] = []
        for frame in frames:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(frame).decode(),
                },
            })
        content.append({
            "type": "text",
            "text": (
                f"{prompt}\n\n"
                f"There are {len(frames)} image(s) above. "
                "Return a JSON array of strings — exactly one description per image, in order."
            ),
        })

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _VISION_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)

    def generate(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
