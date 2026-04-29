import io
from pathlib import Path

from PIL import Image

from .providers.base import VisionProvider
from .utils import chunk_list

_FRAME_PROMPT = (
    "Describe what you see in this video frame. Include the main subject or scene, "
    "any visible text or graphics, notable actions, and the setting. "
    "Keep it to 2-3 sentences."
)

_MAX_DIMENSION = 512


def _resize_frame(frame_path: Path) -> bytes:
    with Image.open(frame_path) as img:
        img.thumbnail((_MAX_DIMENSION, _MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def analyze_frames(
    frames: list[tuple[float, Path]],
    provider: VisionProvider,
    batch_size: int = 5,
) -> list[tuple[float, str]]:
    results: list[tuple[float, str]] = []
    for batch in chunk_list(frames, batch_size):
        frame_bytes = [_resize_frame(p) for _, p in batch]
        descriptions = provider.describe_frames(frame_bytes, _FRAME_PROMPT)
        for (ts, _), desc in zip(batch, descriptions):
            results.append((ts, desc))
    return results
