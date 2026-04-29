import pytest
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from videosum.models import TranscriptSegment
from videosum.providers.base import TextProvider, VisionProvider


@pytest.fixture
def sample_jpeg(tmp_path) -> Path:
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    path = tmp_path / "frame.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def sample_frames(tmp_path, sample_jpeg) -> list[tuple[float, Path]]:
    return [(0.0, sample_jpeg), (5.0, sample_jpeg), (10.0, sample_jpeg)]


@pytest.fixture
def sample_transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=3.0, text="Hello and welcome to this tutorial."),
        TranscriptSegment(start=3.0, end=7.0, text="Today we will learn about Python."),
    ]


@pytest.fixture
def mock_vision_provider() -> VisionProvider:
    provider = MagicMock(spec=VisionProvider)
    provider.describe_frames.return_value = ["A test frame description."]
    return provider


@pytest.fixture
def mock_text_provider() -> TextProvider:
    provider = MagicMock(spec=TextProvider)
    provider.generate.return_value = (
        "TL;DR:\nThis is a test summary.\n\n"
        "OUTLINE:\n[0:00] Introduction\n\n"
        "NARRATIVE:\nThis video covers the topic in detail."
    )
    return provider
