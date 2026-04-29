import io
from pathlib import Path

import pytest
from PIL import Image

from videosum.analyzer import _resize_frame, analyze_frames


def make_jpeg(path: Path, size: tuple[int, int] = (64, 64)) -> Path:
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(path, "JPEG")
    return path


def test_resize_frame_caps_max_dimension(tmp_path):
    large = make_jpeg(tmp_path / "big.jpg", size=(1024, 768))
    result = _resize_frame(large)
    img = Image.open(io.BytesIO(result))
    assert max(img.size) <= 512


def test_resize_frame_preserves_small_image(tmp_path):
    small = make_jpeg(tmp_path / "small.jpg", size=(100, 80))
    result = _resize_frame(small)
    img = Image.open(io.BytesIO(result))
    assert img.size == (100, 80)


def test_analyze_frames_calls_provider_in_batches(tmp_path, mock_vision_provider):
    frames = [(float(i * 5), make_jpeg(tmp_path / f"frame_{i}.jpg")) for i in range(6)]
    mock_vision_provider.describe_frames.side_effect = [
        ["desc"] * 5,
        ["desc"],
    ]

    results = analyze_frames(frames, mock_vision_provider, batch_size=5)

    assert len(results) == 6
    assert mock_vision_provider.describe_frames.call_count == 2


def test_analyze_frames_preserves_timestamps(tmp_path, mock_vision_provider):
    frames = [
        (0.0, make_jpeg(tmp_path / "f0.jpg")),
        (5.0, make_jpeg(tmp_path / "f5.jpg")),
    ]
    mock_vision_provider.describe_frames.return_value = ["first scene", "second scene"]

    results = analyze_frames(frames, mock_vision_provider, batch_size=5)

    assert results[0] == (0.0, "first scene")
    assert results[1] == (5.0, "second scene")


def test_analyze_frames_empty_input(mock_vision_provider):
    results = analyze_frames([], mock_vision_provider)
    assert results == []
    mock_vision_provider.describe_frames.assert_not_called()
