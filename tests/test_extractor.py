from pathlib import Path
from unittest.mock import patch

import pytest

from videosum.extractor import extract_audio, extract_frames, get_duration


def test_get_duration_from_video_stream():
    probe_data = {
        "streams": [{"codec_type": "video", "duration": "120.5"}],
        "format": {"duration": "120.0"},
    }
    with patch("videosum.extractor.ffmpeg.probe", return_value=probe_data):
        assert get_duration(Path("fake.mp4")) == 120.5


def test_get_duration_falls_back_to_format():
    probe_data = {
        "streams": [{"codec_type": "audio"}],
        "format": {"duration": "60.0"},
    }
    with patch("videosum.extractor.ffmpeg.probe", return_value=probe_data):
        assert get_duration(Path("fake.mp4")) == 60.0


def test_extract_frames_returns_correct_timestamps(tmp_path):
    for i in range(1, 4):
        (tmp_path / f"frame_{i:06d}.jpg").write_bytes(b"fake")

    with patch("videosum.extractor.ffmpeg.input") as mock_input:
        mock_input.return_value.filter.return_value.output.return_value.run.return_value = None
        frames = extract_frames(Path("fake.mp4"), interval=5, output_dir=tmp_path)

    assert len(frames) == 3
    assert frames[0][0] == 0.0
    assert frames[1][0] == 5.0
    assert frames[2][0] == 10.0
    assert all(isinstance(p, Path) for _, p in frames)


def test_extract_audio_returns_none_on_ffmpeg_error(tmp_path):
    import ffmpeg as ffmpeg_module

    with patch("videosum.extractor.ffmpeg.input") as mock_input:
        mock_input.return_value.audio.output.return_value.run.side_effect = (
            ffmpeg_module.Error("ffmpeg", b"", b"stderr output")
        )
        result = extract_audio(Path("fake.mp4"), tmp_path)

    assert result is None
