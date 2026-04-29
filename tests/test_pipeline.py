from pathlib import Path
from unittest.mock import patch

import pytest

from videosum.models import SummaryResult
from videosum.pipeline import summarize_video


def _make_result(**kwargs) -> SummaryResult:
    defaults = dict(
        tldr="Test TL;DR",
        outline="- Step 1",
        narrative="Full narrative here.",
        duration_seconds=10.0,
        frames_analyzed=2,
        transcript_word_count=0,
    )
    return SummaryResult(**{**defaults, **kwargs})


def test_raises_on_missing_file(mock_vision_provider, mock_text_provider):
    with pytest.raises(FileNotFoundError):
        summarize_video(
            "/nonexistent/video.mp4",
            vision_provider=mock_vision_provider,
            text_provider=mock_text_provider,
        )


def test_returns_summary_result(tmp_path, mock_vision_provider, mock_text_provider):
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake video data")

    with (
        patch("videosum.pipeline.extractor.extract_frames", return_value=[(0.0, tmp_path / "f.jpg")]),
        patch("videosum.pipeline.extractor.extract_audio", return_value=None),
        patch("videosum.pipeline.analyzer.analyze_frames", return_value=[(0.0, "A test scene")]),
        patch("videosum.pipeline.summarizer.summarize", return_value=_make_result()),
    ):
        result = summarize_video(fake_video, mock_vision_provider, mock_text_provider)

    assert isinstance(result, SummaryResult)
    assert result.tldr == "Test TL;DR"


def test_skips_audio_when_transcript_disabled(tmp_path, mock_vision_provider, mock_text_provider):
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake")

    with (
        patch("videosum.pipeline.extractor.extract_frames", return_value=[]),
        patch("videosum.pipeline.extractor.extract_audio") as mock_audio,
        patch("videosum.pipeline.analyzer.analyze_frames", return_value=[]),
        patch("videosum.pipeline.summarizer.summarize", return_value=_make_result()),
    ):
        summarize_video(
            fake_video,
            mock_vision_provider,
            mock_text_provider,
            include_transcript=False,
        )

    mock_audio.assert_not_called()


def test_passes_frame_interval_to_extractor(tmp_path, mock_vision_provider, mock_text_provider):
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake")

    with (
        patch("videosum.pipeline.extractor.extract_frames", return_value=[]) as mock_frames,
        patch("videosum.pipeline.extractor.extract_audio", return_value=None),
        patch("videosum.pipeline.analyzer.analyze_frames", return_value=[]),
        patch("videosum.pipeline.summarizer.summarize", return_value=_make_result()),
    ):
        summarize_video(
            fake_video,
            mock_vision_provider,
            mock_text_provider,
            frame_interval=10,
        )

    _, call_kwargs = mock_frames.call_args
    assert mock_frames.call_args[0][1] == 10
