from pathlib import Path

from . import analyzer, extractor, summarizer, transcriber
from .models import SummaryResult
from .providers.base import TextProvider, VisionProvider
from .utils import TempDir


def summarize_video(
    video_path: str | Path,
    vision_provider: VisionProvider,
    text_provider: TextProvider,
    frame_interval: int = 5,
    summary_length: str = "medium",
    include_transcript: bool = True,
) -> SummaryResult:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    with TempDir() as tmp:
        frames = extractor.extract_frames(video_path, frame_interval, tmp)

        transcript_segments = []
        if include_transcript:
            audio_path = extractor.extract_audio(video_path, tmp)
            if audio_path:
                transcript_segments = transcriber.transcribe(audio_path)

        descriptions = analyzer.analyze_frames(frames, vision_provider)
        return summarizer.summarize(
            descriptions, transcript_segments, text_provider, summary_length
        )
