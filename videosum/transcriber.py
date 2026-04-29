from pathlib import Path

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

from .models import TranscriptSegment


def transcribe(audio_path: Path, model_size: str = "base") -> list[TranscriptSegment]:
    if not _WHISPER_AVAILABLE:
        return []
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), beam_size=5)
    return [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments
    ]
