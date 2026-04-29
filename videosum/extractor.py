from pathlib import Path

import ffmpeg


def get_duration(video_path: Path) -> float:
    probe = ffmpeg.probe(str(video_path))
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    if video_stream and "duration" in video_stream:
        return float(video_stream["duration"])
    return float(probe["format"]["duration"])


def extract_frames(video_path: Path, interval: int, output_dir: Path) -> list[tuple[float, Path]]:
    pattern = str(output_dir / "frame_%06d.jpg")
    (
        ffmpeg
        .input(str(video_path))
        .filter("fps", fps=f"1/{interval}")
        .output(pattern, vsync="vfr", **{"q:v": 2})
        .run(quiet=True, overwrite_output=True)
    )
    frame_paths = sorted(output_dir.glob("frame_*.jpg"))
    return [(float(i * interval), p) for i, p in enumerate(frame_paths)]


def extract_audio(video_path: Path, output_dir: Path) -> Path | None:
    audio_path = output_dir / "audio.wav"
    try:
        (
            ffmpeg
            .input(str(video_path))
            .audio
            .output(str(audio_path), ar=16000, ac=1, acodec="pcm_s16le")
            .run(quiet=True, overwrite_output=True)
        )
        return audio_path
    except ffmpeg.Error:
        return None
