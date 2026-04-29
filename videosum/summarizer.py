from .models import SummaryResult, TranscriptSegment
from .providers.base import TextProvider
from .utils import format_timestamp

_SYSTEM = (
    "You are an expert video content analyst. Your task is to produce structured summaries "
    "from combined visual scene descriptions and speech transcripts of a video. "
    "Be concise, accurate, and well-organized."
)

_LENGTH_INSTRUCTIONS: dict[str, str] = {
    "short": (
        "Keep the TL;DR under 50 words, the outline to 3-5 entries, "
        "and the narrative under 150 words."
    ),
    "medium": (
        "Keep the TL;DR under 100 words, the outline to 5-10 entries, "
        "and the narrative under 400 words."
    ),
    "long": (
        "Keep the TL;DR under 150 words, provide a detailed outline, "
        "and write a thorough narrative."
    ),
}


def _build_prompt(
    descriptions: list[tuple[float, str]],
    transcript: list[TranscriptSegment],
    summary_length: str,
) -> str:
    lines = ["## Visual Scene Descriptions\n"]
    for ts, desc in descriptions:
        lines.append(f"[{format_timestamp(ts)}] {desc}")

    if transcript:
        lines.append("\n## Speech Transcript\n")
        for seg in transcript:
            lines.append(f"[{format_timestamp(seg.start)}] {seg.text}")

    length_instr = _LENGTH_INSTRUCTIONS.get(summary_length, _LENGTH_INSTRUCTIONS["medium"])
    lines.append(f"\n## Task\n{length_instr}")
    lines.append(
        "\nUsing the above, produce a structured summary in exactly this format:\n\n"
        "TL;DR:\n<one paragraph>\n\n"
        "OUTLINE:\n<timestamped bullet points>\n\n"
        "NARRATIVE:\n<prose summary>"
    )
    return "\n".join(lines)


def _parse_response(text: str) -> tuple[str, str, str]:
    sections: dict[str, str] = {"TL;DR:": "", "OUTLINE:": "", "NARRATIVE:": ""}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in sections:
            current = stripped
        elif current is not None:
            sections[current] += line + "\n"
    return (
        sections["TL;DR:"].strip(),
        sections["OUTLINE:"].strip(),
        sections["NARRATIVE:"].strip(),
    )


def summarize(
    descriptions: list[tuple[float, str]],
    transcript: list[TranscriptSegment],
    provider: TextProvider,
    summary_length: str = "medium",
) -> SummaryResult:
    prompt = _build_prompt(descriptions, transcript, summary_length)
    response = provider.generate(_SYSTEM, prompt)
    tldr, outline, narrative = _parse_response(response)

    duration = descriptions[-1][0] if descriptions else 0.0
    word_count = sum(len(seg.text.split()) for seg in transcript)

    return SummaryResult(
        tldr=tldr,
        outline=outline,
        narrative=narrative,
        duration_seconds=duration,
        frames_analyzed=len(descriptions),
        transcript_word_count=word_count,
    )
