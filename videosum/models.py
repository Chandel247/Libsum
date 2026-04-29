from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class SummaryResult:
    tldr: str
    outline: str
    narrative: str
    duration_seconds: float
    frames_analyzed: int
    transcript_word_count: int
    metadata: dict = field(default_factory=dict)
