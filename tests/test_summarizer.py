import pytest

from videosum.models import SummaryResult, TranscriptSegment
from videosum.summarizer import _build_prompt, _parse_response, summarize


def test_parse_response_extracts_all_sections():
    text = (
        "TL;DR:\nThis is a short summary.\n\n"
        "OUTLINE:\n[0:00] Introduction\n[1:00] Main content\n\n"
        "NARRATIVE:\nThis video covers many topics in depth."
    )
    tldr, outline, narrative = _parse_response(text)
    assert "short summary" in tldr
    assert "Introduction" in outline
    assert "Main content" in outline
    assert "many topics" in narrative


def test_parse_response_falls_back_to_raw_on_missing_sections():
    raw = "Some unstructured text with no sections"
    tldr, outline, narrative = _parse_response(raw)
    assert tldr == raw
    assert outline == ""
    assert narrative == ""


def test_parse_response_handles_markdown_headers():
    text = (
        "## **TL;DR:**\nA sci-fi epic unfolds.\n\n"
        "**Outline:**\n[0:00] Opening scene\n\n"
        "### Narrative:\nThe story begins in the desert."
    )
    tldr, outline, narrative = _parse_response(text)
    assert "sci-fi" in tldr
    assert "Opening" in outline
    assert "desert" in narrative


def test_build_prompt_includes_both_streams(sample_transcript):
    descriptions = [(0.0, "A presenter at a whiteboard")]
    prompt = _build_prompt(descriptions, sample_transcript, "medium")
    assert "Visual Scene Descriptions" in prompt
    assert "Speech Transcript" in prompt
    assert "presenter at a whiteboard" in prompt
    assert "tutorial" in prompt


def test_build_prompt_omits_transcript_section_when_empty():
    descriptions = [(0.0, "A scene")]
    prompt = _build_prompt(descriptions, [], "medium")
    assert "Speech Transcript" not in prompt


def test_summarize_returns_correct_metadata(mock_text_provider, sample_transcript):
    descriptions = [(0.0, "Scene A"), (5.0, "Scene B")]
    result = summarize(descriptions, sample_transcript, mock_text_provider, "medium")

    assert isinstance(result, SummaryResult)
    assert result.frames_analyzed == 2
    assert result.duration_seconds == 5.0
    assert result.transcript_word_count == 12


def test_summarize_calls_provider_once(mock_text_provider, sample_transcript):
    summarize([(0.0, "scene")], sample_transcript, mock_text_provider)
    mock_text_provider.generate.assert_called_once()


def test_summarize_no_transcript(mock_text_provider):
    result = summarize([(0.0, "scene")], [], mock_text_provider)
    assert result.transcript_word_count == 0
    prompt_arg = mock_text_provider.generate.call_args[0][1]
    assert "Speech Transcript" not in prompt_arg
