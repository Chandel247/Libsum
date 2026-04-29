"""
Usage examples for videosum.

Run with: python examples/basic_usage.py <path_to_video.mp4>
"""
import sys
from videosum import summarize
from videosum.providers import ClaudeProvider, OllamaProvider


def claude_example(video_path: str) -> None:
    """Pure Claude — best quality, requires ANTHROPIC_API_KEY."""
    provider = ClaudeProvider(model="claude-sonnet-4-6")
    result = summarize(
        video_path,
        vision_provider=provider,
        text_provider=provider,
        frame_interval=5,
        summary_length="medium",
    )
    print("=== TL;DR ===")
    print(result.tldr)
    print("\n=== OUTLINE ===")
    print(result.outline)
    print("\n=== NARRATIVE ===")
    print(result.narrative)
    print(f"\n[{result.frames_analyzed} frames | {result.transcript_word_count} transcript words]")


def ollama_example(video_path: str) -> None:
    """Fully local — requires Ollama running at localhost:11434 with llava and llama3.2 pulled."""
    vision = OllamaProvider(model="llava")
    text = OllamaProvider(model="llama3.2")
    result = summarize(
        video_path,
        vision_provider=vision,
        text_provider=text,
        frame_interval=10,
        summary_length="short",
    )
    print(result.tldr)


def hybrid_example(video_path: str) -> None:
    """Local vision, cloud summarization — balances privacy and summary quality."""
    vision = OllamaProvider(model="llava")
    text = ClaudeProvider(model="claude-sonnet-4-6")
    result = summarize(
        video_path,
        vision_provider=vision,
        text_provider=text,
    )
    print(result.tldr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python basic_usage.py <video_path>")
        sys.exit(1)
    claude_example(sys.argv[1])
