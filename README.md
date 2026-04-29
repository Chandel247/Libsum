# videosum

A small-scale Python library that summarizes video files using AI. It extracts frames and audio from a video, analyses the visuals and transcribes the speech, then produces a structured natural-language summary.

## How it works

```
Video file
  ├── Frame extractor (FFmpeg)  →  sampled JPEG frames
  │                                       ↓
  │                              Vision provider (Claude / Ollama)
  │                                       ↓
  │                              Scene descriptions per frame
  │
  └── Audio extractor (FFmpeg)  →  WAV audio
                                          ↓
                                   Transcriber (Whisper)
                                          ↓
                                   Timestamped transcript
                                          ↓
                             ┌────────────┘
                             │  scenes + transcript
                             ↓
                      Text provider (Claude / Ollama)
                             ↓
                      TL;DR · Outline · Narrative
```

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed on your system

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

## Installation

```bash
git clone <repo-url>
cd videosum_proj
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Providers

The library has no hard-coded AI backend. You choose a **vision provider** (analyses frames) and a **text provider** (writes the summary) independently.

| Provider | Class | `ProviderType` | Requires |
|---|---|---|---|
| Anthropic Claude | `ClaudeProvider` | `API` | `ANTHROPIC_API_KEY` env var |
| Ollama (local) | `OllamaProvider` | `LOCAL_SERVICE` | Ollama running at `localhost:11434` |
| Custom ANN | `NeuralProvider` | `LOCAL_MODEL` | Trained weight files (see [Neural provider](#neural-provider)) |

You can mix and match — e.g. local vision with Claude summarization.

### Provider factory

Instead of importing classes directly you can use the registry:

```python
from videosum.providers import create_provider, available_providers

print(available_providers())  # ['claude', 'ollama', 'neural']

provider = create_provider("claude", model="claude-sonnet-4-6")
provider = create_provider("ollama", model="llava")
```

## Quick start

### Option A — Claude (cloud)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
from videosum import summarize
from videosum.providers import ClaudeProvider

provider = ClaudeProvider()  # uses claude-sonnet-4-6 by default

result = summarize(
    "path/to/video.mp4",
    vision_provider=provider,
    text_provider=provider,
)

print(result.tldr)
print(result.outline)
print(result.narrative)
```

### Option B — Ollama (fully local)

```bash
ollama pull llava      # vision model
ollama pull llama3.2   # text model
```

```python
from videosum import summarize
from videosum.providers import OllamaProvider

result = summarize(
    "path/to/video.mp4",
    vision_provider=OllamaProvider("llava"),
    text_provider=OllamaProvider("llama3.2"),
)

print(result.tldr)
```

### Option C — Hybrid

```python
from videosum import summarize
from videosum.providers import ClaudeProvider, OllamaProvider

result = summarize(
    "path/to/video.mp4",
    vision_provider=OllamaProvider("llava"),   # local, private
    text_provider=ClaudeProvider(),             # cloud, higher quality
)
```

## `summarize()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `video_path` | `str \| Path` | — | Path to the video file |
| `vision_provider` | `VisionProvider` | — | Provider for frame analysis |
| `text_provider` | `TextProvider` | — | Provider for summary generation |
| `frame_interval` | `int` | `5` | Seconds between sampled frames |
| `summary_length` | `str` | `"medium"` | `"short"`, `"medium"`, or `"long"` |
| `include_transcript` | `bool` | `True` | Whether to transcribe audio |

## Output — `SummaryResult`

```python
result.tldr               # one-paragraph overview
result.outline            # timestamped bullet-point outline
result.narrative          # full prose summary
result.duration_seconds   # video length in seconds
result.frames_analyzed    # number of frames sent to vision provider
result.transcript_word_count  # words in the audio transcript
```

## Neural provider

`NeuralProvider` is a structural placeholder for locally-trained ANN models. Instantiation always succeeds; inference raises `ModelNotTrainedError` until the underlying models are built and weights are provided.

Two planned models live in `videosum/nn/` (not yet implemented):

| Model | Architecture | Task |
|---|---|---|
| `FrameCaptionNet` | ResNet-18 + LSTM decoder | Frame captioning (train on MS COCO) |
| `VideoSummaryNet` | BiLSTM encoder + Attention decoder | Text summarization (train on CNN/DailyMail) |

Install the `neural` extra before training:

```bash
pip install -e ".[neural]"
```

Once weights are available:

```python
from videosum.providers import NeuralProvider

provider = NeuralProvider(
    vision_weights="videosum/nn/weights/caption.pt",
    text_weights="videosum/nn/weights/summary.pt",
)
```

## Custom providers

Implement the two abstract methods to plug in any backend:

```python
from videosum.providers import VisionProvider, TextProvider

class MyVisionProvider(VisionProvider):
    def describe_frames(self, frames: list[bytes], prompt: str) -> list[str]:
        # frames: list of JPEG bytes, one per sampled frame
        # return: one description string per frame
        ...

class MyTextProvider(TextProvider):
    def generate(self, system: str, user: str) -> str:
        # return generated text
        ...
```

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

All 20 tests run without a real video file or any API keys — external calls are mocked.

## Generating a test video

If you don't have a video handy, FFmpeg can make one:

```bash
ffmpeg -f lavfi -i testsrc=duration=10:size=640x360:rate=30 \
       -f lavfi -i sine=frequency=440:duration=10 \
       -shortest test_clip.mp4
```