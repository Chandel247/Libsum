"""Tests for FrameCaptionNet and VideoSummaryNet — requires torch (neural extra)."""

import io

import pytest

torch = pytest.importorskip("torch", reason="neural extra (torch) not installed")

from videosum.nn.frame_caption_net import FrameCaptionNet   # noqa: E402
from videosum.nn.tokenizer import Tokenizer                  # noqa: E402
from videosum.nn.video_summary_net import VideoSummaryNet    # noqa: E402


# ── FrameCaptionNet ──────────────────────────────────────────���───────────────

def test_caption_net_forward_shape():
    vocab_size, B, T = 100, 2, 10
    model = FrameCaptionNet(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
    imgs = torch.zeros(B, 3, 224, 224)
    caps = torch.zeros(B, T, dtype=torch.long)
    logits = model(imgs, caps)
    assert logits.shape == (B, T, vocab_size)


def test_caption_net_caption_returns_list_of_strings():
    from PIL import Image

    tok = Tokenizer()
    tok.build_vocab(["a scene with a person walking"], min_freq=1)
    model = FrameCaptionNet(vocab_size=tok.vocab_size, embed_dim=32, hidden_dim=64)

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    frames = [buf.getvalue(), buf.getvalue()]

    captions = model.caption(frames, tok, max_len=10)
    assert len(captions) == 2
    assert all(isinstance(c, str) for c in captions)


def test_caption_net_save_load(tmp_path):
    tok = Tokenizer()
    tok.build_vocab(["hello world test"], min_freq=1)
    model = FrameCaptionNet(vocab_size=tok.vocab_size, embed_dim=32, hidden_dim=64)
    path = tmp_path / "caption.pt"
    model.save(path, tok)

    loaded, loaded_tok = FrameCaptionNet.load(path)
    assert loaded_tok.vocab == tok.vocab
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.embed_dim == 32
    assert loaded.hidden_dim == 64


# ── VideoSummaryNet ────────────────────��─────────────────────────────────��───

def test_summary_net_forward_shape():
    vocab_size, B, S, T = 100, 2, 15, 8
    model = VideoSummaryNet(vocab_size=vocab_size, embed_dim=32, hidden_dim=64, num_layers=1)
    src = torch.zeros(B, S, dtype=torch.long)
    tgt = torch.zeros(B, T, dtype=torch.long)
    logits = model(src, tgt)
    assert logits.shape == (B, T, vocab_size)


def test_summary_net_forward_with_src_mask():
    vocab_size, B, S, T = 100, 2, 10, 6
    model = VideoSummaryNet(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    src = torch.randint(1, vocab_size, (B, S))
    tgt = torch.randint(1, vocab_size, (B, T))
    logits = model(src, tgt, src_mask=src.ne(0))
    assert logits.shape == (B, T, vocab_size)


def test_summary_net_generate_returns_string():
    tok = Tokenizer()
    tok.build_vocab(["the quick brown fox jumps over the lazy dog"], min_freq=1)
    model = VideoSummaryNet(vocab_size=tok.vocab_size, embed_dim=32, hidden_dim=64)
    result = model.generate("the quick brown fox", tok, max_len=15, min_len=1)
    assert isinstance(result, str)


def test_summary_net_generate_empty_input():
    tok = Tokenizer()
    tok.build_vocab(["hello"], min_freq=1)
    model = VideoSummaryNet(vocab_size=tok.vocab_size, embed_dim=32, hidden_dim=64)
    assert model.generate("", tok) == ""


def test_summary_net_save_load(tmp_path):
    tok = Tokenizer()
    tok.build_vocab(["save load roundtrip"], min_freq=1)
    model = VideoSummaryNet(vocab_size=tok.vocab_size, embed_dim=32, hidden_dim=64, num_layers=2)
    path = tmp_path / "summary.pt"
    model.save(path, tok)

    loaded, loaded_tok = VideoSummaryNet.load(path)
    assert loaded_tok.vocab == tok.vocab
    assert loaded.num_layers == 2
    assert loaded.hidden_dim == 64
