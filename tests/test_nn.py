"""Tests for Tokenizer — no torch dependency."""

import pytest

from videosum.nn.tokenizer import Tokenizer


def test_special_token_ids():
    tok = Tokenizer()
    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.bos_id == 2
    assert tok.eos_id == 3


def test_build_vocab():
    tok = Tokenizer()
    tok.build_vocab(["hello world", "hello python", "python is great"], min_freq=1)
    assert "hello" in tok.vocab
    assert "python" in tok.vocab
    assert tok.vocab_size > 4


def test_encode_decode_roundtrip():
    tok = Tokenizer()
    tok.build_vocab(["the cat sat on the mat"], min_freq=1)
    ids = tok.encode("the cat sat", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)
    assert "cat" in decoded
    assert "sat" in decoded


def test_unknown_word_maps_to_unk():
    tok = Tokenizer()
    tok.build_vocab(["known word"], min_freq=1)
    ids = tok.encode("known zxqjunk")
    assert tok.unk_id in ids


def test_save_load_roundtrip(tmp_path):
    tok = Tokenizer()
    tok.build_vocab(["save and load test"], min_freq=1)
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.vocab == tok.vocab
    assert loaded.vocab_size == tok.vocab_size
