import pytest

from videosum.providers import (
    ClaudeProvider,
    ModelNotTrainedError,
    NeuralProvider,
    OllamaProvider,
    ProviderType,
    available_providers,
    create_provider,
)


# ── ProviderType tagging ────────────────────────────────────────────────────

def test_claude_provider_type():
    assert ClaudeProvider.provider_type == ProviderType.API


def test_ollama_provider_type():
    assert OllamaProvider.provider_type == ProviderType.LOCAL_SERVICE


def test_neural_provider_type():
    assert NeuralProvider.provider_type == ProviderType.LOCAL_MODEL


# ── Registry ────────────────────────────────────────────────────────────────

def test_available_providers_lists_all_three():
    names = available_providers()
    assert "claude" in names
    assert "ollama" in names
    assert "neural" in names


def test_create_provider_returns_correct_type():
    p = create_provider("ollama", model="llava")
    assert isinstance(p, OllamaProvider)


def test_create_provider_neural_no_weights():
    p = create_provider("neural")
    assert isinstance(p, NeuralProvider)


def test_create_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("gpt5")


# ── NeuralProvider ──────────────────────────────────────────────────────────

def test_neural_provider_instantiates_without_weights():
    p = NeuralProvider()
    assert p._vision_model is None
    assert p._text_model is None


def test_neural_provider_describe_frames_raises_before_training():
    p = NeuralProvider()
    with pytest.raises(ModelNotTrainedError, match="Vision model"):
        p.describe_frames([b"fake-jpeg"])


def test_neural_provider_generate_raises_before_training():
    p = NeuralProvider()
    with pytest.raises(ModelNotTrainedError, match="Text model"):
        p.generate("system", "user")


def test_neural_provider_missing_vision_weights_file(tmp_path):
    nonexistent = tmp_path / "caption.pt"
    with pytest.raises(FileNotFoundError, match="Vision weights not found"):
        NeuralProvider(vision_weights=nonexistent)


def test_neural_provider_missing_text_weights_file(tmp_path):
    nonexistent = tmp_path / "summary.pt"
    with pytest.raises(FileNotFoundError, match="Text weights not found"):
        NeuralProvider(text_weights=nonexistent)


def test_neural_provider_weights_exist_but_model_not_built(tmp_path):
    # Weights file present but FrameCaptionNet not implemented yet —
    # should raise ModelNotTrainedError, not FileNotFoundError.
    weights = tmp_path / "caption.pt"
    weights.write_bytes(b"fake weights")
    with pytest.raises(ModelNotTrainedError, match="FrameCaptionNet"):
        NeuralProvider(vision_weights=weights)


def test_neural_provider_prompt_is_optional():
    # describe_frames signature accepts prompt with a default — confirm it
    # can be called without the argument (the error should be about the missing
    # model, not a missing argument).
    p = NeuralProvider()
    with pytest.raises(ModelNotTrainedError):
        p.describe_frames([b"fake"])     # no prompt= keyword
