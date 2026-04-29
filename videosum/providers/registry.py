from .base import TextProvider, VisionProvider
from .claude import ClaudeProvider
from .neural import NeuralProvider
from .ollama import OllamaProvider

# Maps short names to provider classes. Extend this dict when adding new backends.
_REGISTRY: dict[str, type] = {
    "claude": ClaudeProvider,
    "ollama": OllamaProvider,
    "neural": NeuralProvider,
}


def create_provider(name: str, **kwargs) -> ClaudeProvider | OllamaProvider | NeuralProvider:
    """Instantiate a provider by name.

    Passes all keyword arguments to the provider's __init__.

    Examples:
        create_provider("claude", model="claude-sonnet-4-6")
        create_provider("ollama", model="llava")
        create_provider("neural", vision_weights="nn/weights/caption.pt",
                                   text_weights="nn/weights/summary.pt")
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Available: {available_providers()}"
        )
    return _REGISTRY[name](**kwargs)


def available_providers() -> list[str]:
    """Return the names of all registered providers."""
    return list(_REGISTRY)
