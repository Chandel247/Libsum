from .models import SummaryResult
from .pipeline import summarize_video as summarize
from .providers import (
    ClaudeProvider,
    ModelNotTrainedError,
    NeuralProvider,
    OllamaProvider,
    ProviderType,
    TextProvider,
    VisionProvider,
    available_providers,
    create_provider,
)

__all__ = [
    "summarize",
    "SummaryResult",
    # Abstract interfaces
    "VisionProvider",
    "TextProvider",
    "ProviderType",
    # Concrete providers
    "ClaudeProvider",
    "OllamaProvider",
    "NeuralProvider",
    # Errors
    "ModelNotTrainedError",
    # Factory
    "create_provider",
    "available_providers",
]
