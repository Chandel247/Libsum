from abc import ABC, abstractmethod
from enum import Enum


class ProviderType(Enum):
    API = "api"                     # Cloud API (Claude, OpenAI, etc.)
    LOCAL_SERVICE = "local_service" # Local inference server (Ollama)
    LOCAL_MODEL = "local_model"     # Embedded model weights (custom ANN)


class VisionProvider(ABC):
    provider_type: ProviderType  # subclasses must set this as a class attribute

    @abstractmethod
    def describe_frames(self, frames: list[bytes], prompt: str = "") -> list[str]:
        """Given a list of JPEG bytes, return one description per frame.

        `prompt` guides API/service-backed providers. Local model providers
        (e.g. NeuralProvider) may ignore it and generate from learned weights.
        """
        ...


class TextProvider(ABC):
    provider_type: ProviderType

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Given system and user prompts, return generated text.

        Local model providers may concatenate or ignore `system` depending on
        how the underlying model was trained.
        """
        ...
