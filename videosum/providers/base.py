from abc import ABC, abstractmethod


class VisionProvider(ABC):
    @abstractmethod
    def describe_frames(self, frames: list[bytes], prompt: str) -> list[str]:
        """Given a list of JPEG bytes, return one description per frame."""
        ...


class TextProvider(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Given system and user prompts, return generated text."""
        ...
