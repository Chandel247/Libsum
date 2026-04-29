from pathlib import Path

from .base import ProviderType, TextProvider, VisionProvider


class ModelNotTrainedError(RuntimeError):
    """Raised when a NeuralProvider method is called before model weights are available."""


class NeuralProvider(VisionProvider, TextProvider):
    """Provider backed by locally-trained ANN models (FrameCaptionNet + VideoSummaryNet).

    Instantiation always succeeds. Errors are raised only when inference is
    attempted without loaded weights, or when the weight file is corrupt /
    the neural extra is not installed.

    Usage (once models are trained):
        provider = NeuralProvider(
            vision_weights="videosum/nn/weights/caption.pt",
            text_weights="videosum/nn/weights/summary.pt",
        )
    """

    provider_type = ProviderType.LOCAL_MODEL

    def __init__(
        self,
        vision_weights: str | Path | None = None,
        text_weights: str | Path | None = None,
    ) -> None:
        self._vision_weights = Path(vision_weights) if vision_weights else None
        self._text_weights = Path(text_weights) if text_weights else None
        self._vision_model = None
        self._vision_tokenizer = None
        self._text_model = None
        self._text_tokenizer = None

        if self._vision_weights:
            self._load_vision()
        if self._text_weights:
            self._load_text()

    def _load_vision(self) -> None:
        if not self._vision_weights.exists():
            raise FileNotFoundError(
                f"Vision weights not found: {self._vision_weights}\n"
                "Train FrameCaptionNet first — see videosum/nn/train_caption.py"
            )
        try:
            from videosum.nn.frame_caption_net import FrameCaptionNet
            self._vision_model, self._vision_tokenizer = FrameCaptionNet.load(
                self._vision_weights
            )
        except ImportError as exc:
            raise ModelNotTrainedError(
                "FrameCaptionNet requires the neural extra: "
                "pip install -e '.[neural]'"
            ) from exc
        except Exception as exc:
            raise ModelNotTrainedError(
                f"FrameCaptionNet: failed to load weights from "
                f"{self._vision_weights}: {exc}"
            ) from exc

    def _load_text(self) -> None:
        if not self._text_weights.exists():
            raise FileNotFoundError(
                f"Text weights not found: {self._text_weights}\n"
                "Train VideoSummaryNet first — see videosum/nn/train_summary.py"
            )
        try:
            from videosum.nn.video_summary_net import VideoSummaryNet
            self._text_model, self._text_tokenizer = VideoSummaryNet.load(
                self._text_weights
            )
        except ImportError as exc:
            raise ModelNotTrainedError(
                "VideoSummaryNet requires the neural extra: "
                "pip install -e '.[neural]'"
            ) from exc
        except Exception as exc:
            raise ModelNotTrainedError(
                f"VideoSummaryNet: failed to load weights from "
                f"{self._text_weights}: {exc}"
            ) from exc

    def describe_frames(self, frames: list[bytes], prompt: str = "") -> list[str]:
        if self._vision_model is None:
            raise ModelNotTrainedError(
                "Vision model not loaded. "
                "Train FrameCaptionNet and pass vision_weights= to NeuralProvider. "
                "See videosum/nn/train_caption.py."
            )
        return self._vision_model.caption(frames, self._vision_tokenizer)

    def generate(self, system: str, user: str) -> str:
        if self._text_model is None:
            raise ModelNotTrainedError(
                "Text model not loaded. "
                "Train VideoSummaryNet and pass text_weights= to NeuralProvider. "
                "See videosum/nn/train_summary.py."
            )
        combined = f"{system}\n\n{user}".strip() if system else user
        return self._text_model.generate(combined, self._text_tokenizer)
