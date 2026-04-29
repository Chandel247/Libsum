from pathlib import Path

from .base import ProviderType, TextProvider, VisionProvider


class ModelNotTrainedError(RuntimeError):
    """Raised when a NeuralProvider method is called before model weights are available."""


class NeuralProvider(VisionProvider, TextProvider):
    """Provider backed by locally-trained ANN models (FrameCaptionNet + VideoSummaryNet).

    This class is a structural placeholder. The underlying models live in
    videosum/nn/ and must be trained before this provider becomes functional.
    Instantiation succeeds regardless; errors are raised only when inference
    is attempted without loaded weights.

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
    ):
        self._vision_weights = Path(vision_weights) if vision_weights else None
        self._text_weights = Path(text_weights) if text_weights else None
        self._vision_model = None
        self._text_model = None

        if self._vision_weights:
            self._try_load_vision()
        if self._text_weights:
            self._try_load_text()

    # ------------------------------------------------------------------
    # Model loading — will wire up to videosum/nn once models are built
    # ------------------------------------------------------------------

    def _try_load_vision(self) -> None:
        if not self._vision_weights.exists():
            raise FileNotFoundError(
                f"Vision weights not found: {self._vision_weights}\n"
                "Train FrameCaptionNet first — see videosum/nn/frame_caption_net.py"
            )
        # TODO: import and load FrameCaptionNet once videosum/nn is implemented
        # from videosum.nn.frame_caption_net import FrameCaptionNet
        # self._vision_model = FrameCaptionNet.load(self._vision_weights)
        raise ModelNotTrainedError(
            "FrameCaptionNet is not yet implemented.\n"
            "Build and train it in videosum/nn/frame_caption_net.py, then revisit this loader."
        )

    def _try_load_text(self) -> None:
        if not self._text_weights.exists():
            raise FileNotFoundError(
                f"Text weights not found: {self._text_weights}\n"
                "Train VideoSummaryNet first — see videosum/nn/video_summary_net.py"
            )
        # TODO: import and load VideoSummaryNet once videosum/nn is implemented
        # from videosum.nn.video_summary_net import VideoSummaryNet
        # self._text_model = VideoSummaryNet.load(self._text_weights)
        raise ModelNotTrainedError(
            "VideoSummaryNet is not yet implemented.\n"
            "Build and train it in videosum/nn/video_summary_net.py, then revisit this loader."
        )

    # ------------------------------------------------------------------
    # VisionProvider
    # ------------------------------------------------------------------

    def describe_frames(self, frames: list[bytes], prompt: str = "") -> list[str]:
        """Caption frames using FrameCaptionNet (CNN-LSTM).

        Unlike API/service providers, `prompt` is not used — the model
        generates captions entirely from learned visual features.
        """
        if self._vision_model is None:
            raise ModelNotTrainedError(
                "Vision model not loaded. "
                "Train FrameCaptionNet and pass vision_weights= to NeuralProvider. "
                "See videosum/nn/frame_caption_net.py."
            )
        return self._vision_model.caption(frames)

    # ------------------------------------------------------------------
    # TextProvider
    # ------------------------------------------------------------------

    def generate(self, system: str, user: str) -> str:
        """Summarize using VideoSummaryNet (Seq2Seq + Attention).

        `system` is prepended to `user` as plain text — the model has no
        concept of a system/user role split.
        """
        if self._text_model is None:
            raise ModelNotTrainedError(
                "Text model not loaded. "
                "Train VideoSummaryNet and pass text_weights= to NeuralProvider. "
                "See videosum/nn/video_summary_net.py."
            )
        combined = f"{system}\n\n{user}".strip() if system else user
        return self._text_model.generate(combined)
