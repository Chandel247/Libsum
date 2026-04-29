import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .tokenizer import Tokenizer

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FrameCaptionNet(nn.Module):
    """ResNet-18 CNN encoder + LSTM decoder for per-frame captioning.

    The image feature vector (512-d avgpool output, projected to hidden_dim)
    initialises the LSTM hidden state; decoding proceeds autoregressively
    from BOS using greedy search.

    Training: teacher-forced cross-entropy on MS COCO 2017 captions
    (see videosum/nn/train_caption.py).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # drop fc, keep avgpool
        self.img_proj = nn.Linear(512, hidden_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(images).flatten(1)   # (B, 512)
        return self.img_proj(feats)            # (B, hidden_dim)

    def _init_hidden(
        self, img_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.tanh(img_feats).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return h0, torch.zeros_like(h0)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass.

        images:   (B, 3, H, W)
        captions: (B, T)  BOS-prefixed token indices
        returns:  (B, T, vocab_size) logits
        """
        hidden = self._init_hidden(self._encode(images))
        out, _ = self.lstm(self.embedding(captions), hidden)   # (B, T, hidden_dim)
        return self.out_proj(out)                               # (B, T, vocab_size)

    @torch.no_grad()
    def caption(
        self,
        frames: list[bytes],
        tokenizer: Tokenizer,
        max_len: int = 50,
    ) -> list[str]:
        """Greedily decode a caption for each JPEG-encoded frame."""
        self.eval()
        device = next(self.parameters()).device
        results: list[str] = []

        for frame_bytes in frames:
            img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            x = _TRANSFORM(img).unsqueeze(0).to(device)

            hidden = self._init_hidden(self._encode(x))
            token = torch.tensor([[tokenizer.bos_id]], device=device)
            generated: list[int] = []

            for _ in range(max_len):
                out, hidden = self.lstm(self.embedding(token), hidden)
                next_id = int(self.out_proj(out[:, -1, :]).argmax(dim=-1))
                if next_id == tokenizer.eos_id:
                    break
                generated.append(next_id)
                token = torch.tensor([[next_id]], device=device)

            results.append(tokenizer.decode(generated))

        return results

    def save(self, path: str | Path, tokenizer: Tokenizer | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "vocab_size": self.vocab_size,
                    "embed_dim": self.embed_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                },
                "state_dict": self.state_dict(),
                "vocab": tokenizer.vocab if tokenizer else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["FrameCaptionNet", Tokenizer | None]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        tokenizer = Tokenizer(payload["vocab"]) if payload.get("vocab") else None
        return model, tokenizer
