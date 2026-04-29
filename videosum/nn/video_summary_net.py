from pathlib import Path

import torch
import torch.nn as nn

from .tokenizer import Tokenizer


class _BahdanauAttention(nn.Module):
    """Additive attention (Bahdanau et al., 2015)."""

    def __init__(self, decoder_dim: int, encoder_dim: int) -> None:
        super().__init__()
        self.W_dec = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.W_enc = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(
        self,
        decoder_h: torch.Tensor,     # (B, decoder_dim)
        encoder_out: torch.Tensor,   # (B, S, encoder_dim)
        mask: torch.Tensor | None,   # (B, S) bool — True = valid position
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.v(
            torch.tanh(self.W_dec(decoder_h).unsqueeze(1) + self.W_enc(encoder_out))
        ).squeeze(-1)                # (B, S)
        if mask is not None:
            score = score.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(score, dim=-1)                    # (B, S)
        context = (weights.unsqueeze(-1) * encoder_out).sum(1)   # (B, encoder_dim)
        return context, weights


class VideoSummaryNet(nn.Module):
    """Bidirectional LSTM encoder + Bahdanau attention + LSTM decoder.

    Source and target share the same word-level vocabulary. The model is
    trained with teacher forcing on the CNN/DailyMail dataset
    (see videosum/nn/train_summary.py).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        encoder_dim = hidden_dim * 2  # BiLSTM concatenates fwd + bwd

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
        )
        self.enc_bridge = nn.Linear(encoder_dim, hidden_dim)

        self.attention = _BahdanauAttention(hidden_dim, encoder_dim)

        self.decoder = nn.LSTM(
            embed_dim + encoder_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _encode(
        self, src: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        enc_out, (h_n, _) = self.encoder(self.embedding(src))
        # h_n[-2]: last-layer forward, h_n[-1]: last-layer backward
        h0 = torch.tanh(self.enc_bridge(torch.cat([h_n[-2], h_n[-1]], dim=-1)))
        h0 = h0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return enc_out, (h0, torch.zeros_like(h0))

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        src:      (B, S)
        tgt:      (B, T)  BOS-prefixed target token indices
        src_mask: (B, S)  bool, True = valid token  (None → all valid)
        returns:  (B, T, vocab_size) logits
        """
        enc_out, hidden = self._encode(src)
        tgt_embed = self.embedding(tgt)        # (B, T, embed_dim)

        logits: list[torch.Tensor] = []
        for t in range(tgt_embed.size(1)):
            context, _ = self.attention(hidden[0][-1], enc_out, src_mask)
            dec_in = torch.cat([tgt_embed[:, t:t+1, :], context.unsqueeze(1)], dim=-1)
            out, hidden = self.decoder(dec_in, hidden)
            logits.append(self.out_proj(out[:, 0, :]))

        return torch.stack(logits, dim=1)      # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        text: str,
        tokenizer: Tokenizer,
        max_len: int = 200,
        min_len: int = 20,
    ) -> str:
        """Greedy-decode a summary for the given input text."""
        self.eval()
        device = next(self.parameters()).device

        src_ids = tokenizer.encode(text)
        if not src_ids:
            return ""

        src = torch.tensor([src_ids], device=device)
        src_mask = src.ne(tokenizer.pad_id)
        enc_out, hidden = self._encode(src)

        token = torch.tensor([[tokenizer.bos_id]], device=device)
        generated: list[int] = []

        for step in range(max_len):
            context, _ = self.attention(hidden[0][-1], enc_out, src_mask)
            dec_in = torch.cat([self.embedding(token), context.unsqueeze(1)], dim=-1)
            out, hidden = self.decoder(dec_in, hidden)
            logit = self.out_proj(out[:, 0, :])           # (1, vocab_size)

            if step < min_len:
                logit[0, tokenizer.eos_id] = float("-inf")

            next_id = int(logit.argmax(dim=-1))
            if next_id == tokenizer.eos_id:
                break
            generated.append(next_id)
            token = torch.tensor([[next_id]], device=device)

        return tokenizer.decode(generated)

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
    def load(cls, path: str | Path) -> tuple["VideoSummaryNet", Tokenizer | None]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        tokenizer = Tokenizer(payload["vocab"]) if payload.get("vocab") else None
        return model, tokenizer
