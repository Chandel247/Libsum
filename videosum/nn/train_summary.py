#!/usr/bin/env python3
"""Train VideoSummaryNet on the CNN/DailyMail summarization dataset.

The dataset is downloaded automatically via HuggingFace datasets on first run.

Usage:
    pip install -e ".[neural]"
    python -m videosum.nn.train_summary \\
        --output videosum/nn/weights/summary.pt \\
        --epochs 5 --batch-size 32
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .tokenizer import Tokenizer
from .video_summary_net import VideoSummaryNet

logger = logging.getLogger(__name__)


def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the neural extra: pip install -e '.[neural]'"
        ) from exc
    return load_dataset("cnn_dailymail", "3.0.0")


class SummarizationDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer: Tokenizer,
        src_max: int = 400,
        tgt_max: int = 100,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.src_max = src_max
        self.tgt_max = tgt_max

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        tok = self.tokenizer
        return (
            torch.tensor(tok.encode(item["article"])[:self.src_max], dtype=torch.long),
            torch.tensor(tok.encode(item["highlights"], add_bos=True)[:self.tgt_max], dtype=torch.long),
            torch.tensor(tok.encode(item["highlights"], add_eos=True)[:self.tgt_max], dtype=torch.long),
        )


def _pad_collate(pad_id: int):
    def collate(batch):
        srcs, tgts_in, tgts_out = zip(*batch)

        def pad(seqs):
            max_len = max(s.size(0) for s in seqs)
            out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :s.size(0)] = s
            return out

        return pad(srcs), pad(tgts_in), pad(tgts_out)

    return collate


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading CNN/DailyMail ...")
    raw = _load_dataset()
    train_samples = list(raw["train"])
    val_samples = list(raw["validation"])

    logger.info("Building vocabulary (%d articles) ...", len(train_samples))
    tokenizer = Tokenizer()
    tokenizer.build_vocab(
        [s["article"] + " " + s["highlights"] for s in train_samples],
        max_vocab=args.vocab_size,
    )
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    train_ds = SummarizationDataset(train_samples, tokenizer, args.src_max, args.tgt_max)
    val_ds = SummarizationDataset(val_samples, tokenizer, args.src_max, args.tgt_max)
    collate_fn = _pad_collate(tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    model = VideoSummaryNet(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_loss = float("inf")
    output_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for src, tgt_in, tgt_out in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in, src.ne(tokenizer.pad_id))
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), tgt_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt_in, tgt_out in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                logits = model(src, tgt_in, src.ne(tokenizer.pad_id))
                val_loss += criterion(
                    logits.reshape(-1, tokenizer.vocab_size), tgt_out.reshape(-1)
                ).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        logger.info("Epoch %d  train=%.4f  val=%.4f", epoch, avg_train, avg_val)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save(output_path, tokenizer)
            logger.info("Saved → %s", output_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Train VideoSummaryNet on CNN/DailyMail")
    p.add_argument("--output", default="videosum/nn/weights/summary.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab-size", type=int, default=30_000)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--src-max", type=int, default=400)
    p.add_argument("--tgt-max", type=int, default=100)
    train(p.parse_args())


if __name__ == "__main__":
    main()
