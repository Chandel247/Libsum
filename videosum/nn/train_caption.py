#!/usr/bin/env python3
"""Train FrameCaptionNet on the MS COCO 2017 captions dataset.

Download COCO first:
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip each into a shared --coco-root directory so the layout is:
        <coco-root>/train2017/
        <coco-root>/val2017/
        <coco-root>/annotations/captions_train2017.json
        <coco-root>/annotations/captions_val2017.json

Usage:
    pip install -e ".[neural]"
    python -m videosum.nn.train_caption \\
        --coco-root /data/coco \\
        --output videosum/nn/weights/caption.pt \\
        --epochs 10 --batch-size 64
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .frame_caption_net import FrameCaptionNet
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class COCOCaptionsDataset(Dataset):
    def __init__(
        self, root: Path, split: str, tokenizer: Tokenizer, max_len: int = 30
    ) -> None:
        ann_file = root / "annotations" / f"captions_{split}2017.json"
        with ann_file.open() as f:
            data = json.load(f)

        id_to_path = {
            img["id"]: root / f"{split}2017" / img["file_name"]
            for img in data["images"]
        }

        self.samples: list[tuple[Path, list[int], list[int]]] = []
        for ann in data["annotations"]:
            img_path = id_to_path.get(ann["image_id"])
            if img_path is None or not img_path.exists():
                continue
            caption = ann["caption"]
            self.samples.append((
                img_path,
                tokenizer.encode(caption, add_bos=True)[:max_len],
                tokenizer.encode(caption, add_eos=True)[:max_len],
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, src, tgt = self.samples[idx]
        img = _TRANSFORM(Image.open(img_path).convert("RGB"))
        return img, torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def _collate(batch):
    imgs, srcs, tgts = zip(*batch)
    max_len = max(s.size(0) for s in srcs)
    srcs_pad = torch.zeros(len(srcs), max_len, dtype=torch.long)
    tgts_pad = torch.zeros(len(tgts), max_len, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        srcs_pad[i, :s.size(0)] = s
        tgts_pad[i, :t.size(0)] = t
    return torch.stack(imgs), srcs_pad, tgts_pad


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    coco_root = Path(args.coco_root)
    ann_file = coco_root / "annotations" / "captions_train2017.json"
    with ann_file.open() as f:
        train_data = json.load(f)

    logger.info("Building vocabulary ...")
    tokenizer = Tokenizer()
    tokenizer.build_vocab(
        [a["caption"] for a in train_data["annotations"]],
        max_vocab=args.vocab_size,
    )
    logger.info("Vocab size: %d", tokenizer.vocab_size)

    train_ds = COCOCaptionsDataset(coco_root, "train", tokenizer, args.max_caption_len)
    val_ds = COCOCaptionsDataset(coco_root, "val", tokenizer, args.max_caption_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=_collate,
    )

    model = FrameCaptionNet(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float("inf")
    output_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, caps_in, caps_out in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            imgs, caps_in, caps_out = imgs.to(device), caps_in.to(device), caps_out.to(device)
            logits = model(imgs, caps_in)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), caps_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, caps_in, caps_out in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                imgs, caps_in, caps_out = imgs.to(device), caps_in.to(device), caps_out.to(device)
                logits = model(imgs, caps_in)
                val_loss += criterion(
                    logits.reshape(-1, tokenizer.vocab_size), caps_out.reshape(-1)
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
    p = argparse.ArgumentParser(description="Train FrameCaptionNet on MS COCO")
    p.add_argument("--coco-root", required=True)
    p.add_argument("--output", default="videosum/nn/weights/caption.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--max-caption-len", type=int, default=30)
    train(p.parse_args())


if __name__ == "__main__":
    main()
