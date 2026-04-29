import json
import re
from collections import Counter
from pathlib import Path

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
_SPECIALS = [PAD, UNK, BOS, EOS]


class Tokenizer:
    """Word-level tokenizer shared by FrameCaptionNet and VideoSummaryNet.

    Vocabulary is built from a corpus of training texts; unknown words map
    to UNK. Special tokens occupy the first four indices so padding_idx=0
    works with nn.Embedding.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab: dict[str, int] = (
            vocab if vocab is not None
            else {tok: i for i, tok in enumerate(_SPECIALS)}
        )
        self.inv_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _split(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def build_vocab(
        self,
        texts: list[str],
        max_vocab: int = 10_000,
        min_freq: int = 2,
    ) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._split(text))
        top = [
            w for w, c in counter.most_common(max_vocab - len(_SPECIALS))
            if c >= min_freq
        ]
        self.vocab = {tok: i for i, tok in enumerate(_SPECIALS + top)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids = [self.vocab.get(t, self.unk_id) for t in self._split(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return " ".join(self.inv_vocab.get(i, UNK) for i in ids if i not in skip)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls(json.loads(Path(path).read_text()))
