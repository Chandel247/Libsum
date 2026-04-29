import shutil
import tempfile
from pathlib import Path


class TempDir:
    def __init__(self):
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="videosum_"))
        return self.path

    def __exit__(self, *_):
        if self.path and self.path.exists():
            shutil.rmtree(self.path)


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def chunk_list(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]
