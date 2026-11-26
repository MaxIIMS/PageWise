"""Logging and filesystem helpers."""
from __future__ import annotations
import logging
from pathlib import Path
from rich.logging import RichHandler

def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Set up console and privacy-conscious file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(RichHandler(rich_tracebacks=True))

def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 digest in bounded-memory blocks."""
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def ensure_data_dirs(*paths: Path) -> None:
    """Create configured data directories."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
