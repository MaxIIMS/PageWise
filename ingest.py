"""Document parsing, chunking, embeddings and persistent indexing."""
from __future__ import annotations
import logging
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable
from bs4 import BeautifulSoup
from docx import Document
from ebooklib import ITEM_DOCUMENT, epub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from local_rag.config import Settings
from local_rag.security import load_security_rules, sanitize_document_text
from local_rag.utils import sha256_file

logger = logging.getLogger(__name__)
SUPPORTED = {".pdf", ".docx", ".txt", ".epub"}

def clean_text(text: str) -> str:
    """Normalize whitespace and remove common OCR noise conservatively."""
    text = text.replace("\x00", " ").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*(?:page\s+)?\d{1,4}\s*$", "", text, flags=re.I)
    return text.strip()

def parse_document(path: Path) -> list[tuple[int | None, str]]:
    """Parse a supported file into page/section records."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix == ".docx":
        doc = Document(str(path))
        return [(None, "\n".join(p.text for p in doc.paragraphs))]
    if suffix == ".txt":
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".epub":
        book = epub.read_epub(str(path))
        records = []
        for section, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT), start=1):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style", "iframe"]):
                tag.decompose()
            records.append((section, soup.get_text(" ", strip=True)))
        return records
    raise ValueError(f"Неподдерживаемый формат: {suffix}")

def split_records(records: Iterable[tuple[int | None, str]], source_file: str,
                  book_title: str, chunk_size: int, chunk_overlap: int,
                  min_chars: int, rules: dict[str, list[str]]) -> list[dict[str, object]]:
    """Split page/section records and preserve source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""], add_start_index=True
    )
    output = []
    for page, raw in records:
        safe = sanitize_document_text(clean_text(raw), rules)
        for doc in splitter.create_documents([safe]):
            text = doc.page_content.strip()
            if len(text) < min_chars:
                continue
            output.append({"text": text, "metadata": {
                "source_file": source_file, "page": int(page) if page is not None else -1,
                "chunk_id": str(uuid.uuid4()), "book_title": book_title
            }})
    return output

class IndexRegistry:
    """SQLite registry for file hashes and indexed sources."""
    def __init__(self, index_dir: Path) -> None:
        self.path = index_dir / "registry.sqlite3"
        index_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS books (
                source_file TEXT PRIMARY KEY, sha256 TEXT NOT NULL, book_title TEXT NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    def get_hash(self, source: str) -> str | None:
        """Get last indexed SHA-256 digest for a source."""
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT sha256 FROM books WHERE source_file=?", (source,)).fetchone()
        return str(row[0]) if row else None

    def save(self, source: str, digest: str, title: str) -> None:
        """Insert or update a registry entry."""
        with sqlite3.connect(self.path) as db:
            db.execute("""INSERT INTO books(source_file,sha256,book_title) VALUES(?,?,?)
                ON CONFLICT(source_file) DO UPDATE SET sha256=excluded.sha256,
                book_title=excluded.book_title,indexed_at=CURRENT_TIMESTAMP""",
                (source, digest, title))

    def list_books(self) -> list[tuple[str, str]]:
        """Return indexed source paths and titles."""
        with sqlite3.connect(self.path) as db:
            return list(db.execute("SELECT source_file,book_title FROM books ORDER BY book_title"))

    def remove(self, source: str) -> None:
        """Delete a source registry record."""
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM books WHERE source_file=?", (source,))

def _get_embeddings(settings: Settings):
    """Load SentenceTransformer locally and use CUDA when available."""
    import torch
    from sentence_transformers import SentenceTransformer
    requested = settings.models.embedding_device
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = "cpu"
    return SentenceTransformer(settings.models.embedding_model, device=device)

class LocalEmbeddings:
    """SentenceTransformer adapter with E5 query/document prefixes."""
    def __init__(self, model) -> None:
        self.model = model
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages."""
        return self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True).tolist()
    def embed_query(self, text: str) -> list[float]:
        """Embed a user query."""
        return self.model.encode(f"query: {text}", normalize_embeddings=True).tolist()

def ingest_path(path: Path, settings: Settings, force: bool = False) -> tuple[int, int]:
    """Index a file or directory; unchanged files are skipped by SHA-256."""
    from langchain_community.vectorstores import Chroma
    index_dir = settings.resolve(settings.paths.index_dir)
    registry = IndexRegistry(index_dir)
    rules = load_security_rules(settings.resolve(settings.paths.security_rules))
    candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    candidates = [p for p in candidates if p.suffix.lower() in SUPPORTED]
    if not candidates:
        raise ValueError("В указанном пути не найдено поддерживаемых файлов.")
    max_bytes = settings.security.max_file_size_mb * 1024 * 1024
    embeddings = LocalEmbeddings(_get_embeddings(settings))
    store = Chroma(collection_name=settings.retrieval.collection_name,
        persist_directory=str(index_dir), embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"})
    indexed = skipped = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn()) as progress:
        task = progress.add_task("Индексация", total=len(candidates))
        for file in candidates:
            try:
                if file.stat().st_size > max_bytes:
                    logger.warning("File skipped: configured size limit exceeded.")
                    skipped += 1
                    continue
                source = str(file.resolve())
                digest = sha256_file(file)
                if not force and registry.get_hash(source) == digest:
                    skipped += 1
                    continue
                records = parse_document(file)
                chunks = split_records(records, file.name, file.stem,
                    settings.retrieval.chunk_size, settings.retrieval.chunk_overlap,
                    settings.retrieval.min_chunk_chars, rules)
                if not chunks:
                    logger.warning("No extractable text in document.")
                    skipped += 1
                    continue
                old = store.get(where={"source_file": file.name})
                if old and old.get("ids"):
                    store.delete(ids=old["ids"])
                texts = [str(c["text"]) for c in chunks]
                metas = [c["metadata"] for c in chunks]
                batch = max(1, settings.models.embedding_batch_size)
                for start in range(0, len(texts), batch):
                    store.add_texts(texts[start:start+batch], metadatas=metas[start:start+batch])
                registry.save(source, digest, file.stem)
                indexed += 1
            except Exception as exc:
                logger.exception("Document indexing failed (%s).", type(exc).__name__)
                raise RuntimeError(f"Не удалось обработать файл {file.name}: {exc}") from exc
            finally:
                progress.advance(task)
    return indexed, skipped
