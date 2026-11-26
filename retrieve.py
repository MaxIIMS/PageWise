"""Chroma retrieval and bounded context assembly."""
from __future__ import annotations
from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from local_rag.config import Settings
from local_rag.ingest import LocalEmbeddings, _get_embeddings

@dataclass
class RetrievedChunk:
    """Retrieved text with its metadata and similarity score."""
    text: str
    metadata: dict
    score: float

def get_vectorstore(settings: Settings) -> Chroma:
    """Open the configured persistent Chroma collection."""
    return Chroma(collection_name=settings.retrieval.collection_name,
        persist_directory=str(settings.resolve(settings.paths.index_dir)),
        embedding_function=LocalEmbeddings(_get_embeddings(settings)),
        collection_metadata={"hnsw:space": "cosine"})

def retrieve(query: str, settings: Settings) -> list[RetrievedChunk]:
    """Return the top-k chunks ranked by cosine similarity."""
    store = get_vectorstore(settings)
    results = store.similarity_search_with_score(query, k=settings.retrieval.top_k)
    found = []
    for doc, distance in results:
        similarity = 1.0 - float(distance)
        if similarity >= settings.retrieval.min_similarity:
            found.append(RetrievedChunk(doc.page_content, doc.metadata, similarity))
    return found

def build_context(chunks: list[RetrievedChunk], settings: Settings) -> tuple[str, list[RetrievedChunk]]:
    """Wrap retrieved passages as untrusted data and enforce a context budget."""
    blocks, selected, used = [], [], 0
    for number, chunk in enumerate(chunks, start=1):
        snippet = chunk.text[:settings.retrieval.max_fragment_chars]
        page = chunk.metadata.get("page", -1)
        page_text = f"стр. {page}" if page and int(page) > 0 else "фрагмент"
        block = f'[{number}] (Книга: {chunk.metadata.get("book_title", "Неизвестно")}, {page_text}): "{snippet}"'
        if used + len(block) > settings.retrieval.max_context_chars:
            break
        blocks.append(block)
        selected.append(chunk)
        used += len(block)
    return "<context>\n" + "\n".join(blocks) + "\n</context>", selected
