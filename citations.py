"""Citation extraction and source formatting."""
from __future__ import annotations
import re
from local_rag.retrieve import RetrievedChunk

def cited_indices(answer: str) -> list[int]:
    """Extract unique citation indices in order of appearance."""
    return list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", answer)))

def format_sources(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Format citations that refer to retrieved chunks."""
    indices = cited_indices(answer)
    if not indices:
        return "Источники:\n  Модель не указала номера фрагментов."
    lines = ["Источники:"]
    for index in indices:
        if not 1 <= index <= len(chunks):
            continue
        chunk = chunks[index - 1]
        meta = chunk.metadata
        page = meta.get("page", -1)
        page_text = f"стр. {page}" if page and int(page) > 0 else "страница не определена"
        snippet = chunk.text.replace("\n", " ")[:220]
        lines.append(f'  [{index}] {meta.get("book_title", "Неизвестно")} — {page_text}\n      {snippet}')
    return "\n".join(lines)
