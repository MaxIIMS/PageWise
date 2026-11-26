from local_rag.config import Settings
from local_rag.retrieve import RetrievedChunk, build_context

def test_context_is_wrapped():
    cfg = Settings()
    chunk = RetrievedChunk("Текст источника " * 10, {"book_title": "Тест", "page": 4}, 0.9)
    context, selected = build_context([chunk], cfg)
    assert context.startswith("<context>")
    assert context.endswith("</context>")
    assert "[1]" in context
    assert len(selected) == 1
