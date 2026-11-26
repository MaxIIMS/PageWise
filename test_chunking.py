from local_rag.ingest import clean_text, split_records

def test_clean_text_normalizes_whitespace():
    assert clean_text("alpha   beta\n\n\n gamma") == "alpha beta\n\n gamma"

def test_split_records_preserves_page_metadata():
    text = "Это тестовый текст для проверки разбиения на фрагменты. " * 30
    chunks = split_records([(7, text)], "book.pdf", "Книга", 160, 20, 50, {})
    assert chunks
    assert all(item["metadata"]["page"] == 7 for item in chunks)
    assert all(item["metadata"]["book_title"] == "Книга" for item in chunks)
