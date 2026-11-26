# Local RAG

Локальный RAG-ассистент для PDF, DOCX, TXT и EPUB. Документы, эмбеддинги, ChromaDB и генерация через Ollama остаются на компьютере. Облачные API не используются. Работает оффлайн. 

## Требования
- Windows 11 / macOS / Linux
- Python 3.11+
- Ollama
- Для конфигурации по умолчанию рекомендуется 16 ГБ RAM

## Установка Windows 11 (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```
Если активация запрещена:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Установите Ollama с https://ollama.com/download, запустите приложение и выполните:
```powershell
ollama pull qwen2.5:7b-instruct
```

Положите книги в `data/books/`, затем:
```powershell
rag ingest .\data\books
rag list
rag ask "Какие основные идеи изложены в книге?"
rag chat
rag remove "имя книги"
rag config show
rag config set retrieval.top_k 10
pytest -q
```

Первый запуск embedding-модели скачивает её из Hugging Face. Для подготовки локального кэша заранее:
```powershell
rag models-warmup
```
После загрузки моделей и зависимостей можно отключить интернет. Ollama вызывается только через localhost.

## Настройка
Редактируйте `config.yaml`. На 8 ГБ RAM задайте `llm_model: qwen2.5:3b`, `embedding_batch_size: 4`, `max_context_chars: 5000`. На 16 ГБ оставлена 7B модель. `embedding_device: auto` выбирает CUDA, если PyTorch её видит, иначе CPU. Ollama самостоятельно выбирает GPU для LLM.

## Источники и ограничения
PDF хранит номера страниц. Для DOCX/TXT страницы отсутствуют; EPUB использует номер XHTML-раздела. Сканированные PDF без текстового слоя требуют OCR, который не включён. Фильтры безопасности — эвристические, не абсолютная гарантия. В логах не сохраняется содержимое запросов/книг.

## Удаление данных
Остановите приложение и удалите содержимое `data/index/`, `data/books/` и `logs/`.
