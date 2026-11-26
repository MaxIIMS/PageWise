"""Typer CLI for local RAG."""
from __future__ import annotations
import logging
import time
from collections import deque
from pathlib import Path
import typer
import yaml
from rich.console import Console
from rich.table import Table
from local_rag.citations import format_sources
from local_rag.config import Settings
from local_rag.generate import stream_answer
from local_rag.ingest import IndexRegistry, ingest_path
from local_rag.retrieve import retrieve
from local_rag.security import input_guard, load_security_rules, output_guard
from local_rag.utils import ensure_data_dirs, setup_logging

app = typer.Typer(help="Локальный RAG-ассистент по вашим книгам.")
config_app = typer.Typer(help="Настройки приложения.")
app.add_typer(config_app, name="config")
console = Console()
logger = logging.getLogger(__name__)
_request_times: deque[float] = deque()

def _settings() -> Settings:
    """Load settings and prepare configured local directories."""
    cfg = Settings.load()
    ensure_data_dirs(cfg.resolve(cfg.paths.books_dir), cfg.resolve(cfg.paths.index_dir), cfg.resolve(cfg.paths.logs_dir))
    setup_logging(cfg.resolve(cfg.paths.logs_dir), cfg.app.log_level)
    return cfg

def _guard(query: str, cfg: Settings) -> str:
    """Run the query guard and show a truncation warning if needed."""
    rules = load_security_rules(cfg.resolve(cfg.paths.security_rules))
    try:
        safe, warning = input_guard(query, rules, cfg.security.max_query_chars)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")
    return safe

def _rate_limit(cfg: Settings) -> None:
    """Limit interactive requests to a configured number per rolling minute."""
    now = time.monotonic()
    while _request_times and now - _request_times[0] > 60:
        _request_times.popleft()
    if len(_request_times) >= cfg.security.requests_per_minute:
        raise RuntimeError("Достигнут лимит запросов в минуту. Подождите немного.")
    _request_times.append(now)

def _answer(query: str, cfg: Settings, limited: bool = False) -> None:
    """Retrieve context, stream the model response, filter it and show citations."""
    if limited:
        _rate_limit(cfg)
    safe_query = _guard(query, cfg)
    chunks = retrieve(safe_query, cfg)
    if not chunks:
        console.print("В предоставленных материалах ответа нет.")
        return
    rules = load_security_rules(cfg.resolve(cfg.paths.security_rules))
    # Buffer the stream until output checks pass: unsafe partial text must not reach the terminal.
    tokens = list(stream_answer(safe_query, chunks, cfg))
    answer = "".join(tokens)
    if cfg.security.block_sensitive_output and not output_guard(answer, rules):
        console.print("[red]Не могу ответить на этот запрос.[/red]")
        return
    console.print(answer)
    console.print()
    console.print(format_sources(answer, chunks))

@app.command()
def ingest(path: Path = typer.Argument(..., exists=True, readable=True),
           force: bool = typer.Option(False, "--force", help="Переиндексировать неизменённые файлы.")) -> None:
    """Index a file or a directory."""
    cfg = _settings()
    try:
        count, skipped = ingest_path(path, cfg, force)
        console.print(f"[green]Готово. Проиндексировано: {count}; пропущено: {skipped}.[/green]")
    except Exception as exc:
        console.print(f"[red]Ошибка индексации: {exc}[/red]")
        raise typer.Exit(code=1) from exc

@app.command()
def ask(query: str = typer.Argument(..., help="Вопрос к библиотеке документов.")) -> None:
    """Ask one question and stream the local model answer."""
    cfg = _settings()
    try:
        _answer(query, cfg)
    except Exception as exc:
        console.print(f"[red]Ошибка: {exc}[/red]")
        raise typer.Exit(code=1) from exc

@app.command()
def chat() -> None:
    """Start interactive chat mode."""
    cfg = _settings()
    console.print("[bold]Локальный RAG-чат. Для выхода введите /exit.[/bold]")
    while True:
        try:
            query = console.input("\n[cyan]Вы[/cyan]: ").strip()
            if query.lower() in {"/exit", "exit", "quit"}:
                break
            if query:
                try:
                    _answer(query, cfg, limited=True)
                except Exception as exc:
                    console.print(f"[red]Ошибка: {exc}[/red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\nЗавершение чата.")
            break

@app.command("list")
def list_books() -> None:
    """List indexed documents."""
    cfg = _settings()
    rows = IndexRegistry(cfg.resolve(cfg.paths.index_dir)).list_books()
    table = Table("Файл", "Название")
    for source, title in rows:
        table.add_row(source, title)
    console.print(table if rows else "Индекс пока пуст.")

@app.command()
def remove(book: str = typer.Argument(..., help="Имя файла или часть названия.")) -> None:
    """Remove matching books and their vector chunks."""
    from local_rag.retrieve import get_vectorstore
    cfg = _settings()
    registry = IndexRegistry(cfg.resolve(cfg.paths.index_dir))
    matches = [(src, title) for src, title in registry.list_books()
               if book.lower() in src.lower() or book.lower() in title.lower()]
    if not matches:
        console.print("[yellow]Книга не найдена.[/yellow]")
        raise typer.Exit(code=1)
    store = get_vectorstore(cfg)
    for source, _ in matches:
        found = store.get(where={"source_file": Path(source).name})
        if found and found.get("ids"):
            store.delete(ids=found["ids"])
        registry.remove(source)
    console.print(f"[green]Удалено записей: {len(matches)}.[/green]")

@config_app.command("show")
def config_show() -> None:
    """Show validated configuration."""
    cfg = _settings()
    console.print(yaml.safe_dump(cfg.model_dump(), allow_unicode=True, sort_keys=False))

@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Update one dotted key in config.yaml."""
    path = Path("config.yaml")
    if not path.exists():
        console.print("[red]config.yaml не найден.[/red]")
        raise typer.Exit(code=1)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            console.print(f"[red]Неизвестная настройка: {key}[/red]")
            raise typer.Exit(code=2)
        node = node[part]
    if parts[-1] not in node:
        console.print(f"[red]Неизвестная настройка: {key}[/red]")
        raise typer.Exit(code=2)
    old = node[parts[-1]]
    if isinstance(old, bool):
        new_value = value.lower() in {"true", "1", "yes", "on"}
    elif isinstance(old, int):
        new_value = int(value)
    elif isinstance(old, float):
        new_value = float(value)
    else:
        new_value = value
    node[parts[-1]] = new_value
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    console.print(f"[green]Настройка {key} обновлена.[/green]")

@app.command("models-warmup")
def models_warmup() -> None:
    """Download/cache the configured embedding model for offline use."""
    cfg = _settings()
    from local_rag.ingest import _get_embeddings
    console.print("Подготовка embedding-модели...")
    _get_embeddings(cfg)
    console.print("[green]Embedding-модель готова.[/green]")
