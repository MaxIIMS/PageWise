"""Prompt-injection and output heuristics; not a complete security boundary."""
from __future__ import annotations
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)

def load_security_rules(path: Path) -> dict[str, list[str]]:
    """Load editable security patterns from YAML."""
    if not path.exists():
        return {}
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}

def _patterns(items: list[str]) -> list[re.Pattern[str]]:
    """Compile configured regex patterns, skipping invalid entries."""
    result = []
    for item in items:
        try:
            result.append(re.compile(item))
        except re.error:
            logger.warning("Invalid regex in security rules ignored.")
    return result

def _digest(text: str) -> str:
    """Return a SHA-256 hash without logging the original text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def sanitize_document_text(text: str, rules: dict[str, list[str]]) -> str:
    """Neutralize known prompt injection and remove control characters."""
    value = unicodedata.normalize("NFKC", text)
    for pattern in _patterns(rules.get("document_injection_patterns", [])):
        value = pattern.sub("[REMOVED_UNTRUSTED_INSTRUCTION]", value)
    value = "".join(c for c in value if c in "\n\t" or unicodedata.category(c) not in {"Cc", "Cf"})
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()

def input_guard(query: str, rules: dict[str, list[str]], max_chars: int = 4000) -> tuple[str, str | None]:
    """Reject configured jailbreaks and truncate oversized queries."""
    bounded = query[:max_chars]
    warning = None
    if len(query) > max_chars:
        logger.warning("Security event: query_truncated sha256=%s", _digest(query))
        warning = f"Запрос длиннее {max_chars} символов и был сокращён."
    for pattern in _patterns(rules.get("query_jailbreak_patterns", [])):
        if pattern.search(bounded):
            logger.warning("Security event: jailbreak_blocked sha256=%s", _digest(query))
            raise ValueError("Не могу обработать запрос: обнаружена попытка изменить системные инструкции.")
    if not bounded.strip():
        raise ValueError("Запрос не должен быть пустым.")
    return bounded.strip(), warning

def output_guard(answer: str, rules: dict[str, list[str]]) -> bool:
    """Return False when the answer matches a configured block pattern."""
    for pattern in _patterns(rules.get("output_block_patterns", [])):
        if pattern.search(answer):
            logger.warning("Security event: output_blocked sha256=%s", _digest(answer))
            return False
    return True
