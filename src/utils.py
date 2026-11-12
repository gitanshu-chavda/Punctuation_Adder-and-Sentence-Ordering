"""Shared utility helpers for the pipeline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable


def safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 1 and sum(1 for ln in lines if re.search(r"[।?!]$", ln)) >= max(1, len(lines) - 2):
        return lines
    parts = re.split(r"(?<=[।?!])\s+", stripped)
    return [p.strip() for p in parts if p.strip()]


def chunked(items: Iterable, n: int):
    items = list(items)
    return [items[i : i + n] for i in range(0, len(items), n)]
