"""Compatibility wrapper for punctuation restoration.

Run from project root:
    python src/punctuation.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "punctuation" / "restore.py"), run_name="__main__")
