"""Compatibility wrapper for grouping and translation.

Run from project root:
    python src/group_and_translate.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "translation" / "translate.py"), run_name="__main__")
