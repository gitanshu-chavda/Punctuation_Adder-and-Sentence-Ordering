"""Compatibility wrapper for sentence ordering.

Run from project root:
    python src/order_sentences.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "ordering" / "order.py"), run_name="__main__")
