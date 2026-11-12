"""Basic environment and artifact checks for this repository."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def check(path: str) -> bool:
    ok = (ROOT / path).exists()
    print(("OK  " if ok else "MISS"), path)
    return ok


def main() -> int:
    print("Python:", sys.version)
    checks = [
        "data/raw/samples",
        "data/model_input",
        "data/punctuated",
        "data/ordered",
        "outputs",
        "reports",
        "models/punctuation/onnx/model_quantized.onnx",
        "models/punctuation/tokenizer",
        "src/punctuation.py",
        "src/order_sentences.py",
        "src/group_and_translate.py",
        "src/pipeline.py",
    ]
    ok = all(check(c) for c in checks)
    print("\nStatus:", "READY (basic checks passed)" if ok else "MISSING ITEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
