"""Run the full project pipeline (as far as implemented).

Steps:
1. Prepare model inputs (copy raw -> model_input, preserve NULL)
2. Restore punctuation (ONNX, CPU) -> data/punctuated
3. Order sentences for all samples (candidate) -> data/ordered
4. Group every 5 ordered sentences and translate Hindi->English -> data/final

Usage (from project root):
    python scripts/run_full_pipeline.py

Notes:
- Requires dependencies in requirements/venv.
- Translation step may download Helsinki-NLP/opus-mt-hi-en on first run and cache it
  under models/translation/opus-mt-hi-en.
- This pipeline does not use GPT/Gemini/Claude.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(step: str, args: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"STEP: {step}")
    print("CMD:", " ".join(args))
    print("=" * 80, flush=True)
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"Step failed: {step} (exit code {proc.returncode})")


def main() -> int:
    run("Prepare input", [PY, "src/prepare_input.py"])
    run("Restore punctuation", [PY, "src/punctuation.py"])
    run("Order sentences (all samples)", [PY, "src/order_sentences.py"])
    run("Group (5) and translate", [PY, "src/group_and_translate.py"])
    print("\nPipeline completed. See data/punctuated, data/ordered, and data/final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
