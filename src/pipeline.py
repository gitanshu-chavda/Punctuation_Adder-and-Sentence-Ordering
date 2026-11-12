"""Unified pipeline CLI for Hindi punctuation restoration, sentence ordering, grouping and translation.

Usage:
    python -m src.pipeline
    python -m src.pipeline --limit 1 --no-translate
    python -m src.pipeline --strategy borda  (ordering strategy hint)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_step(name: str, args: list[str]) -> None:
    print(f"\n{'='*80}\nSTEP: {name}\nCMD: {' '.join(args)}\n{'='*80}", flush=True)
    t0 = time.time()
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit code {proc.returncode})")
    print(f"Completed in {time.time()-t0:.1f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hindi Text Restoration and Translation Pipeline")
    parser.add_argument("--strategy", default="borda", help="Ordering strategy hint (e.g. borda)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples to process (0 = all)")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation step")
    parser.add_argument("--skip-prep", action="store_true", help="Skip prepare_input step")
    parser.add_argument("--skip-punct", action="store_true", help="Skip punctuation restoration")
    parser.add_argument("--skip-order", action="store_true", help="Skip sentence ordering")
    args = parser.parse_args(argv)

    print("Hindi Text Restoration and Translation Pipeline")
    print(f"Root: {ROOT}")
    print(f"Strategy: {args.strategy}")

    if not args.skip_prep:
        run_step("Prepare inputs (preserve NULL)", [PY, "src/prepare_input.py"])
    if not args.skip_punct:
        run_step("Punctuate (ONNX/CPU)", [PY, "src/punctuation.py"])
    if not args.skip_order:
        run_step("Order sentences (all samples)", [PY, "src/order_sentences.py"])
    if not args.no_translate:
        run_step("Group (5) and translate", [PY, "src/group_and_translate.py"])
        run_step("Generate aligned outputs (sentences_hi/en, groups_hi/en)", [PY, "scripts/generate_aligned_outputs.py"])

    # Update run summary
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "pipeline": "src/pipeline.py",
        "strategy": args.strategy,
        "args": vars(args),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": "Check outputs/ and data/ordered/ for results. Ordering currently uses implemented method (mBERT pairwise+Borda in src/order_sentences.py).",
    }
    (reports / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nPipeline finished. See reports/run_summary.json, outputs/, and data/final/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
