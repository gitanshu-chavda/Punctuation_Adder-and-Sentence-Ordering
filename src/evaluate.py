"""Generate/update basic evaluation and run reports for this repository.

This is a lightweight evaluator aligned to the current project artifacts.
It aggregates available proxy metrics and integrity checks into reports/evaluation.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # Run integrity and proxy punctuation evaluation scripts and capture outputs
    integrity = subprocess.run([PY, "scripts/verify_integrity.py"], cwd=ROOT, capture_output=True, text=True)
    punct = subprocess.run([PY, "scripts/evaluate_punctuation.py"], cwd=ROOT, capture_output=True, text=True)

    eval_path = reports / "evaluation.json"
    data = {}
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["integrity_check_stdout"] = integrity.stdout.strip()
    data["proxy_punctuation_eval_stdout"] = punct.stdout.strip()
    data["notes"] = (
        "Ordering and translation benchmark metrics (Kendall tau/pair accuracy/BLEU/chrF) "
        "are not computed automatically yet; manual/benchmark evaluation required."
    )
    eval_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
