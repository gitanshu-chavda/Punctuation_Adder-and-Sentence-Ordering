"""Profile basic runtime and memory for key pipeline scripts (best-effort, CPU-focused).

Uses tracemalloc for Python allocations and psutil (if available) for process RSS.
Writes reports/resource_profile.json.

Usage:
  python scripts/profile_resources.py
  python scripts/profile_resources.py --steps integrity,metrics
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def get_rss_mb():
    try:
        import psutil  # type: ignore
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def run_cmd(args):
    t0 = time.perf_counter()
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    return dt, proc.returncode, (proc.stdout or "")[-4000:], (proc.stderr or "")[-2000:]


def profile_step(name, args):
    tracemalloc.start()
    rss_before = get_rss_mb()
    dt, rc, out, err = run_cmd(args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = get_rss_mb()
    return {
        "step": name,
        "command": args,
        "seconds": round(dt, 3),
        "returncode": rc,
        "python_tracemalloc_peak_mb": round(peak / (1024 * 1024), 3),
        "rss_mb_before": round(rss_before, 3) if rss_before is not None else None,
        "rss_mb_after": round(rss_after, 3) if rss_after is not None else None,
        "stdout_tail": out,
        "stderr_tail": err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="integrity,metrics,generate_outputs", help="Comma-separated step keys")
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    available = {
        "integrity": [PY, "scripts/verify_integrity.py"],
        "metrics": [PY, "scripts/compute_metrics.py"],
        "generate_outputs": [PY, "scripts/generate_aligned_outputs.py"],
        "prepare": [PY, "src/prepare_input.py"],
        "punctuation": [PY, "src/punctuation.py"],
        "ordering": [PY, "src/order_sentences.py"],
        "translate": [PY, "src/group_and_translate.py"],
    }

    results = []
    for s in steps:
        if s not in available:
            results.append({"step": s, "error": "unknown step"})
            continue
        print(f"Profiling {s}...", flush=True)
        results.append(profile_step(s, available[s]))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": sys.platform,
        "python": sys.version,
        "cpu_only_pipeline_note": "Current scripts target CPU. GPU VRAM not measured in this profile; if using GPU, measure peak VRAM and keep < 1.5 GB.",
        "steps": results,
    }
    out = ROOT / "reports/resource_profile.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
