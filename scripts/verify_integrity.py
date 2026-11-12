"""Verify basic integrity across pipeline stages.

Checks (per sample where files exist):
- Literal `NULL` count preserved (raw vs model_input vs punctuated vs ordered vs final hi groups)
- Rough token counts and character counts
- Sentence/group counts from generated metadata

Usage:
    python scripts/verify_integrity.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def count_null(text: str) -> int:
    return text.count("NULL")


def rough_tokens(text: str) -> int:
    # strip common punctuation marks used in this project for a rough comparison
    t = re.sub(r"[.,;:!?。，、；：！？؟،¿¡…।॥۔።፣፧\-]", " ", text)
    return len([w for w in t.split() if w])


def read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def main() -> int:
    _safe_stdout()
    samples = [f"sample-{i}" for i in range(1, 8)]
    print("sample,raw_NULL,model_input_NULL,punctuated_NULL,ordered_NULL,final_hi_NULL,raw_chars,punct_chars,ordered_chars,ordered_sentences,groups")
    for s in samples:
        raw = read(Path(f"data/raw/samples/{s}.txt"))
        mi = read(Path(f"data/model_input/{s}.txt"))
        pu = read(Path(f"data/punctuated/{s}.txt"))
        od = read(Path(f"data/ordered/{s}.ordered.txt"))
        fh = read(Path(f"data/final/{s}.groups_hi.txt"))
        stats_path = Path(f"data/ordered/{s}.order_stats.json")
        pipe_path = Path(f"data/final/{s}.pipeline.json")
        ord_sent = groups = ""
        if stats_path.exists():
            try:
                ord_sent = json.loads(stats_path.read_text(encoding="utf-8")).get("sentences", "")
            except Exception:
                pass
        if pipe_path.exists():
            try:
                groups = json.loads(pipe_path.read_text(encoding="utf-8")).get("groups", "")
            except Exception:
                pass
        def n(v): return "" if v is None else str(count_null(v))
        def c(v): return "" if v is None else str(len(v))
        print(
            f"{s},{n(raw)},{n(mi)},{n(pu)},{n(od)},{n(fh)},"
            f"{c(raw)},{c(pu)},{c(od)},{ord_sent},{groups}"
        )
    print("\nNote: This is a quick integrity check. Always inspect outputs manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
