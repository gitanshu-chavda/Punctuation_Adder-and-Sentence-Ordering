"""Analyze punctuation behavior around NULL markers and dialogue samples.

Provides concrete observations to guide manual review and cautious improvements.
Does NOT modify files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


RAW_DIR = Path("data/raw/samples")
PUNCT_DIR = Path("data/punctuated")
OUT = Path("reports/null_punct_analysis.json")


def find_all(haystack: str, needle: str = "NULL"):
    idxs = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            break
        idxs.append(i)
        start = i + len(needle)
    return idxs


def nearest_nonspace(text: str, pos: int, direction: int):
    i = pos
    while 0 <= i < len(text):
        ch = text[i]
        if not ch.isspace():
            return ch
        i += direction
    return None


def window(text: str, index: int, length: int = 4, ctx: int = 45):
    s = max(0, index - ctx)
    e = min(len(text), index + length + ctx)
    return text[s:e].replace("\r", "\\r").replace("\n", "\\n")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in RAW_DIR.glob("*.txt"))
    result = {"files": [], "summary": {}}
    total = 0
    colon_after = danda_after = punct_before = 0
    for name in files:
        raw = (RAW_DIR / name).read_text(encoding="utf-8")
        punct = (PUNCT_DIR / name).read_text(encoding="utf-8") if (PUNCT_DIR / name).exists() else ""
        raw_idxs = find_all(raw)
        punct_idxs = find_all(punct)
        occs = []
        for k, (ri, pi) in enumerate(zip(raw_idxs, punct_idxs), start=1):
            pl = nearest_nonspace(punct, pi - 1, -1)
            pr = nearest_nonspace(punct, pi + 4, +1)
            rl = nearest_nonspace(raw, ri - 1, -1)
            rr = nearest_nonspace(raw, ri + 4, +1)
            if pl and pl in "।,?:-":
                punct_before += 1
            if pr == ":":
                colon_after += 1
            if pr == "।":
                danda_after += 1
            occs.append({
                "n": k,
                "raw_left": rl, "raw_right": rr,
                "punct_left": pl, "punct_right": pr,
                "punct_ctx": window(punct, pi),
            })
            total += 1
        is_dialogue = name in ("sample-6.txt", "sample-7.txt")
        result["files"].append({"name": name, "dialogue_like": is_dialogue, "null_count": len(raw_idxs), "occurrences": occs})
    result["summary"] = {"total_null_checked": total, "punct_before_null": punct_before, "colon_immediately_after_null": colon_after, "danda_immediately_after_null": danda_after, "note": "Descriptive only; NULL semantics unknown - no automatic deletion/reinterpretation recommended."}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
