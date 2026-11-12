"""
Diagnostic script: compare raw and punctuated contexts around every literal `NULL` marker.

Usage (from project root):
    python scripts/null_diagnostic.py

This does NOT modify any files. It helps investigate how punctuation restoration
treats NULL markers. The meaning of NULL is unknown - do not assume.
"""

from __future__ import annotations

import sys
from pathlib import Path


RAW_DIR = Path("data/raw/samples")
PUNCT_DIR = Path("data/punctuated")
CONTEXT = 40


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_all(haystack: str, needle: str = "NULL") -> list[int]:
    idxs = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            break
        idxs.append(i)
        start = i + len(needle)
    return idxs


def window(text: str, index: int, length: int = 4, ctx: int = CONTEXT) -> str:
    s = max(0, index - ctx)
    e = min(len(text), index + length + ctx)
    snippet = text[s:e]
    # make whitespace visible-ish
    snippet = snippet.replace("\r", "\\r").replace("\n", "\\n")
    return snippet


def nearest_nonspace(text: str, pos: int, direction: int) -> str | None:
    """direction: -1 for left, +1 for right. pos is index relative to NULL start/end."""
    i = pos
    while 0 <= i < len(text):
        ch = text[i]
        if not ch.isspace():
            return ch
        i += direction
    return None


def analyze_file(name: str) -> dict:
    raw_path = RAW_DIR / name
    punct_path = PUNCT_DIR / name
    raw = raw_path.read_text(encoding="utf-8")
    punct = punct_path.read_text(encoding="utf-8") if punct_path.exists() else ""

    raw_idxs = find_all(raw)
    punct_idxs = find_all(punct)

    result = {
        "file": name,
        "raw_count": len(raw_idxs),
        "punct_count": len(punct_idxs),
        "occurrences": [],
    }

    for k, (ri, pi) in enumerate(zip(raw_idxs, punct_idxs), start=1):
        # chars immediately adjacent (ignoring spaces) around marker in punctuated text
        left_char = nearest_nonspace(punct, pi - 1, -1)
        right_char = nearest_nonspace(punct, pi + 4, +1)
        raw_left = nearest_nonspace(raw, ri - 1, -1)
        raw_right = nearest_nonspace(raw, ri + 4, +1)
        result["occurrences"].append(
            {
                "n": k,
                "raw_left": raw_left,
                "raw_right": raw_right,
                "punct_left": left_char,
                "punct_right": right_char,
                "raw_ctx": window(raw, ri),
                "punct_ctx": window(punct, pi) if punct else "",
            }
        )
    return result


def main() -> int:
    _reconfigure_stdout()
    files = sorted(p.name for p in RAW_DIR.glob("*.txt"))
    if not files:
        print(f"No raw files found in {RAW_DIR}")
        return 1

    total_raw = 0
    total_punct = 0
    punct_after_colon = 0
    punct_after_danda = 0

    for name in files:
        r = analyze_file(name)
        total_raw += r["raw_count"]
        total_punct += r["punct_count"]
        print("=" * 80)
        print(f"{r['file']}: raw NULL={r['raw_count']} punctuated NULL={r['punct_count']}")
        if r["raw_count"] != r["punct_count"]:
            print("  WARNING: count mismatch - investigate alignment carefully!")
        for occ in r["occurrences"]:
            pr = occ["punct_right"]
            if pr == ":":
                punct_after_colon += 1
            if pr == "।":
                punct_after_danda += 1
            print(
                f"  #{occ['n']:02d} left(raw->punct): {occ['raw_left']!r} -> {occ['punct_left']!r} | "
                f"right(raw->punct): {occ['raw_right']!r} -> {occ['punct_right']!r}"
            )
            print(f"       RAW : ...{occ['raw_ctx']}...")
            if occ["punct_ctx"]:
                print(f"       PUNC: ...{occ['punct_ctx']}...")
        print()

    print("=" * 80)
    print(f"TOTAL literal NULL: raw={total_raw} punctuated={total_punct}")
    print(f"Punctuated contexts where next non-space after NULL is ':' : {punct_after_colon}")
    print(f"Punctuated contexts where next non-space after NULL is '।': {punct_after_danda}")
    print()
    print("NOTE: This is a diagnostic only. The meaning of NULL is unknown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
