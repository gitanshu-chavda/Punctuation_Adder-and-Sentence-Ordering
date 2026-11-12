"""Generate friend-style aligned outputs in outputs/.

Creates per sample:
- outputs/sample-N.sentences_hi.txt  (one ordered Hindi sentence per line)
- outputs/sample-N.groups_hi.txt     (five Hindi sentences per line)
- outputs/sample-N.sentences_en.txt  (one English translation per line, aligned)
- outputs/sample-N.groups_en.txt     (five English translations per line, aligned)

Uses src.group_and_translate.load_translator() so backend follows the same config
(IndicTrans2 preferred when available, else MarianMT fallback).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.group_and_translate import extract_sentences, chunked, load_translator  # noqa: E402


ORDERED_DIR = ROOT / "data" / "ordered"
OUT_DIR = ROOT / "outputs"
GROUP_SIZE = 5


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(ORDERED_DIR.glob("sample-*.ordered.txt"))
    if not files:
        print("No ordered files found. Run src/order_sentences.py first.")
        return 1

    translator = load_translator()
    print(f"Using translation backend: {translator.backend} ({translator.model_name})", flush=True)

    for f in files:
        sample = f.stem.replace(".ordered", "")
        print(f"\nProcessing {sample}...", flush=True)
        text = f.read_text(encoding="utf-8")
        hi_sentences = extract_sentences(text)
        en_sentences = translator.translate_texts(hi_sentences)
        if len(en_sentences) != len(hi_sentences):
            print(f"WARNING: sentence count mismatch {len(hi_sentences)} vs {len(en_sentences)}")
            # pad/trim defensively
            if len(en_sentences) < len(hi_sentences):
                en_sentences += [""] * (len(hi_sentences) - len(en_sentences))
            else:
                en_sentences = en_sentences[: len(hi_sentences)]

        hi_groups = [" ".join(g) for g in chunked(hi_sentences, GROUP_SIZE)]
        en_groups = [" ".join(g) for g in chunked(en_sentences, GROUP_SIZE)]

        (OUT_DIR / f"{sample}.sentences_hi.txt").write_text("\n".join(hi_sentences) + "\n", encoding="utf-8")
        (OUT_DIR / f"{sample}.sentences_en.txt").write_text("\n".join(en_sentences) + "\n", encoding="utf-8")
        (OUT_DIR / f"{sample}.groups_hi.txt").write_text("\n".join(hi_groups) + "\n", encoding="utf-8")
        (OUT_DIR / f"{sample}.groups_en.txt").write_text("\n".join(en_groups) + "\n", encoding="utf-8")
        print(f"  wrote 4 aligned files for {sample} (sentences={len(hi_sentences)}, groups={len(hi_groups)})")

    print(f"\nDone. Outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
