"""Group ordered sentences into chunks of 5 and translate Hindi -> English.

Reads:  data/ordered/sample-*.ordered.txt
Writes: data/final/<stem>.groups_hi.txt      (groups of 5 Hindi sentences)
        data/final/<stem>.groups_en.txt      (translated groups)
        data/final/<stem>.groups.tsv         (group_id, hindi, english)
        data/final/<stem>.pipeline.json
        data/final/translation_summary.json

Translation model: Helsinki-NLP/opus-mt-hi-en (MarianMT encoder-decoder), run on CPU.
This is a neural machine translation model (not a GPT/Claude/Gemini-style chat LLM).
Model is cached locally under models/translation/opus-mt-hi-en when downloaded.

Run from project root:
    python src/group_and_translate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ORDERED_FOLDER = Path("data/ordered")
OUTPUT_FOLDER = Path("data/final")
MODEL_ID = "Helsinki-NLP/opus-mt-hi-en"
LOCAL_MODEL_DIR = Path("models/translation/opus-mt-hi-en")
GROUP_SIZE = 5
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 256


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_sentences(text: str) -> list[str]:
    """Extract ordered sentences from ordered text.

    Ordering scripts write one sentence per line. Prefer line-based splitting to
    preserve segments that may not end with sentence punctuation.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    parts = re.split(r"(?<=[।?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunked(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def load_translator():
    """Load translation model/tokenizer from local cache dir if available, else download."""
    source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_ID
    print(f"Loading translation model from: {source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source)
    model.to("cpu")
    model.eval()

    # If loaded from hub, save a local copy for reproducibility/offline reuse
    if source == MODEL_ID:
        try:
            LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(LOCAL_MODEL_DIR)
            model.save_pretrained(LOCAL_MODEL_DIR)
            print(f"Saved local translation model copy to: {LOCAL_MODEL_DIR}")
        except Exception as e:
            print(f"Warning: could not save local model copy: {e}")

    return tokenizer, model


def translate_texts(tokenizer, model, texts: list[str]) -> list[str]:
    outputs = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
            )
        out = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.append(out[0].strip() if out else "")
    return outputs


def process_file(tokenizer, model, ordered_file: Path, output_folder: Path) -> dict:
    text = ordered_file.read_text(encoding="utf-8")
    sentences = extract_sentences(text)
    groups = chunked(sentences, GROUP_SIZE)
    hi_groups = [" ".join(g) for g in groups]
    en_groups = translate_texts(tokenizer, model, hi_groups)

    stem = ordered_file.stem.replace(".ordered", "")
    hi_path = output_folder / f"{stem}.groups_hi.txt"
    en_path = output_folder / f"{stem}.groups_en.txt"
    tsv_path = output_folder / f"{stem}.groups.tsv"
    json_path = output_folder / f"{stem}.pipeline.json"

    def fmt_groups(groups_text):
        lines = []
        for i, g in enumerate(groups_text, 1):
            lines.append(f"[Group {i}]")
            lines.append(g)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    hi_path.write_text(fmt_groups(hi_groups), encoding="utf-8")
    en_path.write_text(fmt_groups(en_groups), encoding="utf-8")

    tsv_lines = ["group_id\tsentence_count\thindi\tenglish"]
    for i, (hi, en, g) in enumerate(zip(hi_groups, en_groups, groups), 1):
        hi_clean = hi.replace("\t", " ").replace("\n", " ")
        en_clean = en.replace("\t", " ").replace("\n", " ")
        tsv_lines.append(f"{i}\t{len(g)}\t{hi_clean}\t{en_clean}")
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    stats = {
        "ordered_input": str(ordered_file),
        "sentences": len(sentences),
        "group_size": GROUP_SIZE,
        "groups": len(groups),
        "group_sentence_counts": [len(g) for g in groups],
        "translation_model": MODEL_ID,
        "outputs": {
            "hindi_groups": str(hi_path),
            "english_groups": str(en_path),
            "tsv": str(tsv_path),
        },
    }
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    _safe_stdout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    ordered_files = sorted(ORDERED_FOLDER.glob("sample-*.ordered.txt"))
    if not ordered_files:
        print(f"No ordered files found in {ORDERED_FOLDER}. Run: python src/order_sentences.py")
        return 1

    tokenizer, model = load_translator()

    results = []
    for f in ordered_files:
        print(f"\nGrouping & translating {f.name} ...", flush=True)
        stats = process_file(tokenizer, model, f, OUTPUT_FOLDER)
        results.append(stats)
        print(f"  sentences={stats['sentences']} groups={stats['groups']} -> {stats['outputs']['english_groups']}")

    summary = {
        "translation_model": MODEL_ID,
        "local_model_dir": str(LOCAL_MODEL_DIR),
        "group_size": GROUP_SIZE,
        "files": results,
    }
    (OUTPUT_FOLDER / "translation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nDone. Processed {len(results)} files. Summary: {OUTPUT_FOLDER / 'translation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
