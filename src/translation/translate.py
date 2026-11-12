"""Group ordered sentences into chunks of 5 and translate Hindi -> English.

Reads:  data/ordered/sample-*.ordered.txt
Writes: data/final/<stem>.groups_hi.txt
        data/final/<stem>.groups_en.txt
        data/final/<stem>.groups.tsv
        data/final/<stem>.pipeline.json
        data/final/translation_summary.json

Translation backends (in priority order):
1. IndicTrans2 (AI4Bharat) when available/configured (preferred per project update request).
   - Requires model access and dependencies (e.g. IndicTransToolkit) and may require HF_TOKEN
     for gated models. Model ID can be set via INDICTRANS2_MODEL_ID.
2. Fallback: Helsinki-NLP/opus-mt-hi-en (MarianMT), not a GPT/Claude/Gemini chat LLM.

Run from project root:
    python src/group_and_translate.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ORDERED_FOLDER = Path("data/ordered")
OUTPUT_FOLDER = Path("data/final")
GROUP_SIZE = 5
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 256

# Preferred (IndicTrans2) - user-requested
INDIC_DEFAULT_MODEL_ID = os.environ.get("INDICTRANS2_MODEL_ID", "ai4bharat/indictrans2-indic-en-dist-200M")
INDIC_LOCAL_DIR = Path(os.environ.get("INDICTRANS2_LOCAL_DIR", "models/translation/indictrans2-indic-en"))

# Fallback (MarianMT)
MARIAN_MODEL_ID = "Helsinki-NLP/opus-mt-hi-en"
MARIAN_LOCAL_DIR = Path("models/translation/opus-mt-hi-en")


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_sentences(text: str) -> list[str]:
    """Extract ordered sentences (one per line preferred)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) > 1:
        return lines
    parts = re.split(r"(?<=[।?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunked(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


class BaseTranslator:
    backend = "base"
    model_name = ""

    def translate_texts(self, texts: list[str]) -> list[str]:
        raise NotImplementedError


class MarianTranslator(BaseTranslator):
    backend = "marian"

    def __init__(self):
        source = str(MARIAN_LOCAL_DIR) if MARIAN_LOCAL_DIR.exists() else MARIAN_MODEL_ID
        self.model_name = source
        print(f"Loading translation model (fallback MarianMT): {source}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source)
        self.model.to("cpu")
        self.model.eval()
        if source == MARIAN_MODEL_ID:
            try:
                MARIAN_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save_pretrained(MARIAN_LOCAL_DIR)
                self.model.save_pretrained(MARIAN_LOCAL_DIR)
                print(f"Saved local MarianMT copy to: {MARIAN_LOCAL_DIR}")
            except Exception as e:
                print(f"Warning: could not save local MarianMT copy: {e}")

    @torch.no_grad()
    def translate_texts(self, texts: list[str]) -> list[str]:
        outputs = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True,
            )
            out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.append(out[0].strip() if out else "")
        return outputs


class IndicTrans2Translator(BaseTranslator):
    backend = "indictrans2"

    def __init__(self):
        # Try to use IndicTransToolkit preprocessing/postprocessing if available (recommended by AI4Bharat).
        try:
            from IndicTransToolkit.processor import IndicProcessor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "IndicTrans2 requested but IndicTransToolkit is not available. "
                "Install IndicTransToolkit and ensure model access/token if gated."
            ) from e

        source = str(INDIC_LOCAL_DIR) if INDIC_LOCAL_DIR.exists() else INDIC_DEFAULT_MODEL_ID
        self.model_name = source
        print(f"Loading translation model (IndicTrans2): {source}", flush=True)
        self.ip = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, trust_remote_code=True)
        self.model.to("cpu")
        self.model.eval()

        if source == INDIC_DEFAULT_MODEL_ID:
            try:
                INDIC_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save_pretrained(INDIC_LOCAL_DIR)
                self.model.save_pretrained(INDIC_LOCAL_DIR)
                print(f"Saved local IndicTrans2 copy to: {INDIC_LOCAL_DIR}")
            except Exception as e:
                print(f"Warning: could not save local IndicTrans2 copy: {e}")

    @torch.no_grad()
    def translate_texts(self, texts: list[str]) -> list[str]:
        # IndicTrans2 expects specific language tags and preprocessing; use common hin_Deva -> eng_Latn.
        src_lang, tgt_lang = "hin_Deva", "eng_Latn"
        batch = self.ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        generated = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            early_stopping=True,
        )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs = self.ip.postprocess_batch(decoded, lang=tgt_lang)
        return [o.strip() for o in outputs]


def load_translator() -> BaseTranslator:
    """Prefer IndicTrans2 when possible, else fall back to MarianMT."""
    prefer = os.environ.get("TRANSLATION_BACKEND", "auto").lower()  # auto|indictrans2|marian
    errors = []

    if prefer in ("auto", "indictrans2"):
        try:
            return IndicTrans2Translator()
        except Exception as e:
            errors.append(f"IndicTrans2 unavailable: {e}")
            if prefer == "indictrans2":
                raise

    if prefer in ("auto", "marian"):
        try:
            return MarianTranslator()
        except Exception as e:
            errors.append(f"MarianMT unavailable: {e}")
            raise

    raise RuntimeError("No translation backend available. " + " | ".join(errors))


def process_file(translator: BaseTranslator, ordered_file: Path, output_folder: Path) -> dict:
    text = ordered_file.read_text(encoding="utf-8")
    sentences = extract_sentences(text)
    groups = chunked(sentences, GROUP_SIZE)
    hi_groups = [" ".join(g) for g in groups]
    en_groups = translator.translate_texts(hi_groups)

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
        tsv_lines.append(
            f"{i}\t{len(g)}\t{hi.replace(chr(9),' ').replace(chr(10),' ')}\t{en.replace(chr(9),' ').replace(chr(10),' ')}"
        )
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    stats = {
        "ordered_input": str(ordered_file),
        "sentences": len(sentences),
        "group_size": GROUP_SIZE,
        "groups": len(groups),
        "group_sentence_counts": [len(g) for g in groups],
        "translation_backend": translator.backend,
        "translation_model": translator.model_name,
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

    translator = load_translator()
    print(f"Using translation backend: {translator.backend} ({translator.model_name})", flush=True)

    results = []
    for f in ordered_files:
        print(f"\nGrouping & translating {f.name} ...", flush=True)
        stats = process_file(translator, f, OUTPUT_FOLDER)
        results.append(stats)
        print(f"  sentences={stats['sentences']} groups={stats['groups']} -> {stats['outputs']['english_groups']}")

    summary = {
        "translation_backend": translator.backend,
        "translation_model": translator.model_name,
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
