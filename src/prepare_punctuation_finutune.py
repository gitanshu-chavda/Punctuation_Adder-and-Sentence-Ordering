"""Prepare article-level train/validation JSONL for Hindi punctuation fine-tuning.

Run from the project root:
    python src/prepare_punctuation_finetune.py

Reads existing punctuated Hindi Wikipedia text. It never modifies that source.
Outputs independent training/validation examples, with one punctuation label
AFTER each whitespace-delimited word. This does NOT train a model yet.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

SOURCE = Path("data/external/hindi_wikipedia.txt")
OUTPUT = Path("data/finetune")
SEED = 42
MAX_DOCUMENTS = 2_000  # small, reproducible first experiment
WORDS_PER_EXAMPLE = 70  # short sequences for memory-efficient training
MIN_WORDS_PER_DOCUMENT = 30

LABEL_TO_ID = {"0": 0, ".": 1, ",": 2, "?": 3, "-": 4, ":": 5}
PUNCT_TO_LABEL = {
    "।": ".", "॥": ".", ".": ".", "!": ".", "…": ".", ";": ".",
    ",": ",", "?": "?", ":": ":",
    "，": ",", "؟": "?", "。": ".", "：": ":", "？": "?",
}
CLOSERS = set('”’"\'»)]}')
OPENERS = set('“‘"\'«([{')


def get_documents(raw_text: str) -> list[str]:
    # Prefer blank-line-separated articles. If the corpus uses one line per
    # article, fall back to nonempty lines instead.
    blocks = [p.strip() for p in re.split(r"\n[ \t]*\n+", raw_text) if p.strip()]
    if len(blocks) < 20:
        blocks = [p.strip() for p in raw_text.splitlines() if p.strip()]
    if len(blocks) < 20:
        raise ValueError(
            "Could not identify at least 20 documents. Inspect the corpus "
            "format before splitting; do not train on a single large block."
        )
    return blocks


def parse_document(text: str) -> tuple[list[str], list[int]]:
    words: list[str] = []
    labels: list[int] = []

    for token in text.split():
        core = token
        found_marks: list[str] = []

        # Remove marks/closing quotation characters at the END of a word.
        while core and (core[-1] in PUNCT_TO_LABEL or core[-1] in CLOSERS):
            char = core[-1]
            if char in PUNCT_TO_LABEL:
                found_marks.append(PUNCT_TO_LABEL[char])
            core = core[:-1]

        core = core.lstrip("".join(OPENERS))
        if not core or not any(c.isalnum() for c in core):
            continue

        # Avoid training on tokens containing hidden sentence boundaries.
        # They need a more sophisticated sentence-aware tokenizer.
        if any(c in core for c in "।॥!?…"):
            continue

        # Choose the strongest boundary if multiple marks followed a word.
        label = next((m for m in ("?", ".", ":", ",") if m in found_marks), "0")
        words.append(core)
        labels.append(LABEL_TO_ID[label])

    return words, labels


def make_examples(documents: list[str]) -> tuple[list[dict], Counter]:
    result: list[dict] = []
    counts: Counter = Counter()

    for document in documents:
        words, labels = parse_document(document)
        if len(words) < MIN_WORDS_PER_DOCUMENT:
            continue
        for start in range(0, len(words), WORDS_PER_EXAMPLE):
            chunk_words = words[start:start + WORDS_PER_EXAMPLE]
            chunk_labels = labels[start:start + WORDS_PER_EXAMPLE]
            if len(chunk_words) < 15:
                continue
            result.append({"words": chunk_words, "labels": chunk_labels})
            counts.update(chunk_labels)

    return result, counts


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing Wikipedia source: {SOURCE}")

    documents = get_documents(SOURCE.read_text(encoding="utf-8"))
    random.Random(SEED).shuffle(documents)
    documents = documents[:MAX_DOCUMENTS]

    # Split by DOCUMENT before chunking to avoid train/validation leakage.
    split_index = int(0.9 * len(documents))
    training_docs = documents[:split_index]
    validation_docs = documents[split_index:]

    training, train_counts = make_examples(training_docs)
    validation, val_counts = make_examples(validation_docs)
    if not training or not validation:
        raise ValueError("No usable examples. Check source formatting and punctuation.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_jsonl(OUTPUT / "train.jsonl", training)
    save_jsonl(OUTPUT / "val.jsonl", validation)

    inverse = {i: mark for mark, i in LABEL_TO_ID.items()}
    print(f"Source documents: {len(documents)}")
    print(f"Training documents: {len(training_docs)}")
    print(f"Validation documents: {len(validation_docs)}")
    print(f"Training examples: {len(training)}")
    print(f"Validation examples: {len(validation)}")
    print("Training label counts:", {inverse[i]: train_counts[i] for i in range(6)})
    print("Validation label counts:", {inverse[i]: val_counts[i] for i in range(6)})
    print("Saved to:", OUTPUT)
    print("\nOne example:")
    print(" ".join(training[0]["words"][:25]))
    print(training[0]["labels"][:25])


if __name__ == "__main__":
    main()
