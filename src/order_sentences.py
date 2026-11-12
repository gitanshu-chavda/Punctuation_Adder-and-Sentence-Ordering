"""Order punctuated Hindi sentences for all samples (candidate ordering).

Reads:  data/punctuated/sample-*.txt
Writes: data/ordered/<stem>.ordered.txt
        data/ordered/<stem>.mapping.tsv
        data/ordered/<stem>.order_stats.json

This produces a *candidate* logical/original order, not ground truth.
Run from project root:
    python src/order_sentences.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INPUT_FOLDER = Path("data/punctuated")
OUTPUT_FOLDER = Path("data/ordered")
RANDOM_SEED = 42

CONTEXT_STARTERS = {
    "इसलिए", "इसके", "इसीलिए", "तब", "फिर", "लेकिन", "किंतु",
    "परंतु", "अतः", "बाद", "उन्होंने", "उसने", "उसके", "इनके",
}
FOLLOW_UP_STARTERS = {"फिर", "तब", "इसके", "बाद", "इसलिए", "अतः"}


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_sentences(text: str) -> list[str]:
    """Extract sentences from punctuated or ordered text.

    - For source punctuated text: split on danda/question-mark boundaries.
    - For already-ordered files (one sentence per line): prefer lines so that
      any segment without terminal punctuation is preserved.
    """
    stripped = text.strip()
    if not stripped:
        return []
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    # If it looks like a pre-segmented file, keep lines as units.
    if len(lines) > 1 and sum(1 for ln in lines if re.search(r"[।?!]$", ln)) >= max(1, len(lines) - 2):
        return lines
    parts = re.split(r"(?<=[।?])\s+", stripped)
    return [p.strip() for p in parts if p.strip()]


def first_word(sentence: str) -> str:
    return sentence.split()[0].strip('.,:;!?।""\'"') if sentence.split() else ""


def start_penalty(sentence: str) -> float:
    return 0.12 if first_word(sentence) in CONTEXT_STARTERS else 0.0


def order_one_topic(indices, similarities, sentences):
    if len(indices) <= 1:
        return list(indices)

    def opening_score(i):
        relatedness = np.mean([similarities[i, j] for j in indices if j != i])
        return float(relatedness) - start_penalty(sentences[i])

    current = max(indices, key=lambda i: (opening_score(i), -i))
    result = [current]
    remaining = set(indices) - {current}

    while remaining:
        def transition_score(j):
            score = float(similarities[current, j])
            if first_word(sentences[j]) in FOLLOW_UP_STARTERS:
                score += 0.04
            return score

        current = max(remaining, key=lambda j: (transition_score(j), -j))
        result.append(current)
        remaining.remove(current)

    return result


def suggest_order(sentences, embeddings):
    count = len(sentences)
    if count == 0:
        return [], []
    if count == 1:
        return [0], [1]

    n_topics = max(1, min(8, round(count / 8)))
    if n_topics == 1:
        labels = np.zeros(count, dtype=int)
    else:
        labels = KMeans(
            n_clusters=n_topics,
            random_state=RANDOM_SEED,
            n_init=10,
        ).fit_predict(embeddings)

    similarities = embeddings @ embeddings.T
    groups = {label: np.where(labels == label)[0].tolist() for label in np.unique(labels)}
    topic_ids = sorted(groups, key=lambda label: (-len(groups[label]), int(label)))

    ordered_ids = []
    topic_numbers = []
    for topic_number, topic_id in enumerate(topic_ids, start=1):
        candidates = order_one_topic(groups[topic_id], similarities, sentences)
        ordered_ids.extend(candidates)
        topic_numbers.extend([topic_number] * len(candidates))

    assert sorted(ordered_ids) == list(range(count))
    return ordered_ids, topic_numbers


def process_file(model, input_file: Path, output_folder: Path) -> dict:
    text = input_file.read_text(encoding="utf-8")
    sentences = extract_sentences(text)
    if not sentences:
        raise ValueError(f"No sentences found in {input_file}")

    embeddings = model.encode(
        sentences,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    ordered_ids, topic_numbers = suggest_order(sentences, embeddings)

    stem = input_file.stem  # sample-N
    ordered_file = output_folder / f"{stem}.ordered.txt"
    mapping_file = output_folder / f"{stem}.mapping.tsv"
    stats_file = output_folder / f"{stem}.order_stats.json"

    ordered_file.write_text("\n".join(sentences[i] for i in ordered_ids) + "\n", encoding="utf-8")

    mapping_lines = ["new_position\toriginal_position\ttopic"]
    for new_pos, (orig_idx, topic) in enumerate(zip(ordered_ids, topic_numbers), start=1):
        mapping_lines.append(f"{new_pos}\t{orig_idx + 1}\t{topic}")
    mapping_file.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")

    stats = {
        "input": str(input_file),
        "sentences": len(sentences),
        "topics": len(set(topic_numbers)) if topic_numbers else 0,
        "outputs": {
            "ordered": str(ordered_file),
            "mapping": str(mapping_file),
        },
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    _safe_stdout()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_FOLDER.glob("sample-*.txt"))
    if not files:
        print(f"No input files found in {INPUT_FOLDER}")
        return 1

    print(f"Loading sentence encoder on CPU: {MODEL_NAME}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    results = []
    for f in files:
        print(f"\nOrdering {f.name} ...", flush=True)
        stats = process_file(model, f, OUTPUT_FOLDER)
        results.append(stats)
        print(f"  sentences={stats['sentences']} topics={stats['topics']} -> {stats['outputs']['ordered']}")

    # Backward-compatible copy for sample-1 (older naming used by earlier script/notebook)
    s1_ordered = OUTPUT_FOLDER / "sample-1.ordered.txt"
    s1_candidate = OUTPUT_FOLDER / "sample-1.candidate.txt"
    if s1_ordered.exists():
        s1_candidate.write_text(s1_ordered.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {"files": results}
    (OUTPUT_FOLDER / "ordering_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nDone. Processed {len(results)} files. Summary: {OUTPUT_FOLDER / 'ordering_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
