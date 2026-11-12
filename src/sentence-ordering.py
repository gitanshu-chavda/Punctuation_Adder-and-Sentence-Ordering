
"""First-pass, reviewable sentence ordering using multilingual embeddings.

Run from the project root:
    python src/sentence_ordering.py

This produces a *candidate* order, not a guaranteed original sequence.
"""

from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INPUT_FILE = Path("data/punctuated/sample-1.txt")
OUTPUT_FOLDER = Path("data/ordered")
RANDOM_SEED = 42

# These commonly introduce sentences that refer back to earlier context.
CONTEXT_STARTERS = {
    "इसलिए", "इसके", "इसीलिए", "तब", "फिर", "लेकिन", "किंतु",
    "परंतु", "अतः", "बाद", "उन्होंने", "उसने", "उसके", "इनके",
}

# Some words can signal a following step. This is only a weak heuristic.
FOLLOW_UP_STARTERS = {"फिर", "तब", "इसके", "बाद", "इसलिए", "अतः"}


def extract_sentences(text):
    """Split on Hindi danda/question mark followed by whitespace."""
    parts = re.split(r"(?<=[।?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def first_word(sentence):
    """Get the first word, without punctuation around it."""
    return sentence.split()[0].strip('.,:;!?।“”\'"') if sentence.split() else ""


def start_penalty(sentence):
    """How much the opening suggests missing previous context."""
    return 0.12 if first_word(sentence) in CONTEXT_STARTERS else 0.0


def order_one_topic(indices, similarities, sentences):
    """Build a greedy high-coherence path inside one topic group."""
    if len(indices) <= 1:
        return list(indices)

    # Pick a topical, relatively self-contained opening sentence.
    def opening_score(i):
        relatedness = np.mean([similarities[i, j] for j in indices if j != i])
        return float(relatedness) - start_penalty(sentences[i])

    current = max(indices, key=lambda i: (opening_score(i), -i))
    result = [current]
    remaining = set(indices) - {current}

    while remaining:
        def transition_score(j):
            score = float(similarities[current, j])
            # Directional hint: a follow-up phrase is more likely after another sentence.
            if first_word(sentences[j]) in FOLLOW_UP_STARTERS:
                score += 0.04
            return score

        current = max(remaining, key=lambda j: (transition_score(j), -j))
        result.append(current)
        remaining.remove(current)

    return result


def suggest_order(sentences, embeddings):
    """Group related sentences, then suggest a path within each group."""
    count = len(sentences)
    if count == 0:
        return [], []
    if count == 1:
        return [0], [1]

    # Roughly 8 sentences per topic; adjustable after inspecting sample-1.
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


    # Group sentence indices by their topic labels
    groups = {
        label: np.where(labels == label)[0].tolist()
        for label in np.unique(labels)
    }

    # Process larger topic groups first
    topic_ids = sorted(
        groups,
        key=lambda label: (-len(groups[label]), int(label))
    )

    ordered_ids = []
    topic_numbers = []

    for topic_number, topic_id in enumerate(topic_ids, start=1):
        candidates = order_one_topic(
            groups[topic_id],
            similarities,
            sentences
        )

        ordered_ids.extend(candidates)
        topic_numbers.extend([topic_number] * len(candidates))

    # Ensure every sentence appears exactly once
    assert sorted(ordered_ids) == list(range(count))

    return ordered_ids, topic_numbers


def main():
    import sys

    # Ensure Hindi/Unicode output prints safely on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing punctuated file: {INPUT_FILE}"
        )

    text = INPUT_FILE.read_text(encoding="utf-8")
    sentences = extract_sentences(text)

    if not sentences:
        raise ValueError("No sentences found in input")

    print(f"Extracted {len(sentences)} sentences", flush=True)
    print("Loading multilingual sentence encoder on CPU...", flush=True)

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    embeddings = model.encode(
        sentences,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print(f"Embedding matrix shape: {embeddings.shape}")

    ordered_ids, topic_numbers = suggest_order(
        sentences,
        embeddings
    )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    candidate_file = OUTPUT_FOLDER / "sample-1.candidate.txt"

    candidate_file.write_text(
        "\n".join(sentences[i] for i in ordered_ids) + "\n",
        encoding="utf-8"
    )

    mapping_file = OUTPUT_FOLDER / "sample-1.mapping.tsv"

    mapping_lines = [
        "new_position\toriginal_position\ttopic"
    ]

    for new_position, (original_index, topic) in enumerate(
        zip(ordered_ids, topic_numbers),
        start=1
    ):
        mapping_lines.append(
            f"{new_position}\t{original_index + 1}\t{topic}"
        )

    mapping_file.write_text(
        "\n".join(mapping_lines) + "\n",
        encoding="utf-8"
    )

    print(f"\nSuggested topics: {len(set(topic_numbers))}")

    print("\nFirst 12 sentences in candidate order:")

    for pos, idx in enumerate(ordered_ids[:12], start=1):
        print(f"{pos:02d}. [original #{idx + 1:02d}] {sentences[idx]}")

    print(f"\nCandidate order: {candidate_file}")
    print(f"Position mapping: {mapping_file}")


if __name__ == "__main__":
    main()

