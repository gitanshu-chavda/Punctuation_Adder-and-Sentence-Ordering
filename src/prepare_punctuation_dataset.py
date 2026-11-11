
from pathlib import Path
from collections import Counter
import json
import unicodedata

from indicnlp.tokenize import indic_tokenize

INPUT_FILE = Path("data/external/hindi_wikipedia.txt")
OUTPUT_FILE = Path("data/processed/custom_punctuation_training.jsonl")

PUNCTUATION_MAP = {
    ",": "COMMA",
    "।": "FULL_STOP",
    "॥": "FULL_STOP",
    ".": "FULL_STOP",
    "?": "QUESTION",
    "!": "EXCLAMATION"
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

label_counts = Counter()
total_words = 0
total_articles = 0

with INPUT_FILE.open("r", encoding="utf-8") as source, \
     OUTPUT_FILE.open("w", encoding="utf-8") as destination:

    for article in source:

        article = article.strip()

        if not article:
            continue

        tokens = indic_tokenize.trivial_tokenize(
            article,
            lang="hi"
        )

        words = []
        labels = []

        for token in tokens:

            if token in PUNCTUATION_MAP:

                if words:
                    labels[-1] = PUNCTUATION_MAP[token]

                continue

            if all(
                unicodedata.category(char).startswith("P")
                for char in token
            ):
                continue

            words.append(token)
            labels.append("NONE")


        if not words:
            continue


        record = {
            "tokens": words,
            "labels": labels
        }

        destination.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_articles += 1
        total_words += len(words)
        label_counts.update(labels)

print("Articles processed:", total_articles)
print("Total words:", total_words)
print("Label distribution:")

for label, count in label_counts.most_common():
    print(label, ":", count)

print("Output file:", OUTPUT_FILE)
print("Output size (MB):", round(
    OUTPUT_FILE.stat().st_size / (1024 * 1024), 2
))
