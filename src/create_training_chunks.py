
from pathlib import Path
import json

INPUT_FOLDER = Path("data/processed/encoded")
OUTPUT_FOLDER = Path("data/processed/encoded")

MAX_SEQUENCE_LENGTH = 128

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def create_chunks(input_file, output_file):

    total_articles = 0
    total_chunks = 0
    total_words = 0

    with input_file.open("r", encoding="utf-8") as source, \
         output_file.open("w", encoding="utf-8") as destination:

        for line in source:

            # Read one encoded article
            record = json.loads(line)

            input_ids = record["input_ids"]
            label_ids = record["label_ids"]

            if len(input_ids) != len(label_ids):
                raise ValueError("Input and label lengths do not match")

            total_articles += 1
            total_words += len(input_ids)

            for start in range(0, len(input_ids), MAX_SEQUENCE_LENGTH):

                end = min(start + MAX_SEQUENCE_LENGTH, len(input_ids))

                chunk = {
                    "input_ids": input_ids[start:end],
                    "label_ids": label_ids[start:end]
                }

                destination.write(
                    json.dumps(chunk) + "\n"
                )

                total_chunks += 1

    return total_articles, total_chunks, total_words


train_stats = create_chunks(
    INPUT_FOLDER / "train.jsonl",
    OUTPUT_FOLDER / "train_chunks.jsonl"
)

validation_stats = create_chunks(
    INPUT_FOLDER / "validation.jsonl",
    OUTPUT_FOLDER / "validation_chunks.jsonl"
)


print("TRAINING DATA")
print("Articles:", train_stats[0])
print("Chunks:", train_stats[1])
print("Words:", train_stats[2])

print("\nVALIDATION DATA")
print("Articles:", validation_stats[0])
print("Chunks:", validation_stats[1])
print("Words:", validation_stats[2])

print("\nMaximum sequence length:", MAX_SEQUENCE_LENGTH)
print("Chunk files saved in:", OUTPUT_FOLDER)
