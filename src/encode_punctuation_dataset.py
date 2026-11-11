from pathlib import Path
from collections import Counter
import json
import random

INPUT_FILE = Path("data/processed/custom_punctuation_training.jsonl")
OUTPUT_FOLDER = Path("data/processed/encoded")

MAX_VOCAB_SIZE = 50000
TRAIN_RATIO = 0.90
RANDOM_SEED = 42

LABEL_TO_ID = {
    "NONE": 0,
    "COMMA": 1,
    "FULL_STOP": 2,
    "QUESTION": 3,
    "EXCLAMATION": 4
}

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

with INPUT_FILE.open("r", encoding="utf-8") as file:
    total_articles = sum(1 for _ in file)

indices = list(range(total_articles))

random.Random(RANDOM_SEED).shuffle(indices)

train_size = int(total_articles * TRAIN_RATIO)

train_indices = set(indices[:train_size])
validation_indices = set(indices[train_size:])

word_counts = Counter()

with INPUT_FILE.open("r", encoding="utf-8") as file:

    for index, line in enumerate(file):

        if index not in train_indices:
            continue

        record = json.loads(line)

        word_counts.update(record["tokens"])


vocabulary = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in word_counts.most_common(MAX_VOCAB_SIZE - 2):
    vocabulary[word] = len(vocabulary)


with (OUTPUT_FOLDER / "vocabulary.json").open("w", encoding="utf-8") as file:
    json.dump(vocabulary, file, ensure_ascii=False)

with (OUTPUT_FOLDER / "labels.json").open("w", encoding="utf-8") as file:
    json.dump(LABEL_TO_ID, file, ensure_ascii=False, indent=4)


train_words = 0
validation_words = 0
unknown_words = 0

with INPUT_FILE.open("r", encoding="utf-8") as source:

    with (OUTPUT_FOLDER / "train.jsonl").open("w", encoding="utf-8") as train_file, \
         (OUTPUT_FOLDER / "validation.jsonl").open("w", encoding="utf-8") as validation_file:

        for index, line in enumerate(source):

            record = json.loads(line)

            input_ids = [
                vocabulary.get(word, vocabulary["<UNK>"])
                for word in record["tokens"]
            ]

            label_ids = [
                LABEL_TO_ID[label]
                for label in record["labels"]
            ]

            encoded_record = {
                "input_ids": input_ids,
                "label_ids": label_ids
            }

            if index in train_indices:
                json.dump(encoded_record, train_file)
                train_file.write("\n")
                train_words += len(input_ids)

            else:
                json.dump(encoded_record, validation_file)
                validation_file.write("\n")
                validation_words += len(input_ids)

            unknown_words += sum(
                1 for word_id in input_ids
                if word_id == vocabulary["<UNK>"]
            )

print("Total articles:", total_articles)
print("Training articles:", len(train_indices))
print("Validation articles:", len(validation_indices))
print("Vocabulary size:", len(vocabulary))
print("Training words:", train_words)
print("Validation words:", validation_words)
print("Unknown word occurrences:", unknown_words)
print("Unknown word percentage:", round(
    unknown_words / (train_words + validation_words) * 100, 2
))
print("Saved in:", OUTPUT_FOLDER)
