
from pathlib import Path
from collections import Counter
import json


# 1. Define the training dataset path
INPUT_FILE = Path("data/processed/encoded/train_chunks.jsonl")


# 2. Count the occurrences of each label
label_counts = Counter()

with INPUT_FILE.open("r", encoding="utf-8") as file:

    for line in file:

        record = json.loads(line)

        label_counts.update(record["label_ids"])


# 3. Define label names
ID_TO_LABEL = {
    0: "NONE",
    1: "COMMA",
    2: "FULL_STOP",
    3: "QUESTION",
    4: "EXCLAMATION"
}


# 4. Calculate the total number of labels
total_labels = sum(label_counts.values())


# 5. Display the distribution
print("TRAINING LABEL DISTRIBUTION")
print("-" * 45)

for label_id, label_name in ID_TO_LABEL.items():

    count = label_counts[label_id]

    percentage = count / total_labels * 100

    print(
        f"{label_name:15} "
        f"{count:12,} "
        f"{percentage:8.3f}%"
    )

print("-" * 45)
print("Total labels:", f"{total_labels:,}")
