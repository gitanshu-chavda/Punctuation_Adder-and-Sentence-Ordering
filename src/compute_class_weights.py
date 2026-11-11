
from pathlib import Path
from collections import Counter
import json
import math


# 1. Define file paths
INPUT_FILE = Path("data/processed/encoded/train_chunks.jsonl")
OUTPUT_FILE = Path("data/processed/encoded/class_weights.json")


# 2. Count each label
label_counts = Counter()

with INPUT_FILE.open("r", encoding="utf-8") as file:

    for line in file:

        record = json.loads(line)

        label_counts.update(record["label_ids"])


# 3. Define the number of classes
NUM_CLASSES = 5

total_labels = sum(label_counts.values())


# 4. Calculate inverse square-root weights
class_weights = []

for label_id in range(NUM_CLASSES):

    count = label_counts[label_id]

    weight = math.sqrt(
        total_labels / (NUM_CLASSES * count)
    )

    class_weights.append(weight)


# 5. Normalize weights to have an average of 1
average_weight = sum(class_weights) / NUM_CLASSES

class_weights = [
    weight / average_weight
    for weight in class_weights
]


# 6. Save the weights
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as file:

    json.dump(class_weights, file, indent=4)


# 7. Display results
ID_TO_LABEL = {
    0: "NONE",
    1: "COMMA",
    2: "FULL_STOP",
    3: "QUESTION",
    4: "EXCLAMATION"
}

print("CLASS WEIGHTS")
print("-" * 40)

for label_id, label_name in ID_TO_LABEL.items():

    print(
        f"{label_name:15} "
        f"{class_weights[label_id]:.4f}"
    )

print("\nSaved to:", OUTPUT_FILE)
