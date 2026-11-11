
from pathlib import Path
from collections import Counter
import json

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import PunctuationBiLSTM


DATA_FOLDER = Path("data/processed/encoded")
MODEL_FOLDER = Path("models")

TRAIN_FILE = DATA_FOLDER / "train_chunks.jsonl"
VALIDATION_FILE = DATA_FOLDER / "validation_chunks.jsonl"
WEIGHTS_FILE = DATA_FOLDER / "class_weights.json"

MODEL_FILE = MODEL_FOLDER / "best_punctuation_bilstm.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 5

DEVICE = torch.device("cpu")


# 2. Create the model and output folder
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

model = PunctuationBiLSTM().to(DEVICE)


# 3. Custom Dataset with lazy file reading
class ChunkDataset(Dataset):

    def __init__(self, file_path):

        self.file_path = file_path
        self.offsets = []
        self.file = None

        with self.file_path.open("rb") as file:

            while True:

                offset = file.tell()
                line = file.readline()

                if not line:
                    break

                self.offsets.append(offset)


    def __len__(self):

        return len(self.offsets)


    def __getitem__(self, index):

        if self.file is None:
            self.file = self.file_path.open("rb")

        self.file.seek(self.offsets[index])

        line = self.file.readline().decode("utf-8")

        record = json.loads(line)

        input_ids = torch.tensor(
            record["input_ids"],
            dtype=torch.long
        )

        label_ids = torch.tensor(
            record["label_ids"],
            dtype=torch.long
        )

        return input_ids, label_ids


# 4. Function to create batches with padding
def collate_batch(batch):

    input_sequences, label_sequences = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in input_sequences],
        dtype=torch.long
    )

    padded_inputs = pad_sequence(
        input_sequences,
        batch_first=True,
        padding_value=0
    )

    padded_labels = pad_sequence(
        label_sequences,
        batch_first=True,
        padding_value=0
    )

    return padded_inputs, padded_labels, lengths


# 5. Load the training and validation datasets
train_dataset = ChunkDataset(TRAIN_FILE)
validation_dataset = ChunkDataset(VALIDATION_FILE)

print("Training chunks:", len(train_dataset))
print("Validation chunks:", len(validation_dataset))


# 6. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0
)


# 7. Load class weights and create loss function
with WEIGHTS_FILE.open("r", encoding="utf-8") as file:

    weights = json.load(file)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# 8. Create the Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 9. Function to calculate loss and metrics
def evaluate(data_loader):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    confusion_matrix = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long
    )

    with torch.no_grad():

        for inputs, labels, lengths in data_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs, lengths)

            positions = torch.arange(
                inputs.size(1)
            ).unsqueeze(0)

            mask = positions < lengths.unsqueeze(1)

            valid_logits = logits[mask]
            valid_labels = labels[mask]

            loss = criterion(
                valid_logits,
                valid_labels
            )

            total_loss += loss.item()
            total_batches += 1

            predictions = valid_logits.argmax(dim=1)

            encoded = (
                valid_labels * NUM_CLASSES + predictions
            )

            confusion_matrix += torch.bincount(
                encoded.cpu(),
                minlength=NUM_CLASSES ** 2
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    average_loss = total_loss / total_batches

    return average_loss, confusion_matrix


# 10. Function to calculate precision, recall and F1
def calculate_metrics(confusion_matrix):

    metrics = {}

    f1_scores = []

    for class_id in range(NUM_CLASSES):

        true_positive = confusion_matrix[class_id, class_id].item()

        predicted_count = confusion_matrix[:, class_id].sum().item()

        actual_count = confusion_matrix[class_id, :].sum().item()

        precision = (
            true_positive / predicted_count
            if predicted_count > 0 else 0.0
        )

        recall = (
            true_positive / actual_count
            if actual_count > 0 else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        metrics[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_count
        }

        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / NUM_CLASSES

    return metrics, macro_f1


# 11. Training loop
best_macro_f1 = -1.0

ID_TO_LABEL = {
    0: "NONE",
    1: "COMMA",
    2: "FULL_STOP",
    3: "QUESTION",
    4: "EXCLAMATION"
}

for epoch in range(EPOCHS):

    model.train()

    total_training_loss = 0.0
    total_training_batches = 0

    for inputs, labels, lengths in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs, lengths)

        positions = torch.arange(
            inputs.size(1)
        ).unsqueeze(0)

        mask = positions < lengths.unsqueeze(1)

        valid_logits = logits[mask]
        valid_labels = labels[mask]

        loss = criterion(
            valid_logits,
            valid_labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_training_loss += loss.item()
        total_training_batches += 1

    average_training_loss = (
        total_training_loss / total_training_batches
    )

    validation_loss, confusion_matrix = evaluate(
        validation_loader
    )

    metrics, macro_f1 = calculate_metrics(
        confusion_matrix
    )

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("Training loss:", round(average_training_loss, 4))
    print("Validation loss:", round(validation_loss, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClass-wise metrics:")

    for class_id, label_name in ID_TO_LABEL.items():

        result = metrics[class_id]

        print(
            f"{label_name:15} "
            f"Precision: {result['precision']:.4f} "
            f"Recall: {result['recall']:.4f} "
            f"F1: {result['f1']:.4f} "
            f"Support: {result['support']:,}"
        )

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": macro_f1,
                "validation_loss": validation_loss
            },
            MODEL_FILE
        )

        print("\nBest model saved!")

    print("-" * 60)


print("\nTraining completed!")
print("Best validation Macro F1:", round(best_macro_f1, 4))
print("Best model saved at:", MODEL_FILE)
