
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


DATA_FOLDER = Path("data/processed/encoded")

BATCH_SIZE = 32

PAD_ID = 0
IGNORE_INDEX = -100


class PunctuationDataset(Dataset):

    def __init__(self, file_path):

        self.file_path = file_path

        # Store the starting position of every JSON record
        self.offsets = []

        with file_path.open("rb") as file:

            while True:

                position = file.tell()
                line = file.readline()

                if not line:
                    break

                self.offsets.append(position)

        # Open the file for retrieving individual records
        self.file = file_path.open("rb")


    def __len__(self):

        return len(self.offsets)


    def __getitem__(self, index):

        position = self.offsets[index]

        self.file.seek(position)

        line = self.file.readline()

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



def collate_batch(batch):         #combining into batch

    input_sequences = [item[0] for item in batch]

    label_sequences = [item[1] for item in batch]

    lengths = torch.tensor(
        [len(sequence) for sequence in input_sequences],
        dtype=torch.long
    )    #recording the original length

    padded_inputs = pad_sequence(
        input_sequences,
        batch_first=True,
        padding_value=PAD_ID
    )

    padded_labels = pad_sequence(
        label_sequences,
        batch_first=True,
        padding_value=IGNORE_INDEX
    )

    return padded_inputs, padded_labels, lengths


train_dataset = PunctuationDataset(
    DATA_FOLDER / "train_chunks.jsonl"
)

validation_dataset = PunctuationDataset(
    DATA_FOLDER / "validation_chunks.jsonl"
)


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


input_batch, label_batch, lengths = next(iter(train_loader))


print("TRAINING DATASET")
print("Total chunks:", len(train_dataset))

print("\nVALIDATION DATASET")
print("Total chunks:", len(validation_dataset))

print("\nONE TRAINING BATCH")
print("Input tensor shape:", input_batch.shape)
print("Label tensor shape:", label_batch.shape)
print("Lengths shape:", lengths.shape)

print("\nFirst 10 input IDs:")
print(input_batch[0][:10].tolist())

print("\nFirst 10 label IDs:")
print(label_batch[0][:10].tolist())

print("\nOriginal length of first sequence:", lengths[0].item())

print("\nNumber of padding positions in this batch:",
      (input_batch == PAD_ID).sum().item())

print("\nDataLoader preparation successful!")
