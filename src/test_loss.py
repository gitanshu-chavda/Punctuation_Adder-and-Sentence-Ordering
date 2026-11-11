
from pathlib import Path
import json
import torch
import torch.nn as nn


# 1. Load the saved class weights
WEIGHTS_FILE = Path("data/processed/encoded/class_weights.json")

with WEIGHTS_FILE.open("r", encoding="utf-8") as file:
    weights = json.load(file)


# 2. Convert weights into a PyTorch tensor
class_weights = torch.tensor(
    weights,
    dtype=torch.float32
)


# 3. Create the weighted loss function
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# 4. Create sample model predictions
logits = torch.randn(2, 4, 5)

# Actual labels for two sequences
labels = torch.tensor([
    [0, 1, 2, 0],
    [0, 2, 0, 0]
])

# Original sequence lengths
lengths = torch.tensor([4, 2])


# 5. Create a padding mask
positions = torch.arange(4).unsqueeze(0)

mask = positions < lengths.unsqueeze(1)


# 6. Remove padding positions
valid_logits = logits[mask]
valid_labels = labels[mask]


# 7. Calculate loss
loss = criterion(
    valid_logits,
    valid_labels
)


# 8. Display results
print("Class weights:", class_weights)
print("Original logits shape:", logits.shape)
print("Original labels shape:", labels.shape)
print("Padding mask:")
print(mask)
print("Valid logits shape:", valid_logits.shape)
print("Valid labels shape:", valid_labels.shape)
print("Loss:", loss.item())
