
from pathlib import Path
import importlib.util

import torch
from huggingface_hub import hf_hub_download


# 1. Model and input configuration
MODEL_ID = "Horizon-Labs/punctuation-restoration-small"

INPUT_FILE = Path("data/model_input/sample-1.txt")


# 2. Check whether the input file exists
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


# 3. Download the model's punctuation helper
print("Downloading punctuation helper...")

helper_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="punctuate.py"
)


# 4. Import the downloaded helper
spec = importlib.util.spec_from_file_location(
    "punctuate",
    helper_path
)

punctuate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(punctuate)


# 5. Load the pretrained model on CPU
print("\nLoading pretrained punctuation model...")

model = punctuate.Punctuator(
    MODEL_ID,
    device="cpu"
)

print("Model loaded successfully!")


# 6. Read the Hindi input
text = INPUT_FILE.read_text(encoding="utf-8")


# 7. Restore punctuation
print("\nRestoring punctuation...")

punctuated_text = model(text)


# 8. Display results
print("\nORIGINAL TEXT")
print("-" * 60)
print(text[:1000])

print("\nPUNCTUATED TEXT")
print("-" * 60)
print(punctuated_text[:1000])


# 9. Confirm GPU memory is not being used by this model
print("\nInference device: CPU")
print("CUDA available:", torch.cuda.is_available())
