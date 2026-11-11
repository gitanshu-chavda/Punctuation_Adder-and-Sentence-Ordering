
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_ID = "Horizon-Labs/punctuation-restoration-small"

MODEL_FOLDER = Path("models/punctuation")
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

print("Downloading quantized ONNX model...")

model_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="onnx/model_quantized.onnx",
    local_dir=MODEL_FOLDER
)

print("\nDownload completed!")
print("Model path:", model_path)
print(
    "Model size (MB):",
    round(Path(model_path).stat().st_size / (1024 * 1024), 2)
)

