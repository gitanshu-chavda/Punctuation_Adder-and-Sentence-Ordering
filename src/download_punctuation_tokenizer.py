
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_ID = "Horizon-Labs/punctuation-restoration-small"

TOKENIZER_FOLDER = Path("models/punctuation/tokenizer")
TOKENIZER_FOLDER.mkdir(parents=True, exist_ok=True)

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json"
]

for filename in TOKENIZER_FILES:

    print("Downloading:", filename)

    try:
        downloaded_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            local_dir=TOKENIZER_FOLDER
        )

        print("Saved:", downloaded_path)

    except Exception as error:
        print("Could not download:", filename)
        print("Reason:", error)

print("\nTokenizer download step completed.")

