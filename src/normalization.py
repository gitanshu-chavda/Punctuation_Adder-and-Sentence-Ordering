from pathlib import Path
import unicodedata
import re

DATA_FOLDER = Path("data/raw/samples")

def normalize_text(text):

    text = unicodedata.normalize("NFC", text) #Standardizing unicode rep

    text = re.sub(r"\s+", " ", text) # replacing multiple whitespace character with one space

    text = text.strip() # removing unnecessary spaces at beginning and end

    return text;


text_files = sorted(DATA_FOLDER.glob("*.txt"))

for file_path in text_files:
    text = file_path.read_text(encoding="utf-8")

    normalized_text = normalize_text(text)

    print("\nFile:", file_path.name)
    print("Original characters:", len(text))
    print("Normalized characters:", len(normalized_text))
    print("Preview:", normalized_text[:200])