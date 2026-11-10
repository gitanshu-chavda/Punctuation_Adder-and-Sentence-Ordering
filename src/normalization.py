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


OUTPUT_FOLDER = Path("data/processed")


OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True) #this will create it if it doesnt exist


for file_path in text_files:

    text = file_path.read_text(encoding="utf-8")

    normalized_text = normalize_text(text)

    
    output_path = OUTPUT_FOLDER / file_path.name

    output_path.write_text(
        normalized_text,
        encoding="utf-8"
    )

    print("Processed:", file_path.name)