from pathlib import Path
import re

INPUT_FOLDER = Path("data/processed") 
OUTPUT_FOLDER = Path("data/model_input")

text_files = sorted(INPUT_FOLDER.glob("*.txt"))

for file_path in text_files:

    text = file_path.read_text(encoding="utf-8")

    cleaned_text = re.sub(         #.sub() - meaning to substitute something
        r"\bNULL\b",
        " ",
        text,
        flags=re.IGNORECASE 
    )

    cleaned_text = re.sub(r"\s+", " ", cleaned_text)  #reducing multiple whitespaces to single space

    cleaned_text = cleaned_text.strip()    #removing spaces at start and end

    output_path = OUTPUT_FOLDER / file_path.name
    output_path.write_text(cleaned_text, encoding="utf-8")

    print("Prepared:", file_path.name)

print("\nTotal files prepared:", len(text_files))
print("Output directory:", OUTPUT_FOLDER)