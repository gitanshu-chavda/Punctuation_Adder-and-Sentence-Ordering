from pathlib import Path
import csv
import re

INPUT_FOLDER = Path("data/processed")
OUTPUT_FOLDER = Path("data/custom_model")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

text_files = sorted(INPUT_FOLDER.glob("*.txt"))

null_records = []

for file_path in text_files:

    text = file_path.read_text(encoding="utf-8")

    matches = list(re.finditer(r"\bNULL\b", text, flags=re.IGNORECASE))

    for marker_number, match in enumerate(matches, start=1):

        before = text[:match.start()]
        after = text[match.end():]

        previous_word = before.split()[-1] if before.split() else ""
        next_word = after.split()[0] if after.split() else ""

        start = max(0, match.start()-70)
        end = min(len(text), match.end()+70)

        context = text[start:end]

        null_records.append({
            "file": file_path.name,
            "marker_number": marker_number,
            "previous_word": previous_word,
            "next_word": next_word,
            "context": context
        })

output_file = OUTPUT_FOLDER / "null_audit.csv"

with output_file.open("w", encoding="utf-8-sig", newline="") as file:

    fieldnames = [
        "file",
        "marker_number",
        "previous_word",
        "next_word",
        "context"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(null_records)



print("Total NULL occurrences:", len(null_records))
print("Saved audit file:", output_file)
