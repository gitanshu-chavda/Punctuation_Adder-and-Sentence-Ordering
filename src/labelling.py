from pathlib import Path
from indicnlp.tokenize import indic_tokenize
import csv

DATA_FOLDER = Path("data/processed")

labelled_data = []

text_files = sorted(DATA_FOLDER.glob("*.txt"))

for file_path in text_files:

    text = file_path.read_text(encoding="utf-8")

    tokens = indic_tokenize.trivial_tokenize(text, lang="hi")

    for token in tokens:

        if token.upper() == "NULL":

            if labelled_data and labelled_data[-1]["file"] == file_path.name:
                labelled_data[-1]["label"] = "BOUNDARY"

        else:
            labelled_data.append({
                "file": file_path.name,
                "word": token,
                "label": "NONE"
            })

output_file = Path("weak-labels.csv")

with output_file.open("w", encoding="utf-8", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["file", "word", "label"]
    )

    writer.writeheader()
    writer.writerows(labelled_data)



boundary_count = sum(
    1 for item in labelled_data
    if item["label"] == "BOUNDARY"
)

print("Total labeled words:", len(labelled_data))
print("Boundary labels:", boundary_count)
print("NONE labels:", len(labelled_data) - boundary_count)
print("Saved to:", output_file)


with output_file.open("r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    records = list(reader)

records_by_file = {}

for record in records:

    filename = record["file"]
    if filename not in records_by_file:
        records_by_file[filename] = []

    records_by_file[filename].append(record)

print("\n" + "=" * 50)
print("BOUNDARY LABEL INSPECTION")
print("=" * 50)

for filename, words in records_by_file.items():

    print("\nFile:", filename)

    for i, record in enumerate(words):

        if record["label"] == "BOUNDARY":

            start = max(0,i-5)
            end = min(len(words),i + 6)

            context = [
                item["word"] for item in words[start:end]
            ]


            print("\nContext:", " ".join(context))
            print("Boundary after:", record["word"])