from pathlib import Path
import re 

DATA_FOLDER = Path("data/raw/samples")

text_files = list(DATA_FOLDER.glob("*.txt"))

print("=" *50)
print("HINDI DATASET ANALYSIS")
print("="*50)

total_characters = 0
total_words = 0
total_nulls = 0


for file_path in text_files:

    text = file_path.read_text(encoding="utf-8")

    character_count = len(text)

    words = text.split()
    word_count = len(words)

    null_count = len(re.findall(r"\bNULL\b", text, flags=re.IGNORECASE))

    total_characters += character_count
    total_words += word_count
    total_nulls += null_count

    print("\nFIle:", file_path.name)
    print("Characters:", character_count)
    print("Words:", word_count)
    print("NULL Occurences:", null_count)

    if null_count>0:
        print("NULL Context:")

        matches = list(re.finditer(r"\bNULL\b", text, flags=re.IGNORECASE))

        for match in matches[:3]:

            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)

            print("...", text[start:end], "...")

print("\n" + "="*50)
print("OVERALL DATA STATISTICS")

print("="*50)

print("Total Files:", len(text_files))
print("Total characters:", total_characters)
print("Total words:", total_words)
print("Total NULL occurrences:", total_nulls)

