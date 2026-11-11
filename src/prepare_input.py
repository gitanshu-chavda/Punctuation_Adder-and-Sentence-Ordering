
from pathlib import Path
import re
import shutil


# 1. Define folders
SOURCE_FOLDER = Path("data/raw/samples")
OUTPUT_FOLDER = Path("data/model_input")


# 2. Check source folder
if not SOURCE_FOLDER.exists():
    raise FileNotFoundError(
        f"Source folder not found: {SOURCE_FOLDER}"
    )


# 3. Create output folder
OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# 4. Copy original files without modifying their contents
total_null = 0
total_files = 0

for input_file in sorted(SOURCE_FOLDER.glob("*.txt")):

    output_file = OUTPUT_FOLDER / input_file.name

    # Read only for counting NULL markers
    text = input_file.read_text(encoding="utf-8")

    null_count = len(
        re.findall(r"(?<!\S)NULL(?!\S)", text)
    )

    # Copy the original file exactly
    shutil.copyfile(input_file, output_file)

    total_null += null_count
    total_files += 1

    print(
        f"{input_file.name}: "
        f"{null_count} NULL markers preserved"
    )


# 5. Display summary
print("\n" + "=" * 40)
print("INPUT PREPARATION COMPLETED")
print("Files copied:", total_files)
print("NULL markers preserved:", total_null)
print("Output folder:", OUTPUT_FOLDER)
