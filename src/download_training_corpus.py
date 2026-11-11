
from pathlib import Path
from datasets import load_dataset


OUTPUT_FOLDER = Path("data/external")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


OUTPUT_FILE = OUTPUT_FOLDER / "hindi_wikipedia.txt"


MAX_ARTICLES = 10000


dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.hi",
    split="train",
    streaming=True
)


saved_articles = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as file:

    for article in dataset:

        text = article.get("text", "").strip()

        if not text or len(text) < 300:
            continue

        # Keep each article as a separate text block
        text = " ".join(text.split())

        file.write(text + "\n\n")

        saved_articles += 1

        if saved_articles >= MAX_ARTICLES:
            break


print("Articles saved:", saved_articles)
print("Output file:", OUTPUT_FILE)
print("File size (MB):", round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2))
