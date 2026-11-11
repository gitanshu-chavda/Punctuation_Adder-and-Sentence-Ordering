
from pathlib import Path

FILE_PATH = Path("data/external/hindi_wikipedia.txt")

print("File exists:", FILE_PATH.exists())

text = FILE_PATH.read_text(encoding="utf-8")

print("Total characters:", len(text))
print("Total words:", len(text.split()))

articles = [article for article in text.split("\n\n") if article.strip()]

print("Total articles:", len(articles))

print("\nFirst 500 characters:")
print(text[:500])

print("\nFirst article length:", len(articles[0]))
