# both the .split() method and .trivial_tokenize() gave same number of tokens


from pathlib import Path
from indicnlp.tokenize import indic_tokenize

DATA_FOLDER = Path("data/processed")

file_path = DATA_FOLDER / "sample-1.txt"

text = file_path.read_text(encoding="utf-8")

python_tokens = text.split()   #m1 : separates at whitespaces by python

indic_tokens = indic_tokenize.trivial_tokenize(text, lang="hi")   #m2 : using nltk(can also separate where punctuations are ther)

print("Original text:")
print(text[:200])

print("\nPython tokenization:")
print(python_tokens[:30])

print("\nIndic NLP tokenization:")
print(indic_tokens[:30])

print("\nTotal Python tokens:", len(python_tokens))
print("Total Indic NLP tokens:", len(indic_tokens))


"""test_text = "राम मंदिर गया। सीता घर पर थी, क्या मोहन आया?"

tokens = indic_tokenize.trivial_tokenize(
    test_text,
    lang="hi"
)      #testing the tokenization with punctuations

print(tokens)"""
