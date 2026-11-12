# Learning Plan (Short Guide)

This repo already contains working pipeline code and notebooks. Use this short plan to understand it step by step.

## 1. Start with the data
- Inspect `data/raw/samples/*.txt` (original, unpunctuated/jumbled inputs).
- Note the **67 literal `NULL` markers** across files — these must be preserved.
- Compare with `data/model_input/` (prepared copies).

## 2. Punctuation restoration
- Read `src/punctuation.py` and run `notebooks/01_punctuation_demo.ipynb`.
- Learn: token classification, ONNX inference, chunking, token offsets, and why outputs need manual review.

## 3. NULL diagnostics
- Run `scripts/null_diagnostic.py` or `notebooks/02_null_diagnostic.ipynb`.
- Learn: how to inspect model behavior around ambiguous markers without assuming their meaning.

## 4. Sentence ordering (candidate)
- Read `src/order_sentences.py` and `notebooks/03_sentence_ordering_demo.ipynb`.
- Learn: pairwise scoring idea, multilingual BERT embeddings/pair encodings, Borda aggregation, and why candidate order != ground truth.

## 5. Grouping and translation
- Read `src/group_and_translate.py` and inspect `outputs/` and `data/final/`.
- Learn: chunking into groups of 5, sentence-level alignment, and MT model constraints (IndicTrans2 preferred, MarianMT fallback in this environment).

## 6. Evaluation and honesty
- Run `scripts/verify_integrity.py`, `scripts/evaluate_punctuation.py`, and review `reports/`.
- Learn: difference between execution success and quality; proxy metrics vs gold-standard benchmarks.

## Suggested next learning/build steps
- Create a small manually annotated set for punctuation/order evaluation.
- Add Kendall tau/pair accuracy for ordering and BLEU/chrF/human review for translation.
- Profile runtime and memory (CPU/GPU) to document the <1.5 GB constraint rigorously.
