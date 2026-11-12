# Project Guide (Beginner-Friendly)

This guide explains what the project does, what has been done so far, and how to continue safely.

## 1. Goal

Reconstruct readable, logically ordered Hindi text from unpunctuated and jumbled sentences:

1. Add punctuation
2. Detect sentence boundaries
3. Reorder sentences to the likely original/logical sequence
4. Group every 5 consecutive sentences (in order)
5. Translate each group to English

**Constraint:** No GPT/Gemini/Claude or other off-the-shelf generative LLMs. Any model used must stay under **1.5 GB GPU RAM** (end-to-end). CPU is fine.

## 2. Data and the `NULL` marker

- Original files: `data/raw/samples/sample-1..7.txt` (UTF-8, mostly one long line, no standard punctuation).
- There are **67 literal `NULL` tokens** total across files. Their meaning is **unknown**.
- **Rule:** Never delete, replace, or “fix” `NULL` without evidence. Always preserve the original text.

Prepared copies for the model live in `data/model_input/` and must keep all `NULL` markers.

## 3. Current implementation

### 3.1 Input preparation
- Script: `src/prepare_input.py`
- Copies `data/raw/samples/*.txt` → `data/model_input/*.txt`
- Counts standalone `NULL` markers for diagnostics but does **not** modify content

### 3.2 Punctuation restoration
- Script: `src/punctuation.py`
- Model: `Horizon-Labs/punctuation-restoration-small` (token classification, 6 labels)
- Runtime: **ONNX Runtime on CPU** using `models/punctuation/onnx/model_quantized.onnx`
- Tokenizer: local files in `models/punctuation/tokenizer/`
- Labels: `0` (none), `.`, `,`, `?`, `-`, `:`. Predicted `.` is converted to Hindi danda `।`.
- Long text is handled with chunking (`CHUNK_SIZE=400`, `OVERLAP_SIZE=50`).
- Outputs: `data/punctuated/sample-*.txt`

**Known things to review carefully:**
- The model may insert punctuation around `NULL` (e.g. `NULL:` or `। NULL`). This does not mean `NULL` was removed, but it can change interpretation.
- `restore()` normalizes whitespace (`\s+` → space). Newlines/spacing layout from input are not preserved.
- Punctuation quality varies by genre (informational vs dialogue). Sample-6/7 need extra manual inspection.
- Successful execution is not proof of correctness.

### 3.3 NULL diagnostics
- Script: `scripts/null_diagnostic.py`
- Compares raw vs punctuated text around every `NULL` occurrence (by order) to show inserted punctuation/context.

### 3.4 Sentence ordering (candidate)
- Script: `src/sentence-ordering.py`
- Currently processes `data/punctuated/sample-1.txt` → `data/ordered/sample-1.candidate.txt` and `sample-1.mapping.tsv`
- Uses multilingual sentence embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) and heuristics (topic clustering + greedy coherence path).
- **Important:** This is a *candidate* order, not guaranteed original order. Embedding model download/size and runtime should be checked against project constraints if GPU/large models are involved.

## 4. How to work (recommended workflow)

1. **Inspect before changing**: read relevant `src/` scripts and sample inputs/outputs.
2. **Run diagnostics**: use `scripts/null_diagnostic.py` and compare samples manually.
3. **Make one focused improvement** (e.g. punctuation placement or NULL handling hypothesis) and re-run on all 7 files.
4. **Evaluate**: use `scripts/evaluate_punctuation.py` if gold references exist in `data/processed/*.sentences.txt`, but treat as approximate; prefer manual review on a small set.
5. **Record findings**: update this guide or notebook notes with what you observed.

## 5. Notebooks (for learning and demo)

- `00_start_here.ipynb`: environment, paths, and sanity checks
- `01_punctuation_demo.ipynb`: step-by-step punctuation restoration
- `02_null_diagnostic.ipynb`: understand `NULL` behavior
- `03_sentence_ordering_demo.ipynb`: understand candidate ordering

Use notebooks to **learn and demonstrate**; keep core logic in `src/` and `scripts/`.

## 6. Next steps (after punctuation quality is acceptable)

- Extend sentence ordering to all samples and improve evaluation strategy
- Group every 5 consecutive ordered sentences
- Choose a compliant Hindi→English translation method (no generative LLMs, <1.5GB GPU total) and implement it
- Measure runtime and memory usage end-to-end
- Prepare a clean faculty demo notebook showcasing the full pipeline
