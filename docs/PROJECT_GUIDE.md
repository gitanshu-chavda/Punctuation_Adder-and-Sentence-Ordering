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
- Runtime: **ONNX Runtime on CPU** using `models/punctuation/onnx/model_quantized.onnx` (local tokenizer in `models/punctuation/tokenizer/`)
- Labels: `0` (none), `.`, `,`, `?`, `-`, `:`. Predicted `.` is converted to Hindi danda `।`.
- Long text is handled with chunking (`CHUNK_SIZE=400`, `OVERLAP_SIZE=50`).
- Outputs: `data/punctuated/sample-*.txt`
- Tools/libraries: `onnxruntime`, `transformers`, `numpy`, `torch` (present in env). No generative LLMs.

**Known things to review carefully:**
- The model may insert punctuation around `NULL` (e.g. `NULL:` or `। NULL`). This does not mean `NULL` was removed, but it can change interpretation.
- `restore()` normalizes whitespace (`\s+` → space). Newlines/spacing layout from input are not preserved.
- Punctuation quality varies by genre (informational vs dialogue). Sample-6/7 need extra manual inspection.
- Successful execution is not proof of correctness.

### 3.3 NULL diagnostics
- Script: `scripts/null_diagnostic.py`
- Compares raw vs punctuated text around every `NULL` occurrence (by order) to show inserted punctuation/context.

### 3.4 Sentence ordering (candidate, all samples)
- Primary script: `src/order_sentences.py` (current implementation: **multilingual BERT pairwise scoring + Borda aggregation** using `bert-base-multilingual-cased`).
- Reads `data/punctuated/sample-*.txt` and writes per-sample outputs to `data/ordered/`:
  - `sample-N.ordered.txt` (one sentence per line, candidate order)
  - `sample-N.mapping.tsv` (new position, original position, topic)
  - `sample-N.order_stats.json`
  - `ordering_summary.json`
- **Important:** This is a *candidate* order, not guaranteed original/logical ground truth. Existing generated artifacts may come from earlier runs; regenerate and recompute metrics (`scripts/compute_metrics.py`, `reports/evaluation.json`) after code changes. Current reference-alignment metrics (best-effort vs `data/processed/*.sentences.txt`) are reported in README/reports.
- Tools/libraries: `transformers`, `torch`, `numpy` (CPU). No generative LLMs.

### 3.5 Grouping and translation (implemented)
- Script: `src/group_and_translate.py`
- Reads `data/ordered/sample-*.ordered.txt`, groups every **5 consecutive sentences** (last group may be smaller), and translates to English.
- **Preferred backend:** IndicTrans2 (`ai4bharat/indictrans2-indic-en-dist-200M` by default, configurable via env vars) when dependencies/access are available.
- **Fallback backend (current generated outputs in this environment):** `Helsinki-NLP/opus-mt-hi-en` (MarianMT neural MT, encoder-decoder seq2seq, CPU). Not GPT/Gemini/Claude.
- Notes: IndicTrans2 may require `HF_TOKEN`/license access and `IndicTransToolkit`; MarianMT may be cached locally (e.g. `models/translation/opus-mt-hi-en`, ignored by git).
- Outputs in `data/final/`: `sample-N.groups_hi.txt`, `sample-N.groups_en.txt`, `sample-N.groups.tsv`, `sample-N.pipeline.json`, `translation_summary.json`. Aligned outputs also generated in `outputs/` via `scripts/generate_aligned_outputs.py`.

### 3.6 Verification
- `scripts/verify_integrity.py` checks literal `NULL` preservation and basic counts across stages.
- `scripts/run_full_pipeline.py` runs prepare → punctuation → ordering → grouping/translation end-to-end.

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
- `03_sentence_ordering_demo.ipynb`: understand candidate ordering (sample-level demo)
- `04_full_pipeline.ipynb`: inspect full pipeline outputs (ordering, 5-sentence grouping, translation)

Use notebooks to **learn and demonstrate**; keep core logic in `src/` and `scripts/`.

## 6. Remaining work / quality improvements

- **Punctuation quality:** manual review and targeted fixes (token-boundary placement, NULL-adjacent punctuation, dialogue cases). Consider a small human-annotated subset for evaluation.
- **Ordering quality:** validate candidate order against logical/original sequence; improve heuristics or evaluation if needed.
- **Translation quality:** review `data/final/*.groups_en.txt` (MT errors can come from punctuation/order/domain). Keep translation method compliant (no generative LLMs, <1.5GB GPU total).
- **Resource reporting:** measure model load time, per-file inference time, CPU/GPU memory usage for faculty demonstration.
- **Demo polish:** curate examples and explanations in `04_full_pipeline.ipynb` (or a dedicated presentation notebook) once outputs are reviewed.
