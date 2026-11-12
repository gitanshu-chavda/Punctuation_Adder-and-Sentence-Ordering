# Hindi Punctuation, Sentence Ordering, and Translation

An end-to-end NLP pipeline for the faculty assignment: restore punctuation in unpunctuated/jumbled Hindi text, recover a candidate sentence order, group every 5 ordered sentences, and translate each group to English — while preserving all original data (including `NULL` markers) and following project constraints.

> **Constraints:** No off-the-shelf generative chat LLMs (GPT/Gemini/Claude). Any model used must stay under **1.5 GB GPU RAM** if GPU is used. Current pipeline scripts run on **CPU** by default.

---

## Problem statement (target)

> You are given a piece of text where sentences do not have any punctuation and sentences are jumbled in the zip file. You need to identify punctuation markers at appropriate places. After this step, you need find the order of sentences. Combine 5 sentences each that are in order and translate them into English. You cannot use any off-the-shelf LLM such as GPT, Gemini, or Claude. Any model you use should utilize less than 1.5GB GPU RAM. You are free to develop your own model for punctuation identification, but it should also follow the constraints.

## What this repository delivers

| Stage | Status | Implementation | Main outputs |
|---|---|---|---|
| **1. Punctuation restoration** | Implemented (quality review pending) | `Horizon-Labs/punctuation-restoration-small` via quantized ONNX, CPU (`src/punctuation.py`) | `data/punctuated/sample-*.txt` |
| **2. Sentence ordering** | Implemented as **candidate** order (not ground-truth proven) | Current: sentence-embedding + clustering/greedy heuristics (`src/order_sentences.py`). *Planned/next upgrade requested:* pairwise scoring with multilingual BERT + Borda aggregation for improved ordering. | `data/ordered/*.ordered.txt`, `*.mapping.tsv`, `*.order_stats.json` |
| **3. Grouping (5 sentences)** | Implemented | Groups every 5 consecutive ordered sentences per sample (last group may be smaller) (`src/group_and_translate.py`) | `data/final/*.groups_hi.txt`, `*.groups.tsv` |
| **4. Hindi → English translation** | Implemented (quality review pending) | Current: `Helsinki-NLP/opus-mt-hi-en` MarianMT seq2seq on CPU. *Planned/next upgrade requested:* IndicTrans2 (gated HF access/token may be required). | `data/final/*.groups_en.txt`, `*.groups.tsv` |

**Important:** Successful execution ≠ correctness. Punctuation placement, sentence order, and translation quality must be manually reviewed/evaluated.

---

## Approach (current pipeline)

### 1) Input preparation and data integrity
- Raw faculty files: `data/raw/samples/sample-1..7.txt` (UTF-8, largely unpunctuated/shuffled content).
- Prepared copies: `src/prepare_input.py` copies raw → `data/model_input/` without modifying content.
- **`NULL` markers:** the dataset contains **67 literal `NULL` tokens** with unknown meaning. We preserve them end-to-end and provide diagnostics (`scripts/null_diagnostic.py`, `notebooks/02_null_diagnostic.ipynb`). Punctuation may be inserted adjacent to `NULL` in outputs; this is observed behavior, not proof of meaning.

### 2) Punctuation restoration
- **Model:** `Horizon-Labs/punctuation-restoration-small` (encoder token-classification model; not GPT/Gemini/Claude).
- **Runtime:** quantized ONNX model (`models/punctuation/onnx/model_quantized.onnx`) with local tokenizer (`models/punctuation/tokenizer/`) via ONNX Runtime **CPUExecutionProvider**.
- **Labels:** `0` (none), `.`, `,`, `?`, `-`, `:`. Predicted `.` is converted to Hindi danda `।`.
- **Method:** tokenize with offsets, run chunked inference for long inputs (`CHUNK_SIZE=400`, `OVERLAP_SIZE=50`), map token predictions to character boundaries, insert punctuation at word boundaries, and save per-sample outputs.
- **Notes:** whitespace is normalized in `restore()`; punctuation around `NULL` and dialogue samples (6–7) require careful review.

### 3) Sentence ordering (candidate)
- **Input:** punctuated text (`data/punctuated/*.txt`) split into sentence-like units on `।`/`?` boundaries (with handling for line-preserved segments).
- **Current method (`src/order_sentences.py`):** encode sentences using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual sentence embeddings), cluster into topical groups (KMeans), then greedily order within/between groups using similarity and light discourse heuristics.
- **Output:** candidate ordered text + mapping back to original positions + topic/stats. This is a **heuristic reconstruction**, not guaranteed original order.
- **Requested upgrade:** replace/augment with **pairwise scoring using multilingual BERT** and **Borda aggregation** to improve global ordering quality (implementation planned next).

### 4) Grouping and translation
- **Grouping:** every 5 consecutive sentences in the candidate ordered sequence form one group (`src/group_and_translate.py`); final group may contain fewer than 5.
- **Current translation:** `Helsinki-NLP/opus-mt-hi-en` (MarianMT encoder-decoder MT model, CPU, via `transformers`). Not a GPT/Claude/Gemini chat LLM. Model may be downloaded/cached locally on first run (e.g. `models/translation/opus-mt-hi-en`, ignored by git).
- **Requested upgrade:** switch translation to **IndicTrans2** (e.g. AI4Bharat IndicTrans2 models). This may require gated Hugging Face access/token and additional toolkit dependencies; configuration/token must be provided in the environment.
- **Outputs:** Hindi groups, English groups, TSV alignment, and per-file/summary JSON in `data/final/`.

---

## Quantitative summary (current generated artifacts)

### Integrity and pipeline coverage
- **Samples processed:** 7/7 (`sample-1` to `sample-7`)
- **Literal `NULL` preserved:** **67/67** across `raw`, `model_input`, `punctuated`, `ordered`, and `final` Hindi groups (per `scripts/verify_integrity.py`)

| Sample | Raw NULL | Punct NULL | Ordered NULL | Final HI NULL | Punct chars | Ordered sentences | 5-sentence groups |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 5 | 5 | 3851 | 50 | 10 |
| 2 | 5 | 5 | 5 | 5 | 3881 | 49 | 10 |
| 3 | 6 | 6 | 6 | 6 | 3989 | 47 | 10 |
| 4 | 4 | 4 | 4 | 4 | 5504 | 50 | 10 |
| 5 | 9 | 9 | 9 | 9 | 5079 | 52 | 11 |
| 6 | 11 | 11 | 11 | 11 | 2422 | 48 | 10 |
| 7 | 27 | 27 | 27 | 27 | 4000 | 60 | 12 |

### Punctuation output counts (model predictions)
| Sample | `।` (danda) | `,` | `?` | `:` | `-` | `NULL` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 5 | 1 | 0 | 0 | 5 |
| 2 | 1 | 8 | 1 | 0 | 0 | 5 |
| 3 | 0 | 12 | 0 | 0 | 0 | 6 |
| 4 | 0 | 22 | 0 | 1 | 2 | 4 |
| 5 | 0 | 25 | 0 | 1 | 5 | 9 |
| 6 | 9 | 32 | 9 | 13 | 0 | 11 |
| 7 | 7 | 15 | 7 | 14 | 0 | 27 |

> Note: counts reflect inserted/predicted characters in outputs (including cases adjacent to `NULL`) and are **not** accuracy metrics.

### Proxy punctuation evaluation (`scripts/evaluate_punctuation.py`)
This script compares predictions to available processed sentence references (`data/processed/sample-*.sentences.txt`) using a rough alignment/heuristic. **Treat as indicative only, not gold-standard evaluation.**

| Sample | Char similarity | Proxy mark acc. | Mark match/total | Pred counts (`।, , ? -`) | Gold counts (`।, , ? -`) |
|---:|---:|---:|---:|---|---|
| 1 | 0.999 | 0.911 | 51/56 | {49,5,1,0} | {49,4,0,0} |
| 2 | 0.999 | 0.871 | 54/62 | {48,8,1,0} | {50,11,0,0} |
| 3 | 0.999 | 0.934 | 57/61 | {47,12,0,0} | {49,10,0,0} |
| 4 | 0.999 | 0.929 | 39/42 | {50,22,0,2} | {50,21,0,0} |
| 5 | 0.999 | 0.000* | 0/1* | {52,25,0,5} | {52,22,0,0} |
| 6 | 0.753 | 0.866 | 58/67 | {39,32,9,0} | {43,28,10,0} |
| 7 | 0.996 | 0.727 | 16/22 | {52,15,7,0} | {58,9,9,0} |
| **Average** | **0.963** | **0.884** | **275/311** | - | - |

\*Sample-5 proxy row has an anomalous denominator in this heuristic; inspect manually.

### Ordering / translation metrics
- **Ordering quality metrics:** not yet computed (no Kendall tau / pair accuracy / manual ordering benchmark recorded yet). Current ordering is a candidate heuristic.
- **Translation quality metrics:** not yet computed (no BLEU/chrF/human adequacy scores recorded yet). Current outputs are available for review in `data/final/*.groups_en.txt` and `*.tsv`.
- **Resource metrics:** formal runtime and peak memory report (CPU/GPU) not yet recorded; scripts currently target CPU execution.

---

## Repository structure

```
data/
  raw/samples/          # Original faculty-provided files (read-only)
  model_input/          # Prepared copies for punctuation (NULL preserved)
  punctuated/           # Punctuation-restored outputs
  ordered/              # Candidate sentence-ordering outputs for all samples
  final/                # 5-sentence Hindi groups + English translations
  processed/            # Intermediate/reference artifacts (e.g. sentence refs, eval aids)
  external/             # Optional external corpora

models/
  punctuation/
    onnx/model_quantized.onnx
    tokenizer/
    finetuned-quick/            # Experimental (optional)
  translation/                  # Local cached translation model (git-ignored)

src/
  prepare_input.py
  punctuation.py
  order_sentences.py            # Current candidate ordering (all samples)
  group_and_translate.py        # Grouping + translation
  sentence-ordering.py          # Earlier sample-1 script (reference)

scripts/
  run_full_pipeline.py
  verify_integrity.py
  null_diagnostic.py
  evaluate_punctuation.py
  ...

notebooks/
  00_start_here.ipynb
  01_punctuation_demo.ipynb
  02_null_diagnostic.ipynb
  03_sentence_ordering_demo.ipynb
  04_full_pipeline.ipynb

docs/
  PROJECT_GUIDE.md
```

---

## How to run (Windows PowerShell, project root)

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Full pipeline (prepare -> punctuate -> order -> group+translate)
python scripts/run_full_pipeline.py

# Or run stages individually
python src/prepare_input.py
python src/punctuation.py
python src/order_sentences.py
python src/group_and_translate.py

# Diagnostics / checks
python scripts/null_diagnostic.py
python scripts/verify_integrity.py
python scripts/evaluate_punctuation.py
```

## Notebooks (beginner-friendly)

- `notebooks/00_start_here.ipynb` — setup checks, project map, NULL sanity checks
- `notebooks/01_punctuation_demo.ipynb` — punctuation restoration walkthrough
- `notebooks/02_null_diagnostic.ipynb` — inspect all 67 `NULL` contexts
- `notebooks/03_sentence_ordering_demo.ipynb` — candidate ordering walkthrough
- `notebooks/04_full_pipeline.ipynb` — inspect ordering, 5-sentence groups, and translations

---

## Comparison notes (vs similar project READMEs) and current gaps

Compared to a more presentation-complete reference README for a similar assignment, this project currently:
- **Has:** end-to-end implemented pipeline, all-sample outputs, NULL-preservation checks, beginner notebooks, and runnable scripts.
- **Is missing / needs improvement:**
  - **Quantitative evaluation for all stages** (ordering metrics like pair accuracy/Kendall tau, translation BLEU/chrF or human evaluation, stronger punctuation evaluation).
  - **Consolidated reports** (e.g. `reports/evaluation.json`, `reports/run_summary.json`, data-prep summary) and measured runtime/memory evidence.
  - **Colab/executed-demo notebooks** and setup/verification scripts for turnkey reproduction.
  - **Per-sentence aligned output format** (some projects emit aligned `sentences_hi/en` + `groups_hi/en` files; we currently emit grouped files/TSV/JSON).
  - **Requested model upgrades:** pairwise multilingual-BERT+Borda ordering and IndicTrans2 translation (planned; IndicTrans2 may require HF token/gated access and compatible dependencies).

## Known limitations
- Punctuation quality varies by text type and can insert punctuation around `NULL`; manual review required.
- Ordering is currently heuristic/candidate and not validated against original sequence.
- Translation quality depends on punctuation/order and current MT model choice; not benchmark-scored yet.
- Proxy evaluation script is approximate and should not be treated as gold-standard performance reporting.

## Reproducibility / compliance notes
- No GPT/Gemini/Claude used in pipeline scripts.
- Current implementations run on CPU; keep total GPU memory < 1.5 GB if ported to GPU (measure peak usage).
- Raw data is not modified; `NULL` markers are preserved in generated artifacts.
- Some models may download/cache on first run (sentence encoder, translation model). Local translation model cache is not committed.