# Hindi Text Restoration and Translation

A beginner-friendly NLP project for the assignment text: take Hindi text with missing punctuation and jumbled sentences, then:

1. restore punctuation and sentence boundaries,
2. recover a likely/candidate sentence order,
3. translate the result into English,
4. keep every five sentences together as a group.

**No off-the-shelf generative chat LLM is used** (no GPT/Gemini/Claude). The pipeline uses encoder/MT models and runs on **CPU by default** (GPU usage, if introduced, must stay **< 1.5 GB VRAM** end-to-end).

**Repository:** https://github.com/gitanshu-chavda/Punctuation_Adder-and-Sentence-Ordering  
**Branch (current work):** `sen-ord-1`

---

## What it does, in plain words

Imagine three helpers working on the same Hindi passage:

1. **The editor** decides where punctuation marks (like `।`, `,`, `?`, `:`) belong.
2. **The librarian** tries to place sentences in a sensible/logical order (a *candidate* order).
3. **The translator** turns each Hindi sentence/group into English while preserving the order and grouping.

```text
Input:  Hindi text, little/no punctuation, sentences jumbled (+ 67 literal NULL markers preserved)
        ...

Stage 1: add punctuation and sentence boundaries
        ... restored punctuation (data/punctuated/*.txt)

Stage 2: put sentences in a candidate order
        ... reordered sentences (data/ordered/*.ordered.txt)

Stage 3: group every 5 sentences and translate to English
        ... aligned outputs (outputs/*) and grouped translations (data/final/*, outputs/*)
```

## The three stages

| Stage | Simple job | Current implementation (this repo) | Main quantitative notes |
| --- | --- | --- | --- |
| **Punctuation** | Predict punctuation after words/tokens | `Horizon-Labs/punctuation-restoration-small` (quantized ONNX, CPU, token classification) via `src/punctuation.py` | Proxy mark accuracy **0.884** (275/311), avg char similarity **0.963** on `scripts/evaluate_punctuation.py` (approximate, not gold-standard) |
| **Ordering** | Propose a likely sentence sequence | **Implemented now:** multilingual BERT pairwise scoring + Borda aggregation (`bert-base-multilingual-cased`) in `src/order_sentences.py`. **Existing generated `data/ordered/*` artifacts:** previous embedding+heuristic candidate run. | Reference-alignment metrics vs `data/processed/*.sentences.txt` (best-effort matching, matched coverage **0.897**): macro **pairwise accuracy 0.457**, macro **Kendall tau -0.085** (see caveats below) |
| **Translation** | Translate ordered Hindi to English (grouped by 5) | **Implemented:** IndicTrans2-preferred backend in `src/group_and_translate.py` with fallback to `Helsinki-NLP/opus-mt-hi-en` (MarianMT, non-LLM). **Current generated aligned/final outputs:** MarianMT fallback run (IndicTrans2 requires HF access/token + `IndicTransToolkit`). | No reference translation benchmark yet → **BLEU / chrF: not measured** |

> These numbers are reported honestly for the current artifacts/scripts available in this repo. Where metrics are proxy/unavailable, this README states that explicitly.

---

## Dataset and important `NULL` marker note

- **Input:** 7 Hindi text files in `data/raw/samples/sample-1.txt` … `sample-7.txt` (unpunctuated/jumbled as provided).
- **Literal `NULL` tokens:** 67 total across all samples (meaning unknown).  
  **Rule:** never delete/replace/reinterpret `NULL` without evidence. The pipeline preserves them.
- **Integrity:** `scripts/verify_integrity.py` reports `67/67` literal `NULL` preserved across raw → model_input → punctuated → ordered → final Hindi groups.

| Sample | Raw NULL | Punct NULL | Ordered NULL | Final HI NULL | Ordered sentences | 5-sentence groups |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 5 | 5 | 50 | 10 |
| 2 | 5 | 5 | 5 | 5 | 49 | 10 |
| 3 | 6 | 6 | 6 | 6 | 47 | 10 |
| 4 | 4 | 4 | 4 | 4 | 50 | 10 |
| 5 | 9 | 9 | 9 | 9 | 52 | 11 |
| 6 | 11 | 11 | 11 | 11 | 48 | 10 |
| 7 | 27 | 27 | 27 | 27 | 60 | 12 |

---

## What you get (outputs)

### Friend-style aligned outputs (`outputs/`)
For each sample, the repo generates 4 aligned files:

| File | What it contains |
| --- | --- |
| `outputs/sample-N.sentences_hi.txt` | One Hindi sentence per line (candidate ordered) |
| `outputs/sample-N.groups_hi.txt` | Same Hindi sentences, five per line/group |
| `outputs/sample-N.sentences_en.txt` | One English translation per line (aligned to Hindi sentences) |
| `outputs/sample-N.groups_en.txt` | Same English translations, five per line/group |

Line *i* in `sentences_hi` corresponds to line *i* in `sentences_en`.

### Pipeline/intermediate outputs
- `data/model_input/` — copied inputs (NULL preserved)
- `data/punctuated/` — punctuation-restored text
- `data/ordered/` — candidate ordered sentences, mappings, stats (`*.ordered.txt`, `*.mapping.tsv`, `*.order_stats.json`, `ordering_summary.json`)
- `data/final/` — grouped Hindi/English outputs and summaries (`*.groups_hi.txt`, `*.groups_en.txt`, `*.groups.tsv`, `*.pipeline.json`, `translation_summary.json`)

### Reports (`reports/`)
- `reports/evaluation.json` — quantitative/proxy metrics and status for punctuation/ordering/translation + integrity notes
- `reports/run_summary.json` — run/pipeline summary and current artifact status
- `reports/data_prep_summary.json` — dataset/prep summary
- `reports/resource_profile.json` — partial CPU/runtime/memory profiling (lightweight checks + `prepare` and `punctuation` stage timings; full ordering/translation heavy-stage and GPU VRAM profiling still pending)
- `reports/null_punct_analysis.json` / `reports/punctuation_manual_review.md` — automated-assisted NULL/punctuation review artifacts
- `reports/manual_review_checklist.md` / `reports/translation_review_template.csv` — manual review and translation quality review scaffolding

---

## Results to remember (current)

### Punctuation (proxy evaluation)
Source: `scripts/evaluate_punctuation.py` vs `data/processed/sample-*.sentences.txt` references (approximate alignment; **not gold-standard**).

| Sample | Char similarity | Proxy mark acc. | Match/Total | Pred (`।, , ? -`) | Gold (`।, , ? -`) |
|---:|---:|---:|---:|---|---|
| 1 | 0.999 | 0.911 | 51/56 | {49,5,1,0} | {49,4,0,0} |
| 2 | 0.999 | 0.871 | 54/62 | {48,8,1,0} | {50,11,0,0} |
| 3 | 0.999 | 0.934 | 57/61 | {47,12,0,0} | {49,10,0,0} |
| 4 | 0.999 | 0.929 | 39/42 | {50,22,0,2} | {50,21,0,0} |
| 5 | 0.999 | 0.000* | 0/1* | {52,25,0,5} | {52,22,0,0} |
| 6 | 0.753 | 0.866 | 58/67 | {39,32,9,0} | {43,28,10,0} |
| 7 | 0.996 | 0.727 | 16/22 | {52,15,7,0} | {58,9,9,0} |
| **Average** | **0.963** | **0.884** | **275/311** | - | - |

\*Sample-5 row is affected by an anomalous heuristic denominator in this proxy script; manual inspection required.

**How to read this honestly:**
- This is a useful rough signal, but it is **not** a definitive punctuation quality benchmark.
- Punctuation around `NULL` and dialogue-style samples (6–7) needs manual review.

### Ordering (reference-alignment metrics, best-effort)
Source: `scripts/compute_metrics.py` compares `data/ordered/*.ordered.txt` to `data/processed/sample-*.sentences.txt` by canonical text matching (matched sentences only).  
**Caveat:** This measures alignment to the available processed reference sequence; if that reference is not verified ground-truth original order, interpret as reference-alignment rather than definitive ordering quality. Metrics below are for the **current regenerated mBERT pairwise+Borda artifacts**.

| Sample | Ref sents | Cand sents | Matched | Coverage | Kendall tau | Pairwise acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49 | 50 | 48 | 0.980 | 0.074 | 0.537 |
| 2 | 50 | 49 | 45 | 0.900 | 0.105 | 0.553 |
| 3 | 49 | 47 | 46 | 0.939 | -0.132 | 0.434 |
| 4 | 50 | 50 | 50 | 1.000 | -0.202 | 0.399 |
| 5 | 52 | 52 | 52 | 1.000 | -0.041 | 0.480 |
| 6 | 53 | 48 | 41 | 0.774 | 0.066 | 0.533 |
| 7 | 67 | 60 | 46 | 0.687 | 0.111 | 0.556 |
| **Macro avg** | - | - | - | **0.897** | **-0.003** | **0.499** |

- **Method:** mBERT (`bert-base-multilingual-cased`) **pairwise scoring + Borda aggregation** (`src/order_sentences.py`), regenerated for all samples (CPU-intensive). Results vary by sample and remain near-chance overall on this reference alignment.

### Translation
- **Implemented backend:** **IndicTrans2-preferred** (`ai4bharat/indictrans2-indic-en-dist-200M` default) with fallback to `Helsinki-NLP/opus-mt-hi-en` MarianMT (`src/group_and_translate.py`).
- **Current generated outputs:** produced via MarianMT fallback in this environment (IndicTrans2 requires HF token/license access and `IndicTransToolkit`).
- **Metrics:** BLEU / chrF **not measured yet** (no reference translation benchmark recorded).

---

## How it works

### Stage 1: punctuation restoration
- **Script:** `src/punctuation.py`
- **Model:** `Horizon-Labs/punctuation-restoration-small` (encoder token classifier; not a chat/generative LLM)
- **Runtime:** quantized ONNX + local tokenizer, ONNX Runtime `CPUExecutionProvider`
- **Labels:** predicts `0`, `.`, `,`, `?`, `-`, `:` and converts `.` to Hindi danda `।`
- **Process:** tokenize with offsets → chunked inference for long text → insert predicted punctuation at word boundaries → save to `data/punctuated/`

### Stage 2: sentence ordering (candidate)
- **Script:** `src/order_sentences.py`
- **Model:** `bert-base-multilingual-cased` (multilingual BERT)
- **Process:** extract sentence-like units → compute **pairwise scores** for sentence pairs using mBERT pair encodings/features → aggregate preferences with **Borda-style ranking** → emit candidate ordered sequence + mapping
- **Output:** `data/ordered/*.ordered.txt`, `*.mapping.tsv`, `*.order_stats.json`  
- **Caveat:** candidate order only; original/logical correctness must be reviewed.

### Stage 3: grouping and translation
- **Script:** `src/group_and_translate.py`
- **Grouping:** every 5 consecutive ordered sentences (last group may be smaller)
- **Translation:** sentence-level aligned translations generated for `outputs/` and group translations for `data/final/` and `outputs/`
  - **Preferred:** IndicTrans2 (`ai4bharat/indictrans2-indic-en-dist-200M`, configurable)
  - **Fallback:** MarianMT `Helsinki-NLP/opus-mt-hi-en` (used for currently generated outputs)
- **Caveat:** translation quality depends on punctuation/order and model availability/configuration.

---

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [`notebooks/00_start_here.ipynb`](notebooks/00_start_here.ipynb) | Environment checks, project map, NULL sanity checks |
| [`notebooks/01_punctuation_demo.ipynb`](notebooks/01_punctuation_demo.ipynb) | Punctuation restoration walkthrough |
| [`notebooks/02_null_diagnostic.ipynb`](notebooks/02_null_diagnostic.ipynb) | Inspect all 67 `NULL` marker contexts |
| [`notebooks/03_sentence_ordering_demo.ipynb`](notebooks/03_sentence_ordering_demo.ipynb) | Sentence ordering walkthrough |
| [`notebooks/04_full_pipeline.ipynb`](notebooks/04_full_pipeline.ipynb) | Inspect full pipeline outputs (ordering, grouping, translation) |
| [`notebooks/colab_end_to_end.ipynb`](notebooks/colab_end_to_end.ipynb) | Run the full pipeline in Google Colab (clone, setup, HF token/IndicTrans2, run, metrics, download) |

> Open with the project `venv` kernel in VS Code/Antigravity for best compatibility.

---

## Quick start

### See results (already generated)
- Aligned outputs: [`outputs/`](outputs/)
- Grouped/metadata outputs: [`data/final/`](data/final/)
- Ordered candidates: [`data/ordered/`](data/ordered/)
- Reports: [`reports/`](reports/)

### Run locally (Windows PowerShell, project root)
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run end-to-end (wrapper)
python -m src.pipeline

# Or run stages manually
python src/prepare_input.py
python src/punctuation.py
python src/order_sentences.py
python src/group_and_translate.py
python scripts/generate_aligned_outputs.py

# Checks / diagnostics / metrics / profiling
python scripts/verify_integrity.py
python scripts/null_diagnostic.py
python scripts/evaluate_punctuation.py
python scripts/compute_metrics.py
python scripts/profile_resources.py --steps integrity,metrics
```

### IndicTrans2 notes (translation upgrade path)
- Set `HF_TOKEN` if the selected IndicTrans2 model is gated and accept the model license on Hugging Face.
- Install required toolkit/dependencies (e.g. `IndicTransToolkit`) compatible with your environment.
- Optional env vars:
  - `TRANSLATION_BACKEND=auto|indictrans2|marian`
  - `INDICTRANS2_MODEL_ID=...`
  - `INDICTRANS2_LOCAL_DIR=...`

If IndicTrans2 is unavailable, the pipeline falls back to MarianMT. In the current local Windows environment, IndicTrans2 execution was blocked by gated model access/401 and `IndicTransToolkit` build requirements (MSVC tools); **running in Colab is recommended** (see `notebooks/colab_end_to_end.ipynb` and `scripts/colab_setup.sh`).

---

## Repository structure

```text
baseline/               baseline notes/placeholders
data/
  raw/samples/          original input files (read-only)
  model_input/          prepared inputs (NULL preserved)
  punctuated/           punctuation-restored outputs
  ordered/              candidate ordered sentences + mappings/stats
  final/                grouped Hindi/English outputs + summaries
  processed/            intermediate/reference artifacts
docs/
  learning/             learning materials (extensible)
  PROJECT_GUIDE.md      detailed project guide
lessons/                lesson materials/placeholders
models/
  punctuation/          ONNX model + tokenizer (and experimental artifacts)
  translation/          local cached translation models (git-ignored if large)
notebooks/              walkthrough/demo notebooks
outputs/                friend-style aligned outputs (4 files per sample)
reports/                evaluation/run/data-prep summaries (JSON)
scripts/                pipeline utilities and diagnostics
src/
  prepare_input.py
  punctuation.py
  order_sentences.py
  group_and_translate.py
  pipeline.py           unified CLI wrapper
  sentence-ordering.py  earlier sample-1 script (reference)
main.py                 entry point wrapper
```

---

## Honest limits

- **Punctuation is not perfect.** Proxy metrics are encouraging (avg proxy mark acc. 0.884, char sim 0.963), but evaluation is approximate and manual review is required (especially `NULL`-adjacent punctuation and dialogue samples 6–7). `NULL` semantics remain unknown; handling is intentionally cautious/preservative. Automated-assisted review artifacts are available in `reports/null_punct_analysis.json` and `reports/punctuation_manual_review.md`.
- **Ordering is a candidate reconstruction.** mBERT pairwise+Borda artifacts have been **regenerated** for all samples. Reference-alignment metrics (`scripts/compute_metrics.py`, best-effort vs `data/processed/*.sentences.txt`, coverage **0.897**): macro pairwise acc. **0.499**, Kendall tau **-0.003** (varies by sample, near-chance overall on this reference). Not validated against a verified original-order ground truth.
- **Translation quality is not benchmarked.** No BLEU/chrF/human-scored benchmark yet. Current regenerated outputs used **MarianMT fallback**; IndicTrans2 backend is implemented but execution is blocked locally (gated HF access/401 for the tested model and `IndicTransToolkit` MSVC build requirements). Colab notebook/setup added for IndicTrans2-capable runs. Review template: `reports/translation_review_template.csv`.
- **Resource proof is partial.** CPU-targeted profiling recorded in `reports/resource_profile.json`: `prepare` ~0.115s, `punctuation` ~123.7s (7 samples, ONNX CPU), plus lightweight checks (`verify_integrity.py` ~0.13s, `compute_metrics.py` ~0.28s). Full per-stage runtime/peak RAM for mBERT ordering and translation runs, and **GPU VRAM** (must remain < 1.5 GB if used), are not fully profiled yet.
- **Dataset-specific ambiguity:** `NULL` markers have unknown semantics and must be preserved; automated handling remains cautious. Use `reports/manual_review_checklist.md` for review tracking.

---

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `sentencepiece` / tokenizer/model loading errors | Ensure `venv` dependencies are installed; check model cache paths under `models/` and Hugging Face cache |
| IndicTrans2 model access denied / gated | Accept model license on Hugging Face and set `HF_TOKEN`; install `IndicTransToolkit` |
| IndicTrans2 falls back to MarianMT | Check logs from `src/group_and_translate.py`; verify `TRANSLATION_BACKEND`, token, and dependencies |
| Unicode printing issues on Windows console | Scripts configure UTF-8 stdout where possible; prefer writing to files/notebooks for inspection |
| Ordering step is slow on CPU | mBERT pairwise scoring is compute-heavy; run on fewer samples first or use a machine with more CPU capacity |
| Output alignment questions | Use `outputs/sample-N.sentences_hi.txt` and `outputs/sample-N.sentences_en.txt` (line-aligned) and verify counts |

---

## Learn more

- Project guide: [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)
- Learning notes (extensible): [`docs/learning/`](docs/learning/)
- Start with notebooks in [`notebooks/`](notebooks/) for a step-by-step walkthrough of the pipeline and diagnostics.