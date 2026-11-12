# Hindi Text Restoration, Sentence Ordering, Grouping & Translation

A clean NLP pipeline that takes Hindi text with missing punctuation and shuffled sentences, then:

1. Restores punctuation and sentence boundaries
2. Recovers a sensible (candidate) sentence order
3. Translates the ordered Hindi sentences into English
4. Keeps every five sentences together as a group

**No off-the-shelf generative chat LLM is used** (no GPT/Gemini/Claude). The project uses encoder/ONNX and sequence-to-sequence translation models, runs **CPU by default**, and if a **GPU** is used, GPU memory must stay **< 1.5 GB VRAM** end-to-end.

**Repository:** [gitanshu-chavda/Punctuation_Adder-and-Sentence-Ordering](https://github.com/gitanshu-chavda/Punctuation_Adder-and-Sentence-Ordering)  
**Branch (current):** `sen-ord-1`  
**Executed demo notebook:** [notebooks/executed_demo.ipynb](https://github.com/gitanshu-chavda/Punctuation_Adder-and-Sentence-Ordering/blob/sen-ord-1/notebooks/executed_demo.ipynb)  
**Runnable end-to-end Colab:** [notebooks/colab_end_to_end.ipynb](https://github.com/gitanshu-chavda/Punctuation_Adder-and-Sentence-Ordering/blob/sen-ord-1/notebooks/colab_end_to_end.ipynb)  
**Google Drive folder (notebooks/artifacts):** [Open in Google Drive](https://drive.google.com/drive/folders/1LjbBl2CQbTBjrl1kEuq2KVE-gG8hByhn?usp=sharing)

---

## What it does, in plain words

Imagine three helpers working on a Hindi passage:

1. **The editor** decides where punctuation and sentence boundaries belong.
2. **The librarian** proposes a more sensible sentence order (a *candidate* reconstruction).
3. **The translator** turns each ordered Hindi sentence into English and groups every five sentences together.

```text
Input:  Hindi text, little/no punctuation, sentences jumbled (+ 67 literal NULL markers preserved)
        ...

Stage 1: add punctuation and sentence boundaries  -> data/punctuated/*.txt
Stage 2: put sentences in a candidate order         -> data/ordered/*.ordered.txt
Stage 3: group every 5 and translate to English     -> outputs/* and data/final/*
```

## The three stages

| Stage | Simple job | Model / Implementation | Notes (from Colab run artifacts) |
| --- | --- | --- | --- |
| **Punctuation** | Predict punctuation/sentence boundaries per word/token | `Horizon-Labs/punctuation-restoration-small` (quantized ONNX, token classification), ONNX Runtime (CPU) via `src/punctuation/restore.py` | Runs on CPU. NULL-adjacent cases need manual review. |
| **Ordering** | Propose a sentence sequence | `bert-base-multilingual-cased` (mBERT) **pairwise scoring + Borda aggregation**, `src/ordering/order.py` | Ran on **Colab GPU (Tesla T4)** in the executed run; per-sample peak VRAM **~0.75–0.77 GB allocated / ~0.92–1.05 GB reserved** (budget 1.5 GB, all under budget). Unsupervised/candidate ordering (not trained from scratch here). |
| **Translation** | Translate ordered Hindi to English (grouped by 5) | **MarianMT fallback:** `Helsinki-NLP/opus-mt-hi-en` via `src/translation/translate.py` (IndicTrans2 preferred: `ai4bharat/indictrans2-indic-en-dist-200M`, configurable). | **Current executed run used MarianMT fallback**. IndicTrans2 can be enabled in Colab with HF access/token if available. |

## Notebooks

| # | Notebook | What it is | Best for |
| ---: | --- | --- | --- |
| 1 | [`executed_demo.ipynb`](notebooks/executed_demo.ipynb) | **Already-executed evidence notebook** with saved outputs from a full Colab run (mBERT ordering on T4, <1.5 GB VRAM, MarianMT translation). | Seeing real results quickly |
| 2 | [`colab_end_to_end.ipynb`](notebooks/colab_end_to_end.ipynb) | Runnable end-to-end Colab template (clone, setup, HF token/IndicTrans2, run, metrics, download). | Reproducing from scratch in Colab |
| 3 | [`00_start_here.ipynb`](notebooks/00_start_here.ipynb) | Environment checks, project map, NULL sanity checks | Getting oriented |
| 4 | [`01_punctuation_demo.ipynb`](notebooks/01_punctuation_demo.ipynb) | Punctuation restoration walkthrough | Understanding Stage 1 |
| 5 | [`02_null_diagnostic.ipynb`](notebooks/02_null_diagnostic.ipynb) | Inspect all 67 `NULL` marker contexts | NULL integrity checks |
| 6 | [`03_sentence_ordering_demo.ipynb`](notebooks/03_sentence_ordering_demo.ipynb) | Sentence ordering walkthrough (mBERT pairwise+Borda) | Understanding Stage 2 |
| 7 | [`04_full_pipeline.ipynb`](notebooks/04_full_pipeline.ipynb) | Inspect full pipeline outputs (ordering, grouping, translation) | End-to-end inspection |

> The executed demo is read-only evidence. For a fresh run, use `colab_end_to_end.ipynb`.

## Quick start

### See results (already generated)
- Aligned outputs: [`outputs/`](outputs/) (`sample-N.sentences_hi/en.txt`, `sample-N.groups_hi/en.txt`)
- Grouped/metadata: [`data/final/`](data/final/) (`*.groups_hi/en.txt`, `*.groups.tsv`, `*.pipeline.json`, `translation_summary.json`)
- Ordered candidates + mappings/stats: [`data/ordered/`](data/ordered/) (`*.ordered.txt`, `*.mapping.tsv`, `*.order_stats.json`, `ordering_summary.json`)
- Reports/metrics: [`reports/`](reports/) (`evaluation.json`, `run_summary.json`, `resource_profile.json`, integrity/review artifacts)

### Run locally (Windows PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
python -m src.pipeline
# Or run stages manually
python src/prepare_input.py
python src/punctuation.py
python src/order_sentences.py
python src/group_and_translate.py
python scripts/generate_aligned_outputs.py
# Checks/metrics
python scripts/verify_integrity.py
python scripts/compute_metrics.py
```

### Run in Colab (recommended for IndicTrans2 / GPU)
1. Open [`notebooks/colab_end_to_end.ipynb`](notebooks/colab_end_to_end.ipynb) in Colab (Runtime: CPU or GPU; if GPU, keep VRAM **< 1.5 GB**).
2. Clone/setup (uses `scripts/colab_setup.sh`). For IndicTrans2, accept the HF model license and set `HF_TOKEN` in Colab Secrets.
3. Run stages (punctuation → ordering → grouping+translation → aligned outputs) and metrics. Use the added VRAM logging in `src/ordering/order.py` to verify usage.
4. Download artifacts or commit back to your branch.

## Dataset, NULL markers & integrity

- **Inputs:** 7 Hindi files in `data/raw/samples/sample-1.txt` … `sample-7.txt` (unpunctuated/jumbled as provided).
- **Literal `NULL` tokens:** **67 total** (unknown meaning). The pipeline **preserves every literal `NULL`** across stages.
- **Integrity (from Colab run):** `scripts/verify_integrity.py` reports **67/67 preserved** (raw → model_input → punctuated → ordered → final Hindi groups). See `reports/evaluation.json` and `reports/null_punct_analysis.json`.

| Sample | Raw NULL | Ordered sentences | 5-sentence groups |
| ---: | ---: | ---: | ---: |
| 1 | 5 | 50 | 10 |
| 2 | 5 | 49 | 10 |
| 3 | 6 | 47 | 10 |
| 4 | 4 | 50 | 10 |
| 5 | 9 | 52 | 11 |
| 6 | 11 | 48 | 10 |
| 7 | 27 | 60 | 12 |
| **Total** | **67** | **356** | **73** |

## Results (from the executed Colab run)

### Ordering (mBERT pairwise + Borda, Colab GPU)
- **Device:** CUDA (Tesla T4) | **Batch size:** 16 | **Max length:** 256 | **Budget:** 1.5 GB
- **Peak VRAM (per-sample allocated):** ~**0.753–0.767 GB**; **reserved ~0.92–1.05 GB** — **all under budget, no over-budget samples** (`data/ordered/ordering_summary.json`)
- **Method:** `src/ordering/order.py` (mBERT pairwise scoring + Borda aggregation). Candidate reconstruction (unsupervised; not trained end-to-end here).

### Translation (current executed run)
- **Backend:** `marian` | **Model:** `Helsinki-NLP/opus-mt-hi-en` | **Grouping:** 5 sentences/group
- **BLEU/chrF:** Not computed in this run (no reference translations available in-repo). **IndicTrans2** (`ai4bharat/indictrans2-indic-en-dist-200M`) is implemented and preferred when available/configured (`src/translation/translate.py`).

### Integrity
- **NULL markers:** **67/67 preserved** across checked stages. See `reports/evaluation.json` and `reports/null_punct_analysis.json`.

> Metrics depend on run configuration (translation backend/access, environment). This repo reports results honestly with caveats in `reports/` and README.

## How it works

### Stage 1: Punctuation restoration
- **Code:** `src/punctuation/restore.py` (compat `src/punctuation.py`)
- **Model:** `Horizon-Labs/punctuation-restoration-small` (quantized ONNX, token classifier) — not a chat/generative LLM
- **Runtime:** ONNX Runtime CPU; tokenization + chunked inference; predicted punctuation inserted at word boundaries; saved to `data/punctuated/`

### Stage 2: Sentence ordering (candidate)
- **Code:** `src/ordering/order.py` (compat `src/order_sentences.py`)
- **Model:** `bert-base-multilingual-cased` (mBERT)
- **Approach:** extract sentences → pairwise mBERT scoring for (i,j) pairs → Borda aggregation → candidate order + mapping/stats in `data/ordered/`
- **CUDA/VRAM:** in Colab GPU run uses our built-in logging/budget checks (prints peak allocated/reserved and flags over-budget if any)

### Stage 3: Grouping and translation
- **Code:** `src/translation/translate.py` (compat `src/group_and_translate.py`)
- **Grouping:** every 5 ordered sentences (last group may be smaller)
- **Translation:** sentence/group translations to English (`outputs/*` aligned, `data/final/*` grouped). Prefers IndicTrans2 if available; falls back to MarianMT if not. No silent hidden fallback beyond configured backend/fallback logic.

## Repository structure

```text
data/           raw/samples/, model_input/, punctuated/, ordered/, final/, processed/
models/         punctuation/, translation/ (cached models; git-ignored if large)
notebooks/      executed_demo.ipynb, colab_end_to_end.ipynb, 00–04 walkthroughs
outputs/        aligned outputs (sample-N.sentences_hi/en.txt, sample-N.groups_hi/en.txt)
reports/        evaluation.json, run_summary.json, resource_profile.json, integrity/review artifacts
scripts/        pipeline utilities, diagnostics, executed-demo builder
src/
  punctuation/  restore.py
  ordering/     order.py  (mBERT pairwise+Borda + CUDA/VRAM logging)
  translation/  translate.py
  prepare_input.py, pipeline.py, evaluate.py, utils.py
main.py, setup_check.py, requirements*.txt
```

## Submission

- **GitHub repository:** [gitanshu-chavda/Punctuation_Adder-and-Sentence-Ordering](https://github.com/gitanshu-chavda/Punctuation_Adder-and-Sentence-Ordering) (branch `sen-ord-1`)
- **Google Drive (notebooks/artifacts):** [1LjbBl2CQbTBjrl1kEuq2KVE-gG8hByhn](https://drive.google.com/drive/folders/1LjbBl2CQbTBjrl1kEuq2KVE-gG8hByhn?usp=sharing)
- **Trained/fine-tuned models:** *None in this submission.* Pretrained/downloaded models and configured backends are used (ONNX punctuation, mBERT, MarianMT/IndicTrans2).
- **Executed evidence:** See [`notebooks/executed_demo.ipynb`](notebooks/executed_demo.ipynb) and reports/artifacts in `outputs/`, `reports/`, `data/ordered/`, `data/final/`.

## Honest limits

- **Punctuation:** approximate/proxy evaluation in places; manual review needed, especially around `NULL` and dialogue.
- **Ordering:** candidate reconstruction (mBERT pairwise + Borda). In the Colab bundle’s evaluation, reference-alignment metrics were not computed (no references available in that bundle/run); ordering quality should be reviewed manually. Implemented method with VRAM-aware logging for GPU runs.
- **Translation:** current executed run used **MarianMT fallback**; **IndicTrans2** supported but not used in this run (requires HF access/license). No BLEU/chrF reported from available references.
- **Resources/constraints:** ordering ran on Tesla T4 with peak allocated VRAM **< 0.8 GB / < 1.5 GB budget (OK)**. If GPU used anywhere, keep VRAM **< 1.5 GB** and measure/profile. Resource profiling in `reports/resource_profile.json` includes prepare/integrity/metrics steps from that run.
- **Data ambiguity:** `NULL` markers have unknown semantics and must be preserved (67/67).

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `IndicTransToolkit` install/build errors (Windows) | Use Colab (`scripts/colab_setup.sh`) or follow IndicTrans2 installation guidance; pipeline falls back to MarianMT if unavailable. |
| IndicTrans2 401/gated | Accept model license on Hugging Face and set `HF_TOKEN`. |
| Ordering slow on CPU | Expected for pairwise mBERT scoring; use GPU in Colab with conservative batch/length and monitor VRAM (< 1.5 GB). |
| NULL/format questions | Check `reports/null_punct_analysis.json`, `reports/punctuation_manual_review.md`, and `scripts/verify_integrity.py`. |
| Alignment issues | Use `outputs/sample-N.sentences_hi.txt` / `sentences_en.txt` (line-aligned) and verify counts against `reports/run_summary.json`. |

## Learn more

- Project guide: [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)
- Notebooks: [`notebooks/`](notebooks/) (start with `executed_demo.ipynb` or `00_start_here.ipynb`)
- Scripts & utilities: [`scripts/`](scripts/)
- Learning materials: [`docs/learning/`](docs/learning/)
