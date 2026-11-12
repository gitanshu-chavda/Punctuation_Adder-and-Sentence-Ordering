# Hindi Punctuation Adder and Sentence Ordering

A beginner-friendly project for restoring punctuation in Hindi text and (later) reconstructing sentence order, grouping, and translation under strict constraints.

## Constraints (must follow)

- **No off-the-shelf generative LLMs** (no GPT/Gemini/Claude for generation).
- **GPU memory < 1.5 GB** total if GPU is used. Current punctuation inference runs on **CPU via ONNX Runtime**.
- **Preserve original data**: never modify `data/raw/samples/`.
- **Preserve `NULL` markers**: the dataset contains 67 literal `NULL` tokens with unknown meaning. Do not delete or reinterpret them without evidence.
- **Do not assume correctness**: always inspect outputs and run diagnostics/evaluation.

## Repository structure

```
data/
  raw/samples/          # Original faculty-provided files (read-only)
  model_input/          # Prepared copies for punctuation (NULL preserved)
  punctuated/           # Punctuation-restored outputs
  ordered/              # Candidate sentence-ordering outputs for all samples (.ordered.txt, .mapping.tsv, stats)
  final/                # Grouped (5-sentence) Hindi and translated English outputs (.tsv, summaries)
  processed/            # Intermediate/derived artifacts (e.g. sentence splits, eval refs)
  external/             # Optional external corpora (not required for current pipeline)

models/
  punctuation/
    onnx/model_quantized.onnx   # Quantized ONNX punctuation model (used for inference)
    tokenizer/                  # Local tokenizer files
    finetuned-quick/            # Experimental fine-tuned checkpoint (optional)
  translation/                  # Local cached translation model (created on first translation run; ignored by git)

src/                    # Core reusable Python modules
  prepare_input.py      # Copy raw -> model_input, preserving NULL
  punctuation.py        # Punctuation restoration (ONNX, CPU)
  order_sentences.py    # Candidate sentence ordering for all samples
  group_and_translate.py# Group 5 ordered sentences + Hindi->English translation
  sentence-ordering.py  # Earlier sample-1 ordering script (kept for reference)

scripts/                # Utility/evaluation scripts
  run_full_pipeline.py
  verify_integrity.py
  evaluate_punctuation.py
  null_diagnostic.py
  ...

notebooks/              # Beginner-friendly walkthroughs (run after understanding src/)
  00_start_here.ipynb
  01_punctuation_demo.ipynb
  02_null_diagnostic.ipynb
  03_sentence_ordering_demo.ipynb
  04_full_pipeline.ipynb

docs/
  PROJECT_GUIDE.md      # Detailed guide and current status

requirements.txt
```

## Quick start (Windows PowerShell, from project root)

```powershell
# 1) Activate existing venv (recommended)
.\venv\Scripts\Activate.ps1

# 2) Prepare inputs (preserves NULL markers)
python src/prepare_input.py

# 3) Run punctuation restoration for all samples (CPU/ONNX)
python src/punctuation.py

# 4) Order sentences for all samples (candidate)
python src/order_sentences.py

# 5) Group every 5 ordered sentences and translate Hindi -> English (CPU)
python src/group_and_translate.py

# Optional diagnostics / checks
python scripts/null_diagnostic.py
python scripts/verify_integrity.py
python scripts/evaluate_punctuation.py

# Run the full implemented pipeline in one command
python scripts/run_full_pipeline.py
```

> Notes:
> - Sentence ordering uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (downloaded/cached on first run if needed).
> - Translation uses `Helsinki-NLP/opus-mt-hi-en` (MarianMT seq2seq model, not a GPT/Claude/Gemini chat LLM). On first run it may download and save a local copy under `models/translation/opus-mt-hi-en` if allowed. These model files are not committed to git.

> Note: Scripts may download/cache Hugging Face models on first run (e.g. sentence encoder for ordering). Ensure you have internet access if caches are not present.

## Notebooks

Open notebooks in VS Code/Antigravity and select the `venv` Python kernel:

- **00_start_here.ipynb** – environment checks, project map, what each file does
- **01_punctuation_demo.ipynb** – run and understand punctuation restoration step-by-step
- **02_null_diagnostic.ipynb** – analyze all 67 `NULL` markers in context
- **03_sentence_ordering_demo.ipynb** – understand and run candidate ordering
- **04_full_pipeline.ipynb** – inspect ordering, 5-sentence grouping, and Hindi→English translation outputs

## Current status

- **Punctuation:** `Horizon-Labs/punctuation-restoration-small` (quantized ONNX, CPU) via `src/punctuation.py`; outputs in `data/punctuated/`.
- **Ordering (candidate):** embedding/heuristic ordering for all 7 samples via `src/order_sentences.py`; outputs in `data/ordered/*.ordered.txt` (+ mappings/stats).
- **Grouping:** every 5 consecutive ordered sentences grouped per sample.
- **Translation:** Hindi→English per group via `Helsinki-NLP/opus-mt-hi-en` (MarianMT encoder-decoder MT model, CPU) in `src/group_and_translate.py`; outputs in `data/final/` (`.groups_hi.txt`, `.groups_en.txt`, `.groups.tsv`, summaries).
- **Integrity:** literal `NULL` markers preserved across raw/model_input/punctuated/ordered/final-hi in current generated outputs (67 total). Verify with `python scripts/verify_integrity.py`.
- **Still needs review:** punctuation quality, `NULL` semantics/handling, and ordering correctness are not ground-truth validated. Translation quality depends on upstream steps.

## Notes for faculty/reproducibility

- **Punctuation model:** `Horizon-Labs/punctuation-restoration-small` (encoder token classifier), used as local quantized ONNX for CPU inference.
- **Sentence encoder:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for candidate ordering (embeddings + clustering/heuristics).
- **Translation model:** `Helsinki-NLP/opus-mt-hi-en` (MarianMT neural MT). This is not GPT/Gemini/Claude. The script can cache it locally under `models/translation/opus-mt-hi-en`.
- No off-the-shelf generative chat LLMs are used in the pipeline scripts. Keep total GPU memory under 1.5 GB if adapting components to GPU.
- Outputs should be evaluated manually/sampled; execution success ≠ correctness.

## Notes for faculty/reproducibility

- Model: `Horizon-Labs/punctuation-restoration-small` (encoder token classifier). We use a local quantized ONNX copy for CPU inference.
- The model card notes synthetic training data involving Qwen; we do **not** run Qwen or any generative LLM in this pipeline.
- Outputs should be evaluated manually/sampled; execution success ≠ correctness.
