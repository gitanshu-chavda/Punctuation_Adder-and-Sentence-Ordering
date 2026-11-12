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
  ordered/              # Sentence-ordering outputs (candidate/mapping)
  processed/            # Intermediate/derived artifacts (e.g. sentence splits, eval refs)
  external/             # Optional external corpora (not required for current pipeline)

models/
  punctuation/
    onnx/model_quantized.onnx   # Quantized ONNX model (used for inference)
    tokenizer/                  # Local tokenizer files
    finetuned-quick/            # Experimental fine-tuned checkpoint (optional)

src/                    # Core reusable Python modules
  prepare_input.py      # Copy raw -> model_input, preserving NULL
  punctuation.py        # Punctuation restoration (ONNX, CPU)
  sentence-ordering.py  # Candidate sentence ordering (embeddings + heuristics)

scripts/                # Utility/evaluation scripts
  evaluate_punctuation.py
  null_diagnostic.py
  ...

notebooks/              # Beginner-friendly walkthroughs (run after understanding src/)
  00_start_here.ipynb
  01_punctuation_demo.ipynb
  02_null_diagnostic.ipynb
  03_sentence_ordering_demo.ipynb

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

# 3) Run punctuation restoration for all samples
python src/punctuation.py

# 4) Run NULL diagnostic (inspect how NULL is treated)
python scripts/null_diagnostic.py

# 5) Run sentence ordering (currently configured for sample-1)
python src/sentence-ordering.py
```

> Note: Scripts may download/cache Hugging Face models on first run (e.g. sentence encoder for ordering). Ensure you have internet access if caches are not present.

## Notebooks

Open notebooks in VS Code/Antigravity and select the `venv` Python kernel:

- **00_start_here.ipynb** – environment checks, project map, what each file does
- **01_punctuation_demo.ipynb** – run and understand punctuation restoration step-by-step
- **02_null_diagnostic.ipynb** – analyze all 67 `NULL` markers in context
- **03_sentence_ordering_demo.ipynb** – understand and run candidate ordering

## Current status

- Punctuation restoration implemented with `Horizon-Labs/punctuation-restoration-small` (quantized ONNX, CPU).
- All 7 samples processed to `data/punctuated/` (literal `NULL` preserved).
- Punctuation quality and `NULL` semantics still need careful manual review.
- Sentence ordering produces a **candidate** order for `sample-1` (not ground truth).
- Grouping (5 sentences) and Hindi→English translation are **not implemented yet** and must remain compliant with constraints.

## Notes for faculty/reproducibility

- Model: `Horizon-Labs/punctuation-restoration-small` (encoder token classifier). We use a local quantized ONNX copy for CPU inference.
- The model card notes synthetic training data involving Qwen; we do **not** run Qwen or any generative LLM in this pipeline.
- Outputs should be evaluated manually/sampled; execution success ≠ correctness.
