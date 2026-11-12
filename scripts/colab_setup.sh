#!/usr/bin/env bash
set -euo pipefail

echo "[colab_setup] Installing Python dependencies for Colab..."

python -m pip install --upgrade pip

# Core deps (Colab often has torch/transformers preinstalled; keep compatible)
python -m pip install \
  transformers>=4.40.0 \
  sentencepiece \
  sacremoses \
  accelerate \
  onnxruntime \
  sentence-transformers \
  scikit-learn \
  pandas \
  numpy \
  tqdm \
  huggingface_hub

# IndicTrans2 toolkit (Linux/Colab usually has build tools available)
python -m pip install IndicTransToolkit || echo "[colab_setup] IndicTransToolkit install failed; you may need build tools or use fallback"

echo "[colab_setup] Done. Set HF_TOKEN for gated IndicTrans2 models if needed."
