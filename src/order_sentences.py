"""Candidate sentence ordering for all samples using multilingual BERT pairwise scoring + Borda.

Reads:  data/punctuated/sample-*.txt
Writes: data/ordered/<stem>.ordered.txt
        data/ordered/<stem>.mapping.tsv
        data/ordered/<stem>.order_stats.json
        data/ordered/ordering_summary.json

Approach:
- Split punctuated text into sentence-like units.
- Encode each sentence with a multilingual BERT model (default: bert-base-multilingual-cased).
- Compute pairwise sentence-pair scores for all (i, j), i != j using BERT on concatenated pairs
  ([CLS] sent_i [SEP] sent_j [SEP]) and a coherence score derived from the pair representation.
- Aggregate pairwise preferences with a Borda-style global ranking (with heuristics) to produce
  a candidate order. This is a best-effort reconstruction, not guaranteed original order.

Run from project root:
    python src/order_sentences.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


INPUT_FOLDER = Path("data/punctuated")
OUTPUT_FOLDER = Path("data/ordered")
BERT_MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 256
BATCH_SIZE = 16
RANDOM_SEED = 42

CONTEXT_STARTERS = {
    "इसलिए", "इसके", "इसीलिए", "तब", "फिर", "लेकिन", "किंतु",
    "परंतु", "अतः", "बाद", "उन्होंने", "उसने", "उसके", "इनके",
}


def _safe_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_sentences(text: str) -> list[str]:
    """Extract sentences from punctuated or ordered text."""
    stripped = text.strip()
    if not stripped:
        return []
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 1 and sum(1 for ln in lines if re.search(r"[।?!]$", ln)) >= max(1, len(lines) - 2):
        return lines
    parts = re.split(r"(?<=[।?])\s+", stripped)
    return [p.strip() for p in parts if p.strip()]


def first_word(sentence: str) -> str:
    return sentence.split()[0].strip('.,:;!?।""\'"') if sentence.split() else ""


def start_penalty(sentence: str) -> float:
    return 0.12 if first_word(sentence) in CONTEXT_STARTERS else 0.0


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class MBertPairwiseOrderer:
    def __init__(self, model_name: str = BERT_MODEL_NAME, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_sentences(self, sentences: list[str]) -> np.ndarray:
        """Sentence embeddings from mBERT (mean-pooled token representations)."""
        embs = []
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            pooled = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.cpu().numpy())
        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)

    @torch.no_grad()
    def pair_coherence_scores(self, sentences: list[str]) -> np.ndarray:
        """Compute pairwise coherence scores for ordered pairs (i before j).

        Uses mBERT on concatenated sentence pairs and combines pair [CLS] signal with
        sentence-embedding compatibility to estimate whether sentence j is a plausible
        continuation after sentence i.
        """
        n = len(sentences)
        sent_embs = self.encode_sentences(sentences)
        pair_scores = np.full((n, n), -np.inf, dtype=np.float32)
        if n <= 1:
            return pair_scores

        # Precompute sentence-level compatibility
        sim = sent_embs @ sent_embs.T

        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        cls_scores = np.zeros(len(pairs), dtype=np.float32)

        for start in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[start : start + BATCH_SIZE]
            a_texts = [sentences[i] for i, _ in batch_pairs]
            b_texts = [sentences[j] for _, j in batch_pairs]
            inputs = self.tokenizer(
                a_texts,
                b_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            # Scalar coherence proxy from pair representation magnitude/consistency
            # (unsupervised): mean activation after normalization as a stable pair signal.
            score = cls.mean(dim=1)
            cls_scores[start : start + len(batch_pairs)] = score.detach().cpu().numpy().astype(np.float32)

        # Normalize CLS-derived scores for numerical stability
        if len(cls_scores) > 1:
            cmin, cmax = float(cls_scores.min()), float(cls_scores.max())
            if cmax > cmin:
                cls_norm = (cls_scores - cmin) / (cmax - cmin)
            else:
                cls_norm = np.zeros_like(cls_scores)
        else:
            cls_norm = np.zeros_like(cls_scores)

        for idx, (i, j) in enumerate(pairs):
            # Pairwise score combines mBERT pair encoding signal, semantic compatibility,
            # and a light discourse penalty for unlikely starts.
            pair_scores[i, j] = 0.55 * float(cls_norm[idx]) + 0.45 * float(sim[i, j]) - start_penalty(sentences[j])
        np.fill_diagonal(pair_scores, -np.inf)
        return pair_scores

    def order(self, sentences: list[str]) -> tuple[list[int], list[int], dict]:
        n = len(sentences)
        if n == 0:
            return [], [], {"method": "mbert_pairwise_borda", "sentences": 0}
        if n == 1:
            return [0], [1], {"method": "mbert_pairwise_borda", "sentences": 1}

        pair_scores = self.pair_coherence_scores(sentences)

        # Borda aggregation: for each item i, score = wins/outgoing preference mass minus incoming.
        # Use pairwise comparisons S(i,j) vs S(j,i).
        preference = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if pair_scores[i, j] >= pair_scores[j, i]:
                    preference[i, j] = pair_scores[i, j] - pair_scores[j, i] + 1e-6
                else:
                    preference[j, i] = pair_scores[j, i] - pair_scores[i, j] + 1e-6

        borda = preference.sum(axis=1) - preference.sum(axis=0)
        # Tie-breaker: average outgoing pair score, then earlier original index for stability.
        avg_out = np.array([
            np.mean([pair_scores[i, j] for j in range(n) if j != i]) for i in range(n)
        ], dtype=np.float32)
        order_ids = sorted(range(n), key=lambda i: (-float(borda[i]), -float(avg_out[i]), i))

        # Single "topic"/sequence label for compatibility with downstream artifacts
        topic_numbers = [1] * n
        meta = {
            "method": "mbert_pairwise_borda",
            "model": BERT_MODEL_NAME,
            "sentences": n,
            "notes": "Unsupervised pairwise scoring with multilingual BERT pair encodings and Borda aggregation; candidate order.",
        }
        return order_ids, topic_numbers, meta


def process_file(orderer: MBertPairwiseOrderer, input_file: Path, output_folder: Path) -> dict:
    text = input_file.read_text(encoding="utf-8")
    sentences = extract_sentences(text)
    if not sentences:
        raise ValueError(f"No sentences found in {input_file}")

    ordered_ids, topic_numbers, meta = orderer.order(sentences)
    stem = input_file.stem
    ordered_file = output_folder / f"{stem}.ordered.txt"
    mapping_file = output_folder / f"{stem}.mapping.tsv"
    stats_file = output_folder / f"{stem}.order_stats.json"

    ordered_file.write_text("\n".join(sentences[i] for i in ordered_ids) + "\n", encoding="utf-8")
    mapping_lines = ["new_position\toriginal_position\ttopic"]
    for new_pos, (orig_idx, topic) in enumerate(zip(ordered_ids, topic_numbers), start=1):
        mapping_lines.append(f"{new_pos}\t{orig_idx + 1}\t{topic}")
    mapping_file.write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")

    stats = {
        "input": str(input_file),
        "sentences": len(sentences),
        "topics": len(set(topic_numbers)) if topic_numbers else 0,
        "ordering": meta,
        "outputs": {"ordered": str(ordered_file), "mapping": str(mapping_file)},
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    _safe_stdout()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_FOLDER.glob("sample-*.txt"))
    if not files:
        print(f"No input files found in {INPUT_FOLDER}")
        return 1

    print(f"Loading multilingual BERT for pairwise ordering: {BERT_MODEL_NAME}", flush=True)
    orderer = MBertPairwiseOrderer(BERT_MODEL_NAME, device="cpu")

    results = []
    for f in files:
        print(f"\nOrdering {f.name} ...", flush=True)
        stats = process_file(orderer, f, OUTPUT_FOLDER)
        results.append(stats)
        print(f"  sentences={stats['sentences']} method={stats['ordering'].get('method')} -> {stats['outputs']['ordered']}")

    s1_ordered = OUTPUT_FOLDER / "sample-1.ordered.txt"
    s1_candidate = OUTPUT_FOLDER / "sample-1.candidate.txt"
    if s1_ordered.exists():
        s1_candidate.write_text(s1_ordered.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {"model": BERT_MODEL_NAME, "files": results}
    (OUTPUT_FOLDER / "ordering_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nDone. Processed {len(results)} files. Summary: {OUTPUT_FOLDER / 'ordering_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
