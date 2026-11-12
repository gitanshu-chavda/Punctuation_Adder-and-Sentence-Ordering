"""Compute available quantitative metrics and update reports/evaluation.json.

- Punctuation: re-runs proxy evaluation script output summary.
- Ordering: compares data/ordered/*.ordered.txt against data/processed/*.sentences.txt
  by canonical text matching (best-effort alignment). Reports pairwise accuracy and
  Kendall tau on matched sentences. This is only meaningful when references represent
  the intended/original order.
- Integrity: checks NULL preservation.
- Translation: no automatic BLEU/chrF (no references available in repo).

Writes: reports/evaluation.json, reports/run_summary.json
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _safe_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def read_ref_sentences(i: int):
    p = ROOT / f"data/processed/sample-{i}.sentences.txt"
    if not p.exists():
        return []
    lines = [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sents = []
    for ln in lines:
        sents.append(ln.split("|", 1)[1].strip() if "|" in ln else ln.strip())
    return sents


def read_ordered(i: int):
    p = ROOT / f"data/ordered/sample-{i}.ordered.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def canon(s: str) -> str:
    s = re.sub(r"\bNULL\b", "", s)
    s = re.sub(r"[।,?：:\-\.!\"'“”‘’(){}\[\]…|/\\]", "", s)
    s = re.sub(r"\s+", "", s)
    return s.strip()


def kendall_tau(order_a, order_b):
    # order_a, order_b are lists of same item ids in compared order
    pos_b = {v: i for i, v in enumerate(order_b)}
    b = [pos_b[v] for v in order_a]
    n = len(b)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if b[i] < b[j]:
                concordant += 1
            elif b[i] > b[j]:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else None


def pairwise_accuracy(ref_order_ids, cand_order_ids):
    pos_c = {v: i for i, v in enumerate(cand_order_ids)}
    n = len(ref_order_ids)
    if n < 2:
        return None
    total = correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ref_order_ids[i], ref_order_ids[j]
            if a in pos_c and b in pos_c:
                total += 1
                if pos_c[a] < pos_c[b]:
                    correct += 1
    return (correct / total) if total else None


def ordering_metrics():
    per = []
    all_ref = []
    all_cand = []
    for i in range(1, 8):
        ref = read_ref_sentences(i)
        cand = read_ordered(i)
        # align by unique canonical string match
        ref_map = {}
        for idx, s in enumerate(ref):
            c = canon(s)
            if c:
                ref_map.setdefault(c, []).append(idx)
        cand_ids_in_ref = []
        used_ref = set()
        matched = 0
        for s in cand:
            c = canon(s)
            ids = ref_map.get(c, [])
            pick = None
            for rid in ids:
                if rid not in used_ref:
                    pick = rid
                    break
            if pick is not None:
                used_ref.add(pick)
                cand_ids_in_ref.append(pick)
                matched += 1
        ref_ids_matched = sorted(cand_ids_in_ref)
        # evaluate order over matched set
        tau = kendall_tau(cand_ids_in_ref, ref_ids_matched) if len(cand_ids_in_ref) >= 2 else None
        pacc = pairwise_accuracy(ref_ids_matched, cand_ids_in_ref) if len(cand_ids_in_ref) >= 2 else None
        per.append({
            "sample": i,
            "ref_sentences": len(ref),
            "candidate_sentences": len(cand),
            "matched_sentences": matched,
            "coverage": (matched / len(ref)) if ref else None,
            "kendall_tau_vs_ref": tau,
            "pairwise_accuracy_vs_ref": pacc,
        })
        all_ref.extend([f"{i}:{x}" for x in ref_ids_matched])  # not directly comparable across samples
    # macro averages over samples with metrics
    taus = [x["kendall_tau_vs_ref"] for x in per if x["kendall_tau_vs_ref"] is not None]
    paccs = [x["pairwise_accuracy_vs_ref"] for x in per if x["pairwise_accuracy_vs_ref"] is not None]
    covs = [x["coverage"] for x in per if x["coverage"] is not None]
    return {
        "reference": "data/processed/sample-*.sentences.txt (best-effort alignment by canonical text)",
        "caveat": "Metrics are computed only on matched sentences. If references are not ground-truth original order, interpret as reference-alignment rather than true ordering accuracy.",
        "macro_avg_kendall_tau": (sum(taus) / len(taus)) if taus else None,
        "macro_avg_pairwise_accuracy": (sum(paccs) / len(paccs)) if paccs else None,
        "macro_avg_matched_coverage": (sum(covs) / len(covs)) if covs else None,
        "per_sample": per,
    }


def punctuation_proxy():
    proc = subprocess.run([sys.executable, "scripts/evaluate_punctuation.py"], cwd=ROOT, capture_output=True, text=True)
    return {
        "script": "scripts/evaluate_punctuation.py",
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
        "returncode": proc.returncode,
    }


def integrity():
    proc = subprocess.run([sys.executable, "scripts/verify_integrity.py"], cwd=ROOT, capture_output=True, text=True)
    return {
        "script": "scripts/verify_integrity.py",
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
        "returncode": proc.returncode,
    }


def main():
    _safe_stdout()
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "punctuation": {
            "model": "Horizon-Labs/punctuation-restoration-small (quantized ONNX, CPU)",
            "proxy_evaluation": punctuation_proxy(),
            "note": "Proxy/approximate evaluation; not a gold-standard punctuation benchmark. Manual review required (especially NULL-adjacent and dialogue samples).",
        },
        "ordering": ordering_metrics(),
        "translation": {
            "current_backend_generated_outputs": "Helsinki-NLP/opus-mt-hi-en (MarianMT fallback) for existing outputs in this environment",
            "implemented_preferred_backend": "IndicTrans2 (ai4bharat/indictrans2-indic-en-dist-200M, configurable) with fallback",
            "bleu": None,
            "chrf": None,
            "note": "No reference translations available in-repo; automatic BLEU/chrF not computed.",
        },
        "integrity": integrity(),
    }
    (reports / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # run summary from available artifacts
    run_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "prepare_input -> punctuation -> order_sentences -> group_and_translate (+ aligned outputs)",
        "ordering_code": "src/order_sentences.py (mBERT pairwise scoring + Borda aggregation implemented)",
        "ordering_artifacts_current": "data/ordered/* currently present from previous candidate run; regenerate to apply latest method",
        "translation_artifacts_current": "outputs/* and data/final/* currently generated via MarianMT fallback in this environment",
        "resource_proof": {
            "cpu_runtime_profiled": False,
            "peak_memory_profiled": False,
            "gpu_vram_profiled": False,
            "note": "Add profiling run to capture per-stage timing and peak RAM/VRAM (<1.5GB if GPU used).",
        },
        "samples": [],
    }
    for p in sorted((ROOT / "data/final").glob("sample-*.pipeline.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            run_summary["samples"].append({
                "sample": p.name.split(".")[0],
                "sentences": j.get("sentences"),
                "groups": j.get("groups"),
                "translation_backend": j.get("translation_backend"),
                "translation_model": j.get("translation_model"),
            })
        except Exception:
            pass
    (reports / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Wrote reports/evaluation.json and reports/run_summary.json")
    om = evaluation["ordering"]
    print("Ordering macro avg tau:", om.get("macro_avg_kendall_tau"))
    print("Ordering macro avg pairwise acc:", om.get("macro_avg_pairwise_accuracy"))
    print("Ordering macro avg matched coverage:", om.get("macro_avg_matched_coverage"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
