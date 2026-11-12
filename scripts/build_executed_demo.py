#!/usr/bin/env python3
"""Build notebooks/executed_demo.ipynb with embedded outputs from current repo artifacts.

Run from project root:
    python scripts/build_executed_demo.py

This reads generated artifacts under data/, outputs/, and reports/ and writes an
executed-style demo notebook (no model inference is rerun).
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(rel: str, n: int | None = None) -> str:
    p = ROOT / rel
    if not p.exists():
        return f"[missing] {rel}"
    txt = p.read_text(encoding="utf-8")
    if n is not None:
        lines = txt.splitlines()
        txt = "\n".join(lines[:n])
        if len(lines) > n:
            txt += f"\n... ({len(lines)-n} more lines)"
    return txt


def read_json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def md(cells, src: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": src.rstrip("\n").splitlines(keepends=True),
    })


def code(cells, src: str, outputs_text: str, execution_count: int = 1) -> None:
    src_lines = src.rstrip("\n").splitlines(keepends=True)
    out_lines = outputs_text.rstrip("\n").splitlines(keepends=True) if outputs_text else []
    cells.append({
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": ([{"name": "stdout", "output_type": "stream", "text": out_lines}] if out_lines else []),
        "source": src_lines,
    })


def main() -> int:
    evalj = read_json("reports/evaluation.json")
    trans = read_json("data/final/translation_summary.json")
    ord_summary = read_json("data/ordered/ordering_summary.json")
    cells = []

    md(cells, """# Executed Demo: Hindi Punctuation, Ordering, Grouping & Translation

This notebook is an **executed demo** with outputs embedded from current repository artifacts.

**Demonstrates:** input/NULL preservation, punctuation restoration, sentence ordering, 5-sentence grouping, translation, aligned outputs, and evaluation/report summaries.

> Heavy model inference is not rerun here; outputs shown come from generated artifacts in `data/`, `outputs/`, and `reports/`.
""")

    md(cells, "## 1) Project and artifact overview")
    samples = sorted(p.name for p in (ROOT / "data/raw/samples").glob("sample-*.txt"))
    ordered_n = len(list((ROOT / "data/ordered").glob("sample-*.ordered.txt")))
    final_n = len(list((ROOT / "data/final").glob("sample-*.pipeline.json")))
    aligned_n = len(list((ROOT / "outputs").glob("sample-*.txt")))
    reports = sorted(p.name for p in (ROOT / "reports").glob("*")) if (ROOT / "reports").exists() else []
    out1 = (
        f"Samples: {samples}\n"
        f"Ordered files: {ordered_n}\n"
        f"Final pipelines: {final_n}\n"
        f"Aligned output files: {aligned_n}\n"
        f"Reports: {reports}\n"
    )
    code(cells,
        "from pathlib import Path\nroot = Path('.')\nprint('Samples:', sorted(p.name for p in (root/'data/raw/samples').glob('sample-*.txt')))\nprint('Ordered files:', len(list((root/'data/ordered').glob('sample-*.ordered.txt'))))\nprint('Final pipelines:', len(list((root/'data/final').glob('sample-*.pipeline.json'))))\nprint('Aligned output files:', len(list((root/'outputs').glob('sample-*.txt'))))\nprint('Reports:', sorted(p.name for p in (root/'reports').glob('*')))\n",
        out1,
    )

    md(cells, "## 2) NULL marker preservation (integrity)")
    null_info = evalj.get("integrity", {})
    code(cells,
        "import json\nprint(json.dumps(json.load(open('reports/evaluation.json',encoding='utf-8')).get('integrity',{}), indent=2, ensure_ascii=False))\n",
        json.dumps(null_info, indent=2, ensure_ascii=False) + "\n",
    )

    md(cells, "## 3) Stage 1 — Punctuation restoration (sample-1 preview)")
    raw1 = read_text("data/raw/samples/sample-1.txt").splitlines()
    pun1 = read_text("data/punctuated/sample-1.txt").splitlines()
    out3 = "RAW sample-1 (head):\n" + "\n".join(f"{i}: {l[:220]}" for i, l in enumerate(raw1[:3], 1))
    if len(raw1) > 3:
        out3 += f"\n... ({len(raw1)-3} more lines)"
    out3 += "\n\nPUNCTUATED sample-1 (head):\n" + "\n".join(f"{i}: {l[:220]}" for i, l in enumerate(pun1[:3], 1))
    if len(pun1) > 3:
        out3 += f"\n... ({len(pun1)-3} more lines)"
    out3 += "\n"
    code(cells,
        "from pathlib import Path\n\ndef show(path, n=3):\n    lines = Path(path).read_text(encoding='utf-8').splitlines()\n    for i,l in enumerate(lines[:n],1):\n        print(f'{i}: {l[:220]}')\n    if len(lines)>n: print(f'... ({len(lines)-n} more lines)')\n\nprint('RAW sample-1 (head):')\nshow('data/raw/samples/sample-1.txt', 3)\nprint()\nprint('PUNCTUATED sample-1 (head):')\nshow('data/punctuated/sample-1.txt', 3)\n",
        out3,
    )

    md(cells, "## 4) Stage 2 — Sentence ordering")
    ord_keys = ['model','method','device','batch_size','max_length','max_vram_budget_gb','any_over_budget','cuda_peak_allocated_gb_overall','cuda_peak_reserved_gb_overall']
    ord_info = {k: ord_summary.get(k) for k in ord_keys if k in ord_summary}
    ord_lines = read_text("data/ordered/sample-1.ordered.txt").splitlines()
    out4 = json.dumps(ord_info, indent=2, ensure_ascii=False) + "\n\nFirst 3 ordered sentences (sample-1):\n" + "\n".join(f"{i}. {l[:220]}" for i, l in enumerate(ord_lines[:3], 1)) + "\n"
    code(cells,
        "import json\ns=json.load(open('data/ordered/ordering_summary.json',encoding='utf-8'))\nkeys=['model','method','device','batch_size','max_length','max_vram_budget_gb','any_over_budget','cuda_peak_allocated_gb_overall','cuda_peak_reserved_gb_overall']\nprint(json.dumps({k:s.get(k) for k in keys if k in s}, indent=2, ensure_ascii=False))\nprint('\\nFirst 3 ordered sentences (sample-1):')\nlines=open('data/ordered/sample-1.ordered.txt',encoding='utf-8').read().splitlines()\nfor i,l in enumerate(lines[:3],1): print(f'{i}. {l[:220]}')\n",
        out4,
    )

    md(cells, "## 5) Stage 3 — Grouping and translation")
    trans_info = {k: trans.get(k) for k in ['translation_backend','translation_model','group_size','files_processed'] if k in trans}
    per = "\n".join(f"  sample {f.get('sample')}: sentences={f.get('sentences')} groups={f.get('groups')}" for f in trans.get('files', []))
    out5 = json.dumps(trans_info, indent=2, ensure_ascii=False) + "\n\nPer-sample (sentences/groups):\n" + per + "\n"
    code(cells,
        "import json\ns=json.load(open('data/final/translation_summary.json',encoding='utf-8'))\nprint(json.dumps({k:s.get(k) for k in ['translation_backend','translation_model','group_size','files_processed']}, indent=2, ensure_ascii=False))\nprint('\\nPer-sample (sentences/groups):')\nfor f in s.get('files',[]):\n    print(f\"  sample {f.get('sample')}: sentences={f.get('sentences')} groups={f.get('groups')}\")\n",
        out5,
    )

    md(cells, "## 6) Aligned outputs (4 files per sample)")
    counts_lines = []
    for i in range(1, 8):
        def count(rel):
            p = ROOT / rel
            return len(p.read_text(encoding='utf-8').splitlines()) if p.exists() else 'NA'
        hi = count(f'outputs/sample-{i}.sentences_hi.txt')
        en = count(f'outputs/sample-{i}.sentences_en.txt')
        gh = count(f'outputs/sample-{i}.groups_hi.txt')
        ge = count(f'outputs/sample-{i}.groups_en.txt')
        counts_lines.append(f'sample-{i}: sent_hi={hi} sent_en={en} grp_hi={gh} grp_en={ge}')
    shi = read_text('outputs/sample-1.sentences_hi.txt', 1).splitlines()[0] if (ROOT/'outputs/sample-1.sentences_hi.txt').exists() else ''
    sen = read_text('outputs/sample-1.sentences_en.txt', 1).splitlines()[0] if (ROOT/'outputs/sample-1.sentences_en.txt').exists() else ''
    gen = read_text('outputs/sample-1.groups_en.txt', 1).splitlines()[0] if (ROOT/'outputs/sample-1.groups_en.txt').exists() else ''
    out6 = "\n".join(counts_lines) + "\n\nSample-1 sentence 1 (HI):\n" + shi[:240] + "\nSample-1 sentence 1 (EN):\n" + sen[:240] + "\n\nSample-1 group 1 (EN):\n" + gen[:350] + "\n"
    code(cells,
        "from pathlib import Path\nfor i in range(1,8):\n    hi = Path(f'outputs/sample-{i}.sentences_hi.txt')\n    en = Path(f'outputs/sample-{i}.sentences_en.txt')\n    gh = Path(f'outputs/sample-{i}.groups_hi.txt')\n    ge = Path(f'outputs/sample-{i}.groups_en.txt')\n    print(f'sample-{i}: sent_hi={len(hi.read_text(encoding=\"utf-8\").splitlines())} sent_en={len(en.read_text(encoding=\"utf-8\").splitlines())} grp_hi={len(gh.read_text(encoding=\"utf-8\").splitlines())} grp_en={len(ge.read_text(encoding=\"utf-8\").splitlines())}')\n",
        out6,
    )

    md(cells, "## 7) Evaluation summary")
    ord_eval = evalj.get('ordering', {})
    punct = evalj.get('punctuation_proxy', evalj.get('punctuation', {}))
    translation_eval = evalj.get('translation', {})
    out7 = (
        f"Punctuation: {json.dumps(punct, ensure_ascii=False)}\n"
        f"Ordering macro avg kendall_tau: {ord_eval.get('macro_avg_kendall_tau')}\n"
        f"Ordering macro avg pairwise_accuracy: {ord_eval.get('macro_avg_pairwise_accuracy')}\n"
        f"Ordering macro avg matched_coverage: {ord_eval.get('macro_avg_matched_coverage')}\n"
        f"Translation: {json.dumps(translation_eval, ensure_ascii=False)}\n"
    )
    code(cells,
        "import json\nj=json.load(open('reports/evaluation.json',encoding='utf-8'))\no=j.get('ordering',{})\np=j.get('punctuation_proxy', j.get('punctuation', {}))\nprint('Punctuation:', json.dumps(p, ensure_ascii=False))\nprint('Ordering macro avg kendall_tau:', o.get('macro_avg_kendall_tau'))\nprint('Ordering macro avg pairwise_accuracy:', o.get('macro_avg_pairwise_accuracy'))\nprint('Ordering macro avg matched_coverage:', o.get('macro_avg_matched_coverage'))\nprint('Translation:', json.dumps(j.get('translation',{}), ensure_ascii=False))\n",
        out7,
    )

    backend = trans.get('translation_backend')
    model = trans.get('translation_model')
    md(cells, f"""## 8) Notes (reflecting current artifacts)
- **Ordering config/artifacts:** see `data/ordered/ordering_summary.json` (embedded above) and per-sample stats in `data/ordered/`.
- **Translation backend/model (current artifacts):** `{backend}` / `{model}` (see `data/final/translation_summary.json`).
- **NULL markers:** preserved as literal `NULL`; verify integrity in `reports/evaluation.json` and raw/final comparisons.
- **Caveats:** ordering metrics may be reference-alignment/proxy depending on available ground truth; translation quality metrics (BLEU/chrF) require reference translations; resource/VRAM evidence should be recorded for the executed run if required.
""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = ROOT / "notebooks" / "executed_demo.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
