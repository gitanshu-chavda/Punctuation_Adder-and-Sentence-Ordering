# Manual Review Checklist

## Punctuation (especially NULL-adjacent and dialogue samples 6-7)
- [ ] Compare `data/raw/samples/*.txt` (intent/structure as given) vs `data/punctuated/*.txt`
- [ ] Inspect all 67 `NULL` contexts using `scripts/null_diagnostic.py` and `reports/null_punct_analysis.json`
- [ ] Check punctuation inserted immediately before/after `NULL` (e.g. `। NULL`, `NULL:`, `NULL?`) - do **not** remove or reinterpret `NULL`
- [ ] Review dialogue-heavy samples (`sample-6`, `sample-7`) for question marks, commas, sentence boundaries, and speaker turns
- [ ] Record corrections/suggestions in a copy or annotation file (keep originals unchanged)

## Ordering (candidate reconstruction)
- [ ] Review `data/ordered/sample-*.ordered.txt` against task goal (logical/original sequence), not fluency alone
- [ ] Use `data/ordered/sample-*.mapping.tsv` to trace positions
- [ ] Note clear ordering errors and patterns (topic jumps, discourse markers, pronoun/context dependencies)
- [ ] After rerunning mBERT pairwise+Borda ordering, recompute metrics: `python scripts/compute_metrics.py`

## Translation quality (no BLEU/chrF references yet)
- [ ] Review aligned files: `outputs/sample-*.sentences_hi.txt` vs `outputs/sample-*.sentences_en.txt`
- [ ] Spot-check grouped outputs: `outputs/sample-*.groups_hi.txt` vs `outputs/sample-*.groups_en.txt`
- [ ] Use `reports/translation_review_template.csv` for adequacy/fluency scoring and notes
- [ ] Check impact of punctuation/order errors on translation quality
- [ ] If IndicTrans2 is enabled, compare against MarianMT fallback outputs where available

## Integrity and compliance
- [ ] Run `python scripts/verify_integrity.py` (literal `NULL` preserved 67/67)
- [ ] Run `python setup_check.py`
- [ ] Confirm no GPT/Gemini/Claude used; document model/tool choices
- [ ] Capture full per-stage runtime/peak RAM and GPU VRAM (<1.5 GB if used) for final resource proof
