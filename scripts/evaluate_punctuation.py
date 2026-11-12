"""
Evaluate punctuation restoration outputs against gold segmented sentence files.
"""
import re
from pathlib import Path
from difflib import SequenceMatcher


def normalize(text: str) -> str:
    # remove NULL markers used in dataset
    text = re.sub(r"\bNULL\b", " ", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_marks_by_word_end(text: str):
    """Return list of punctuation marks (or '') attached after each whitespace-delimited token."""
    # split but keep punctuation attached to tokens conceptually by iterating
    # simpler: walk through and record mark when boundary after a word char
    marks = []
    # tokenize into sequences of non-space
    tokens = text.split()
    for tok in tokens:
        m = re.search(r"([।,?：:\-\.!]+)$", tok)
        if m:
            marks.append(m.group(1))
        else:
            marks.append("")
    return tokens, marks


def load_gold(sent_path: Path) -> str:
    lines = []
    for ln in sent_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "|" in ln:
            _, rest = ln.split("|", 1)
            lines.append(rest.strip())
        else:
            lines.append(ln)
    return normalize(" ".join(lines))


def main():
    pred_dir = Path("data/punctuated")
    gold_dir = Path("data/processed")
    results = []
    for i in range(1, 8):
        pred_path = pred_dir / f"sample-{i}.txt"
        gold_path = gold_dir / f"sample-{i}.sentences.txt"
        if not pred_path.exists() or not gold_path.exists():
            continue
        pred = normalize(pred_path.read_text(encoding="utf-8"))
        gold = load_gold(gold_path)
        pred_tokens, pred_marks = extract_marks_by_word_end(pred)
        gold_tokens, gold_marks = extract_marks_by_word_end(gold)
        # align by token strings (stripped of end punctuation)
        def strip_end(t):
            return re.sub(r"[।,?：:\-\.!]+$", "", t)
        pred_words = [strip_end(t) for t in pred_tokens]
        gold_words = [strip_end(t) for t in gold_tokens]
        n = min(len(pred_words), len(gold_words))
        # count matches only where words equal at same index as rough proxy
        same_word = 0
        mark_match = 0
        mark_total = 0
        for j in range(n):
            if pred_words[j] == gold_words[j]:
                same_word += 1
                if gold_marks[j] or pred_marks[j]:
                    mark_total += 1
                    if pred_marks[j] == gold_marks[j]:
                        mark_match += 1
        # char similarity
        sim = SequenceMatcher(None, pred, gold).ratio()
        results.append({
            "sample": i,
            "pred_chars": len(pred),
            "gold_chars": len(gold),
            "similarity": sim,
            "aligned_words": same_word,
            "mark_total": mark_total,
            "mark_match": mark_match,
            "mark_acc": (mark_match / mark_total) if mark_total else None,
            "pred_counts": {m: pred.count(m) for m in ["।", ",", "?", "-"]},
            "gold_counts": {m: gold.count(m) for m in ["।", ",", "?", "-"]},
        })
    import sys
    # avoid Windows console encoding issues
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    for r in results:
        ma = r["mark_acc"]
        ma_s = f"{ma:.3f}" if ma is not None else "n/a"
        print(f"sample-{r['sample']}: sim={r['similarity']:.3f} mark_acc~={ma_s} ({r['mark_match']}/{r['mark_total']}) pred={r['pred_counts']} gold={r['gold_counts']}")
    if results:
        avg_sim = sum(r['similarity'] for r in results)/len(results)
        mm = sum(r['mark_match'] for r in results)
        mt = sum(r['mark_total'] for r in results)
        avg_ma = (mm/mt) if mt else 0
        print(f"AVERAGE: sim={avg_sim:.3f} mark_acc~={avg_ma:.3f} ({mm}/{mt})")


if __name__ == "__main__":
    main()
