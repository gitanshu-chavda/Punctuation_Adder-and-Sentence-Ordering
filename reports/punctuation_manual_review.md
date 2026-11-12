# Punctuation Manual Review Notes (Automated Assist)

**Scope:** Cautious, descriptive review. `NULL` semantics are unknown - do not delete/reinterpret `NULL` markers.

## Summary
- Total NULL markers checked: 67
- Punctuation immediately before NULL (in punctuated output): 36
- Colon immediately after NULL: 22
- Danda immediately after NULL: 0

## Key observations (review required)
- Dialogue-like samples (6,7) show frequent punctuation inserted adjacent to `NULL` in outputs; verify whether this reflects intended boundaries or model artifacts.
- Several samples show `NULL` followed by `:` or adjacent to `?` in punctuated text; keep literal `NULL` tokens but manually judge surrounding punctuation.
- Proxy evaluation is approximate (`scripts/evaluate_punctuation.py`, avg char sim 0.963, proxy mark acc 0.884, sample-5 anomaly).

## Per-sample checklist

### sample-1.txt 
- NULL count: 5
- Flagged adjacent punctuation contexts: 0
- Action: manually inspect `data/punctuated/sample-1.txt` around each NULL using `scripts/null_diagnostic.py` / `reports/null_punct_analysis.json`.

### sample-2.txt 
- NULL count: 5
- Flagged adjacent punctuation contexts: 0
- Action: manually inspect `data/punctuated/sample-2.txt` around each NULL using `scripts/null_diagnostic.py` / `reports/null_punct_analysis.json`.

### sample-3.txt 
- NULL count: 6
- Flagged adjacent punctuation contexts: 1 (examples: #6 left=',' right=',')
- Action: manually inspect `data/punctuated/sample-3.txt` around each NULL using `scripts/null_diagnostic.py` / `reports/null_punct_analysis.json`.

### sample-4.txt 
- NULL count: 4
- Flagged adjacent punctuation contexts: 1 (examples: #4 left=',' right='र')
- Action: manually inspect `data/punctuated/sample-4.txt` around each NULL using `scripts/null_diagnostic.py` / `reports/null_punct_analysis.json`.

### sample-5.txt 
- NULL count: 9
- Flagged adjacent punctuation contexts: 0
- Action: manually inspect `data/punctuated/sample-5.txt` around each NULL using `scripts/null_diagnostic.py` / `reports/null_punct_analysis.json`.

### sample-6.txt (dialogue-like)
- NULL count: 11
- Flagged adjacent punctuation contexts: 8 (examples: #1 left='।' right=':'; #2 left='?' right=':'; #3 left='।' right=':'; #4 left='।' right=':'; #5 left='?' right=':'; #6 left='।' right=':'; #7 left='।' right=':'; #8 left='?' right=':')
- Action: manually inspect `data/punctuated/sample-6.txt` around each NULL using `scripts/null_diagnostic.py` / `reports/null_punct_analysis.json`.

### sample-7.txt (dialogue-like)
- NULL count: 27
- Flagged adjacent punctuation contexts: 15 (examples: #4 left='।' right=':'; #5 left='।' right=':'; #6 left='।' right=':'; #7 left='?' right=':'; #8 left='।' right=':'; #9 left='।' right=':'; #10 left='।' right=':'; #11 left='?' right=':'; ... )
- Action: manually inspect `data/punctuated/sample-7.txt` around each NULL using `scripts/null_diagnostic.py` / `reports/null_punct_analysis.json`.

