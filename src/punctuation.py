"""
punctuation.py  --  Hindi punctuation restoration (quantized ONNX, CPU only)

Model : Horizon-Labs/punctuation-restoration-small  (encoder token classifier,
        NOT a generative LLM).  It looks at every word and predicts which mark,
        if any, should come right AFTER it:  0 (none)  .  ,  ?  -  :

Pipeline for one text
---------------------
1. remove any existing punctuation          (the dataset has none, kept for safety)
2. split the text into words                 (whitespace separated)
3. tokenize with the model's tokenizer       (words -> sub-word tokens)
4. run the model over overlapping WINDOWS of words, with TWO window sizes, and
   AVERAGE the predictions of every word          [NEW: token-safe, ensemble]
5. pick one mark per word (argmax after optional per-mark scaling)  [NEW: scales]
6. fix well-known Hindi mistakes with small grammar rules           [NEW: rules]
7. rebuild the text ('.' becomes the Hindi danda '।')

What is new compared with the first version
-------------------------------------------
A. Windows are also limited by TOKENS, not only words.  Hindi words often split
   into several sub-word tokens, so 200 words can exceed the model's maximum
   length and silently get truncated (= missing punctuation at the end).
B. Two window sizes are averaged (ensemble) -> more stable predictions.
C. Per-mark scale factors (CLASS_SCALE) so '.' / ',' can be made more or less
   eager.  They can be tuned automatically with --tune (see below).
D. Small rule layer (NO_MARK_AFTER, NEVER_END_SENTENCE, NEVER_START_SENTENCE,
   CONTRAST_WORDS, long-sentence splitting).  Every change it makes is listed
   in the review report so you can check it.
E. --tune mode: give it a hand-corrected file and it finds the best CLASS_SCALE
   and tells you whether the rules help or hurt.

Run from anywhere:
    python src/punctuation.py                       # punctuate all samples
    python src/punctuation.py --tune gold1.txt ...  # tune on corrected files

Nothing in data/raw or data/model_input is ever modified.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


# ======================================================================
# 1. Paths and settings
# ======================================================================

# This file lives in <project>/src/, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models/punctuation/onnx/model_quantized.onnx"
TOKENIZER_FOLDER = PROJECT_ROOT / "models/punctuation/tokenizer"

INPUT_FOLDER = PROJECT_ROOT / "data/model_input"
OUTPUT_FOLDER = PROJECT_ROOT / "data/punctuated"
PROCESSED_FOLDER = PROJECT_ROOT / "data/processed"

# Class id -> mark.  Order comes from the model's config.json.
MODEL_ID2LABEL = {0: "0", 1: ".", 2: ",", 3: "?", 4: "-", 5: ":"}
LABEL2ID = {label: i for i, label in MODEL_ID2LABEL.items()}
NUM_LABELS = len(MODEL_ID2LABEL)

# --- Windowing --------------------------------------------------------
# Each entry is (window size in WORDS, step in WORDS).  Every config is run
# over the whole text and the probabilities of all configs are averaged.
# The model card says training windows were up to 230 words, so 200 is the
# largest size used here; 100 gives a second, more local opinion.
WINDOW_CONFIGS = [(200, 100), (100, 50)]

# Hard limit on TOKENS per window (the model adds CLS + SEP on top).  Windows
# are shrunk automatically when the words are too token-heavy.  480 is safe for
# a 512-token model; lower it if you know the model's max length is smaller.
MAX_WINDOW_TOKENS = 480

# --- Decision rule ----------------------------------------------------
# Probabilities are multiplied by these factors BEFORE taking the argmax.
#   > 1  -> that mark is chosen more often      < 1 -> less often
# 1.0 everywhere = plain argmax.  Use  --tune  to find good values.
CLASS_SCALE = {".": 1.0, ",": 1.0, "?": 1.0, "-": 1.0, ":": 1.0}

# Optional: demand a minimum probability before a mark is accepted.
# Example:  {",": 0.6, ":": 0.8, "-": 0.8}
MIN_CONFIDENCE: dict[str, float] = {}

# Every sentence must end somewhere, so the last word of a text always gets a
# full stop if the model left it with nothing / a comma / a colon / a hyphen.
FORCE_FINAL_STOP = True

# --- Rule layer (all rules can be switched off with USE_RULES = False) --
USE_RULES = True

# Titles / honorifics that are part of a name: "योगीराज श्रीकृष्ण",
# "धर्मराज युधिष्ठिर".  A COMMA (or colon / hyphen) right after them is
# almost always wrong, so it is removed.
NO_MARK_AFTER = frozenset({
    "भगवान", "श्री", "श्रीमान", "श्रीमती", "महाराज", "योगीराज", "धर्मराज",
    "पितामह", "माँ", "महर्षि", "आचार्य",
})

# Function words that cannot end a sentence or be followed by a comma:
# a "." or "," right after them is removed.
NEVER_END_SENTENCE = frozenset({
    "के", "की", "का", "कि", "ने", "और", "तथा", "व", "एवं",
})

# Function words that cannot START a sentence: if the model put a "." just
# before them, that "." is removed.
NEVER_START_SENTENCE = frozenset({
    "कि", "तथा", "व", "एवं", "के", "की", "का", "ने", "में", "से", "को",
})

# Contrast words: Hindi normally puts a comma before them ("है, किंतु ...").
# A comma is added when the model left the previous word without any mark.
CONTRAST_WORDS = frozenset({
    "किंतु", "किन्तु", "लेकिन", "परंतु", "परन्तु", "मगर", "बल्कि",
})

# Sentences longer than this are examined for a missed full stop.  The longest
# real sentences in the samples are about 40 words, so 50 is a safe limit.
MAX_SENTENCE_WORDS = 50
# A split is only made where the model gave "." at least this probability ...
SPLIT_MIN_PROB = 0.05
# ... and never closer than this many words to either end of the sentence.
MIN_PIECE_WORDS = 8

# --- Diagnostics ------------------------------------------------------
NULL_MARKER = "NULL"          # only COUNTED and reported, never changed
SHORT_SENTENCE_WORDS = 2      # flag sentences this short ...
LONG_SENTENCE_WORDS = 40      # ... or this long for manual review
REVIEW_TOP_N = 10             # how many items to list in each review section

# Characters ignored when checking "did we lose any words?"
_IGNORED_FOR_COMPARISON = re.compile(
    r"[\s.,;:!?\-。，、；：！？؟،¿¡…।॥۔።፣፧]"
)

# A sentence ends after a danda or question mark followed by whitespace.
_SENTENCE_BREAK = re.compile(r"(?<=[।?])\s+")


# ======================================================================
# 2. Small helpers
# ======================================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn raw model scores (logits) into probabilities that sum to 1."""
    shifted = logits - logits.max(axis=-1, keepdims=True)  # avoids overflow
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def text_is_preserved(original: str, restored: str) -> bool:
    """
    True if `restored` contains exactly the same letters/digits in the same
    order as `original`, ignoring whitespace and punctuation.  So no word
    (and no NULL) was lost, duplicated, or altered.
    """
    same_content = (
        _IGNORED_FOR_COMPARISON.sub("", original)
        == _IGNORED_FOR_COMPARISON.sub("", restored)
    )
    same_nulls = original.count(NULL_MARKER) == restored.count(NULL_MARKER)
    return same_content and same_nulls


def split_sentences(text: str) -> list[str]:
    """Split punctuated text into sentences (after '।' or '?')."""
    return [s for s in _SENTENCE_BREAK.split(text.strip()) if s]


@dataclass
class RestoreResult:
    """Everything produced for one text, so it can be inspected later."""
    text: str                 # the punctuated text
    words: list[str]          # the words, in order
    marks: list[str]          # final mark after each word ("0" = none)
    confidences: list[float]  # model probability of that final choice
    word_probs: np.ndarray    # (n_words, 6) full probability table
    num_tokens: int
    num_windows: int
    # (word index, old mark, new mark, reason) for every rule-based change
    rule_changes: list = field(default_factory=list)


# ======================================================================
# 3. Rule layer
# ======================================================================

def _sentence_spans(marks: list[str]) -> list[tuple[int, int]]:
    """(first word, last word) of every sentence, judged by '.' and '?'."""
    spans, start = [], 0
    for i, mark in enumerate(marks):
        if mark in (".", "?"):
            spans.append((start, i))
            start = i + 1
    if start < len(marks):
        spans.append((start, len(marks) - 1))
    return spans


def apply_rules(words: list[str], marks: list[str], word_probs: np.ndarray):
    """
    Correct typical mistakes of the model using simple Hindi grammar facts.
    Returns (new marks, list of (index, old, new, reason)).
    Words are never touched, only the marks after them.
    """
    marks = list(marks)
    changes: list[tuple[int, str, str, str]] = []
    n = len(words)

    def set_mark(i: int, new: str, reason: str) -> None:
        if marks[i] != new:
            changes.append((i, marks[i], new, reason))
            marks[i] = new

    # R1  "योगीराज, श्रीकृष्ण"  ->  "योगीराज श्रीकृष्ण"   (title + name)
    for i in range(n - 1):
        if words[i] in NO_MARK_AFTER and marks[i] in (",", ":", "-"):
            set_mark(i, "0", "no comma after a title")

    # R2  "... के, ..." / "... और। ..."  (function word cannot end a clause)
    for i in range(n - 1):
        if words[i] in NEVER_END_SENTENCE and marks[i] in (".", ","):
            set_mark(i, "0", "function word cannot end a clause")

    # R3  "... है। तथा ..."  (these words never start a sentence)
    for i in range(1, n):
        if words[i] in NEVER_START_SENTENCE and marks[i - 1] == ".":
            set_mark(i - 1, "0", f"sentence cannot start with '{words[i]}'")

    # R4  "... है किंतु ..."  ->  "... है, किंतु ..."
    for i in range(1, n):
        if (words[i] in CONTRAST_WORDS and marks[i - 1] == "0"
                and words[i - 1] not in NEVER_END_SENTENCE
                and words[i - 1] != NULL_MARKER):
            set_mark(i - 1, ",", f"comma before '{words[i]}'")

    # R5  very long sentence -> add a full stop at the most likely place
    stop_id = LABEL2ID["."]
    while True:
        split_done = False
        for start, end in _sentence_spans(marks):
            if end - start + 1 <= MAX_SENTENCE_WORDS:
                continue
            best = None                              # (probability, index)
            for i in range(start + MIN_PIECE_WORDS - 1, end - MIN_PIECE_WORDS + 1):
                if (words[i] in NEVER_END_SENTENCE or words[i] in NO_MARK_AFTER
                        or words[i] == NULL_MARKER
                        or words[i + 1] in NEVER_START_SENTENCE):
                    continue
                p = float(word_probs[i, stop_id])
                if p >= SPLIT_MIN_PROB and (best is None or p > best[0]):
                    best = (p, i)
            if best is not None:
                set_mark(best[1], ".", f"split {end - start + 1}-word sentence "
                                       f"(P('.')={best[0]:.2f})")
                split_done = True
                break                                # spans changed: restart
        if not split_done:
            break

    return marks, changes


# ======================================================================
# 4. The restorer
# ======================================================================

class PunctuationRestorer:

    def __init__(self):

        if not MODEL_FILE.exists():
            raise FileNotFoundError(MODEL_FILE)
        if not TOKENIZER_FOLDER.exists():
            raise FileNotFoundError(TOKENIZER_FOLDER)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_FOLDER), local_files_only=True, use_fast=True
        )

        # CPU only -> uses no GPU memory at all.
        self.session = ort.InferenceSession(
            str(MODEL_FILE), providers=["CPUExecutionProvider"]
        )

        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        if self.cls_id is None or self.sep_id is None:
            raise ValueError("Tokenizer has no CLS/SEP token ids.")

        print("Punctuation model loaded successfully!")
        print("Inference provider:", self.session.get_providers())

    # ------------------------------------------------------------------
    # 4a. Existing punctuation removal
    # ------------------------------------------------------------------
    @staticmethod
    def remove_existing_punctuation(text: str) -> str:

        punctuation_pattern = r"[.,;:!?。，、；：！？؟،¿¡…।॥۔።፣፧]"

        def should_preserve(match):
            mark = match.group()
            # Keep decimal points / numeric separators such as 3.14 or 1,000
            if mark in ".,:":
                position = match.start()
                if (
                    0 < position < len(text) - 1
                    and text[position - 1].isdigit()
                    and text[position + 1].isdigit()
                ):
                    return mark
            return ""

        return re.sub(punctuation_pattern, should_preserve, text)

    # ------------------------------------------------------------------
    # 4b. Label -> printed mark
    # ------------------------------------------------------------------
    @staticmethod
    def convert_label(label: str) -> str:
        if label == "0":
            return ""
        if label == ".":
            return "।"        # Hindi full stop (danda)
        if label == "-":
            return " -"       # spaced hyphen
        return label          # ',' '?' ':' stay as they are

    # ------------------------------------------------------------------
    # 4c. One model call
    # ------------------------------------------------------------------
    def _run_model(self, token_ids: list[int]) -> np.ndarray:
        """Return an (n_tokens, 6) table of probabilities for these tokens."""
        model_ids = [self.cls_id] + list(token_ids) + [self.sep_id]
        input_ids = np.array([model_ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        logits = self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]

        # [0] = first (only) sentence in the batch; 1:-1 drops CLS and SEP.
        logits = logits[0, 1:-1, :]
        if logits.shape[-1] != NUM_LABELS:
            raise ValueError(f"Expected {NUM_LABELS} classes, got {logits.shape[-1]}")
        return softmax(logits)

    # ------------------------------------------------------------------
    # 4d. Tokens <-> words
    # ------------------------------------------------------------------
    def _align_tokens(self, text: str, words: list[str]):
        """
        Tokenize `text` (the words joined by single spaces) and work out which
        word every token belongs to.

        Returns
        -------
        token_ids        list[int]   tokens that belong to some word
        token_word       np.ndarray  word index of each of those tokens
        word_last_token  np.ndarray  for each word, index (into token_ids) of
                                     its LAST token, or -1 if it has none.
                                     The model card: "the label of each word is
                                     the label of its last token".
        """
        encoded = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )

        # char_to_word[c] = which word the character at position c belongs to
        # (-1 for the spaces between words)
        char_to_word = np.full(len(text), -1, dtype=np.int64)
        position = 0
        for index, word in enumerate(words):
            char_to_word[position: position + len(word)] = index
            position += len(word) + 1          # +1 for the single space

        token_ids, token_word = [], []
        for token_id, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
            # Some tokenizers include a space in a token's span, so step back
            # to the last real character of the token.
            last_char = min(end, len(text)) - 1
            while last_char >= start and text[last_char] == " ":
                last_char -= 1
            if last_char < start:
                continue                        # empty / whitespace-only token
            token_ids.append(token_id)
            token_word.append(int(char_to_word[last_char]))

        token_word = np.array(token_word, dtype=np.int64)

        word_last_token = np.full(len(words), -1, dtype=np.int64)
        for token_index, word_index in enumerate(token_word):
            word_last_token[word_index] = token_index   # later tokens overwrite

        return token_ids, token_word, word_last_token

    # ------------------------------------------------------------------
    # 4e. One sliding-window pass with ONE window size
    # ------------------------------------------------------------------
    def _sliding_window(self, token_ids, word_last_token, cum_tokens,
                        n_words: int, window_words: int, stride_words: int):
        """
        Returns (probabilities (n_words, 6), covered (n_words,) bool, windows).
        `cum_tokens[w]` = number of tokens that belong to words 0..w-1, so the
        tokens of words [a, b) are token_ids[cum_tokens[a]:cum_tokens[b]].
        """
        prob_sum = np.zeros((n_words, NUM_LABELS))
        weight_sum = np.zeros(n_words)
        num_windows = 0

        start = 0
        while True:
            # (1) window limited by words ...
            end = min(start + window_words, n_words)
            # (2) ... and by tokens: largest `end` with <= MAX tokens inside
            token_end = int(np.searchsorted(
                cum_tokens, cum_tokens[start] + MAX_WINDOW_TOKENS, side="right")) - 1
            end = max(start + 1, min(end, token_end))

            first = int(cum_tokens[start])
            last = min(int(cum_tokens[end]), first + MAX_WINDOW_TOKENS)  # safety cap

            if last > first:
                probs = self._run_model(token_ids[first:last])
                num_windows += 1
                length = end - start

                for word_index in range(start, end):
                    token_index = int(word_last_token[word_index])
                    if token_index < 0 or token_index >= last:
                        continue                      # no token / cut by the cap
                    j = word_index - start             # position inside the window
                    # Triangle weight: 1 at both edges, biggest in the middle.
                    # Edge words have little context on one side, so they get
                    # little say; the window where the word is central wins.
                    weight = min(j + 1, length - j)
                    prob_sum[word_index] += weight * probs[token_index - first]
                    weight_sum[word_index] += weight

            if end == n_words:
                break
            # normal step, but never more than half of a (shrunken) window
            start += max(1, min(stride_words, (end - start) // 2))

        covered = weight_sum > 0
        averaged = np.zeros((n_words, NUM_LABELS))
        averaged[covered] = prob_sum[covered] / weight_sum[covered, None]
        return averaged, covered, num_windows

    # ------------------------------------------------------------------
    # 4f. Probabilities for every word (ensemble of window sizes)
    # ------------------------------------------------------------------
    def _word_probabilities(self, words: list[str]):
        text = " ".join(words)
        token_ids, token_word, word_last_token = self._align_tokens(text, words)
        n_words = len(words)

        tokens_per_word = np.bincount(token_word, minlength=n_words)
        cum_tokens = np.concatenate([[0], np.cumsum(tokens_per_word)])

        ensemble_sum = np.zeros((n_words, NUM_LABELS))
        ensemble_count = np.zeros(n_words)
        total_windows = 0

        for window_words, stride_words in WINDOW_CONFIGS:
            assert 0 < stride_words <= window_words
            probs, covered, num_windows = self._sliding_window(
                token_ids, word_last_token, cum_tokens,
                n_words, window_words, stride_words)
            ensemble_sum[covered] += probs[covered]
            ensemble_count[covered] += 1
            total_windows += num_windows

        # Default for any word that got no prediction: "no mark" with p = 1.
        word_probs = np.zeros((n_words, NUM_LABELS))
        word_probs[:, LABEL2ID["0"]] = 1.0
        got_some = ensemble_count > 0
        word_probs[got_some] = ensemble_sum[got_some] / ensemble_count[got_some, None]

        return word_probs, len(token_ids), total_windows

    # ------------------------------------------------------------------
    # 4g. Probabilities -> one mark per word
    # ------------------------------------------------------------------
    @staticmethod
    def _choose_marks(word_probs: np.ndarray, scale: dict | None = None):
        scale = CLASS_SCALE if scale is None else scale
        factors = np.array([scale.get(MODEL_ID2LABEL[i], 1.0) for i in range(NUM_LABELS)])
        chosen = (word_probs * factors).argmax(axis=1)

        marks, confidences = [], []
        for probs, label_id in zip(word_probs, chosen):
            label_id = int(label_id)
            label = MODEL_ID2LABEL[label_id]
            # Optional confidence filter (empty by default)
            if label != "0" and probs[label_id] < MIN_CONFIDENCE.get(label, 0.0):
                label_id, label = LABEL2ID["0"], "0"
            marks.append(label)
            confidences.append(float(probs[label_id]))
        return marks, confidences

    # ------------------------------------------------------------------
    # 4h. Public API
    # ------------------------------------------------------------------
    def restore_detailed(self, text: str) -> RestoreResult:
        text = self.remove_existing_punctuation(text)
        words = text.split()      # also normalizes spaces / newlines

        if not words:
            return RestoreResult("", [], [], [], np.zeros((0, NUM_LABELS)), 0, 0)

        word_probs, num_tokens, num_windows = self._word_probabilities(words)
        marks, _ = self._choose_marks(word_probs)

        changes = []
        if USE_RULES:
            marks, changes = apply_rules(words, marks, word_probs)

        if FORCE_FINAL_STOP and marks[-1] not in (".", "?"):
            marks[-1] = "."

        confidences = [float(word_probs[i, LABEL2ID[m]]) for i, m in enumerate(marks)]
        pieces = [w + self.convert_label(m) for w, m in zip(words, marks)]

        return RestoreResult(
            text=" ".join(pieces),
            words=words,
            marks=marks,
            confidences=confidences,
            word_probs=word_probs,
            num_tokens=num_tokens,
            num_windows=num_windows,
            rule_changes=changes,
        )

    def restore(self, text: str) -> str:
        """Text in, punctuated text out."""
        return self.restore_detailed(text).text


# ======================================================================
# 5. Review report (what to check by hand)
# ======================================================================

def show_context(words, marks, index, before=4, after=3) -> str:
    """Words around `index`, with the word at `index` wrapped in [[ ]]."""
    low = max(0, index - before)
    high = min(len(words), index + after + 1)
    parts = []
    for k in range(low, high):
        piece = words[k] + PunctuationRestorer.convert_label(marks[k])
        parts.append(f"[[{piece}]]" if k == index else piece)
    return " ".join(parts)


def null_statistics(result: RestoreResult):
    """Count marks placed right before / after each literal NULL word."""
    positions = [i for i, w in enumerate(result.words) if w == NULL_MARKER]
    before = Counter(result.marks[i - 1] for i in positions if i > 0 and result.marks[i - 1] != "0")
    after = Counter(result.marks[i] for i in positions if result.marks[i] != "0")
    return positions, before, after


def build_review_section(name: str, result: RestoreResult, sentences: list[str]) -> list[str]:
    words, marks, confs, probs = result.words, result.marks, result.confidences, result.word_probs
    sizes = [len(s.split()) for s in sentences]
    lines = [f"=== {name} ===",
             f"sentences: {len(sentences)}   words per sentence: "
             f"min {min(sizes)}, median {statistics.median(sizes):.0f}, max {max(sizes)}", ""]

    # (a) sentences that look wrong by length
    lines.append(f"-- Sentences with <= {SHORT_SENTENCE_WORDS} or >= {LONG_SENTENCE_WORDS} words --")
    flagged = [(i, n, s) for i, (s, n) in enumerate(zip(sentences, sizes), 1)
               if n <= SHORT_SENTENCE_WORDS or n >= LONG_SENTENCE_WORDS]
    lines += [f"  #{i:03d} ({n} word{'' if n == 1 else 's'}): {s}" for i, n, s in flagged] or ["  (none)"]

    # (b) marks the model was least sure about
    lines += ["", f"-- {REVIEW_TOP_N} least confident marks (check first) --"]
    marked = sorted((confs[i], i) for i in range(len(words)) if marks[i] != "0")
    for conf, i in marked[:REVIEW_TOP_N]:
        lines.append(f"  p={conf:.2f}  {marks[i]!r}  {show_context(words, marks, i)}")

    # (c) places where a full stop was nearly chosen but wasn't
    stop_id = LABEL2ID["."]
    lines += ["", f"-- {REVIEW_TOP_N} strongest 'maybe a missed full stop' candidates --"]
    near = sorted(((probs[i, stop_id], i) for i in range(len(words))
                   if marks[i] == "0" and probs[i, stop_id] >= 0.10), reverse=True)
    for p, i in near[:REVIEW_TOP_N]:
        lines.append(f"  P('.')={p:.2f}  {show_context(words, marks, i)}")
    if not near:
        lines.append("  (none)")

    # (d) every change made by the rule layer
    lines += ["", f"-- Rule-based corrections: {len(result.rule_changes)} --"]
    for i, old, new, reason in result.rule_changes:
        lines.append(f"  {old!r} -> {new!r}  ({reason})  "
                     f"{show_context(words, marks, i)}")
    if not result.rule_changes:
        lines.append("  (none)")

    # (e) NULL neighbourhood (counts only - no meaning is assumed)
    positions, before, after = null_statistics(result)
    lines += ["", f"-- NULL markers: {len(positions)} found --",
              f"  mark right BEFORE NULL: {dict(before) or 'none'}",
              f"  mark right AFTER  NULL: {dict(after) or 'none'}"]
    for n, i in enumerate(positions, 1):
        lines.append(f"  {n:2d}. {show_context(words, marks, i, before=3, after=3)}")

    lines.append("")
    return lines


# ======================================================================
# 6. Tuning against hand-corrected text   (python src/punctuation.py --tune ...)
# ======================================================================

_GOLD_MARKS = {"।": ".", "॥": ".", ".": ".", "!": ".", "?": "?", ":": ":",
               ",": ",", ";": ",", "-": "-"}
_GOLD_PRIORITY = [".", "?", ":", ",", "-"]
_ALL_PUNCT = "".join(_GOLD_MARKS) + "。，、；：！？؟،¿¡…۔።፣፧\"'“”‘’()[]"


def parse_gold(text: str):
    """
    Turn a correctly punctuated text into (words, marks).  A word's mark is the
    punctuation directly after it ('0' if none).  A lone '-' is attached to the
    previous word.
    """
    words, marks = [], []
    for token in text.split():
        if token == "-" and words and marks[-1] == "0":
            marks[-1] = "-"
            continue
        trailing = set()
        core = token
        while core and core[-1] in _ALL_PUNCT:
            if core[-1] in _GOLD_MARKS:
                trailing.add(_GOLD_MARKS[core[-1]])
            core = core[:-1]
        core = core.lstrip(_ALL_PUNCT)
        if not core:
            continue
        words.append(core)
        marks.append(next((m for m in _GOLD_PRIORITY if m in trailing), "0"))
    return words, marks


def _predict_marks(words, word_probs, scale, use_rules):
    marks, _ = PunctuationRestorer._choose_marks(word_probs, scale)
    if use_rules:
        marks, _ = apply_rules(words, marks, word_probs)
    if marks and marks[-1] not in (".", "?"):
        marks[-1] = "."
    return marks


def _f1_table(gold: list[str], pred: list[str]):
    table = {}
    for label in (".", ",", "?", ":", "-"):
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        table[label] = (precision, recall, f1, tp + fn)
    return table


def _objective(table) -> float:
    """Mean F1 over the marks that actually occur in the gold text."""
    scores = [f1 for (_, _, f1, support) in table.values() if support > 0]
    return sum(scores) / len(scores) if scores else 0.0


def _evaluate(data, scale, use_rules):
    gold_all, pred_all = [], []
    for words, gold, word_probs in data:
        gold_all += gold
        pred_all += _predict_marks(words, word_probs, scale, use_rules)
    return _f1_table(gold_all, pred_all)


def _print_table(title, table):
    print(f"  {title}   (mean F1 = {_objective(table):.3f})")
    for label, (p, r, f1, support) in table.items():
        if support:
            print(f"     {label}  precision {p:.2f}  recall {r:.2f}  F1 {f1:.2f}  (gold count {support})")


def tune(gold_files: list[str]) -> None:
    restorer = PunctuationRestorer()

    data = []
    for path in gold_files:
        words, gold = parse_gold(Path(path).read_text(encoding="utf-8"))
        if not words:
            raise ValueError(f"{path}: no words found")
        word_probs, _, _ = restorer._word_probabilities(words)
        data.append((words, gold, word_probs))
        print(f"loaded {path}: {len(words)} words, "
              f"{sum(m == '.' for m in gold)} full stops, {sum(m == ',' for m in gold)} commas")

    plain = {k: 1.0 for k in CLASS_SCALE}
    print("\nBaseline (all scales 1.0):")
    _print_table("model only", _evaluate(data, plain, use_rules=False))
    _print_table("model + rules", _evaluate(data, plain, use_rules=True))

    # coordinate ascent over the two marks that matter most
    grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    best_scale = dict(plain)
    for _ in range(2):
        for label in (".", ","):
            def key(s):
                trial = dict(best_scale, **{label: s})
                return (-_objective(_evaluate(data, trial, USE_RULES)), abs(math.log(s)))
            best_scale[label] = min(grid, key=key)

    print("\nBest scales found:", best_scale)
    _print_table("model + rules, tuned", _evaluate(data, best_scale, USE_RULES))
    print("\nTo use them, set in this file:")
    print(f"  CLASS_SCALE = {best_scale}")
    print("If 'model only' beats 'model + rules', set USE_RULES = False.")
    print("(With only a few sentences of gold text, trust big changes only if "
          "they are consistent across several files.)")


# ======================================================================
# 7. Process all files
# ======================================================================

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    input_files = sorted(INPUT_FOLDER.glob("sample-*.txt"))
    if not input_files:
        raise FileNotFoundError(f"No sample files found in {INPUT_FOLDER}")

    load_start = time.perf_counter()
    restorer = PunctuationRestorer()
    print(f"Model load time: {time.perf_counter() - load_start:.1f} s\n")

    report_lines = []

    for input_file in input_files:
        original_text = input_file.read_text(encoding="utf-8")

        start = time.perf_counter()
        result = restorer.restore_detailed(original_text)
        seconds = time.perf_counter() - start

        # Safety net: refuse to save if any word or NULL was lost or changed.
        if not text_is_preserved(original_text, result.text):
            raise RuntimeError(f"{input_file.name}: words/NULL changed - output NOT saved!")

        (OUTPUT_FOLDER / input_file.name).write_text(result.text, encoding="utf-8")

        sentences = split_sentences(result.text)
        numbered = [f"{i:03d} | {s}" for i, s in enumerate(sentences, 1)]
        sentence_file = PROCESSED_FOLDER / f"{input_file.stem}.sentences.txt"
        sentence_file.write_text("\n".join(numbered) + "\n", encoding="utf-8")

        counts = Counter(result.marks)
        positions, before, after = null_statistics(result)

        print(f"{input_file.name}")
        print(f"  words {len(result.words)} | tokens {result.num_tokens} | "
              f"windows {result.num_windows} | {seconds:.1f} s")
        print(f"  danda {counts['.']} | comma {counts[',']} | question {counts['?']} | "
              f"colon {counts[':']} | hyphen {counts['-']} | "
              f"rule changes {len(result.rule_changes)}")
        print(f"  sentences {len(sentences)} | NULL in/out "
              f"{original_text.count(NULL_MARKER)}/{result.text.count(NULL_MARKER)} | "
              f"mark before NULL {sum(before.values())}, after NULL {sum(after.values())}")
        print("  words preserved: OK\n")

        report_lines += build_review_section(input_file.name, result, sentences)

    report_file = PROCESSED_FOLDER / "review_report.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Punctuated files : {OUTPUT_FOLDER}")
    print(f"Sentence files   : {PROCESSED_FOLDER}  (*.sentences.txt)")
    print(f"Review report    : {report_file}")
    print("\nAll punctuation files generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hindi punctuation restoration")
    parser.add_argument("--tune", nargs="+", metavar="GOLD_FILE",
                        help="tune CLASS_SCALE on hand-corrected punctuated file(s)")
    cli_args, _ = parser.parse_known_args()     # tolerant of notebook arguments
    if cli_args.tune:
        tune(cli_args.tune)
    else:
        main()