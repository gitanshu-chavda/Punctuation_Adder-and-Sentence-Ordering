"""
Fine-tune Horizon-Labs/punctuation-restoration-small on local Hindi data.

Uses data/processed/custom_punctuation_training.jsonl (tokens + labels).
Label mapping to model:
  NONE -> 0
  FULL_STOP -> .
  COMMA -> ,
  QUESTION -> ?
  EXCLAMATION -> . (fallback; model has no exclamation class)

Outputs a HuggingFace checkpoint and optionally exports ONNX.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_ID = "Horizon-Labs/punctuation-restoration-small"
LABEL2ID = {"0": 0, ".": 1, ",": 2, "?": 3, "-": 4, ":": 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MAP = {
    "NONE": "0",
    "COMMA": ",",
    "FULL_STOP": ".",
    "QUESTION": "?",
    "EXCLAMATION": ".",  # fallback
}


class PunctDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        tokens = row["tokens"]
        labels = [MAP.get(l, "0") for l in row["labels"]]
        # tokenize as pre-tokenized words; align labels to first subtoken
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        word_ids = enc.word_ids()
        label_ids = []
        prev_w = None
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev_w:
                label_ids.append(LABEL2ID[labels[w]])
            else:
                # subword token: ignore for loss (predict at word boundary via first token)
                label_ids.append(-100)
            prev_w = w
        enc["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in enc.items()}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    acc = (preds[mask] == labels[mask]).mean() if mask.any() else 0.0
    # per-class accuracy for non-0 labels
    non_o = mask & (labels != 0)
    punct_acc = (preds[non_o] == labels[non_o]).mean() if non_o.any() else 0.0
    return {"token_acc": float(acc), "punct_token_acc": float(punct_acc)}


def export_onnx(model, tokenizer, out_path, max_length=256):
    model.eval()
    dummy = tokenizer(
        "यह एक उदाहरण है",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {n: {0: "batch", 1: "seq"} for n in input_names}
    dynamic_axes["logits"] = {0: "batch", 1: "seq"}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(out_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
    print(f"Exported ONNX to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/custom_punctuation_training.jsonl")
    ap.add_argument("--output", default="models/punctuation/finetuned")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None, help="limit number of training rows for quick runs")
    ap.add_argument("--export-onnx", default=None, help="path to export ONNX (e.g. models/punctuation/onnx/model_finetuned.onnx)")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    rows = load_jsonl(args.data)
    if not rows:
        raise SystemExit("No training rows found")
    random.shuffle(rows)
    if args.max_samples:
        rows = rows[: args.max_samples]
    split = max(1, int(len(rows) * (1 - args.val_ratio)))
    train_rows, val_rows = rows[:split], rows[split:]
    print(f"Loaded {len(rows)} rows: train={len(train_rows)} val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=6, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=False
    )

    train_ds = PunctDataset(train_rows, tokenizer, args.max_length)
    val_ds = PunctDataset(val_rows, tokenizer, args.max_length) if val_rows else None

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="punct_token_acc",
        greater_is_better=True,
        fp16=use_cuda,
        report_to="none",
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics if val_ds else None,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fine-tuned model to {args.output}")

    if args.export_onnx:
        # load best from output
        best = AutoModelForTokenClassification.from_pretrained(args.output)
        export_onnx(best, tokenizer, args.export_onnx, args.max_length)


if __name__ == "__main__":
    main()
