import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train bert-base-uncased for fake-news classification")
    parser.add_argument("--train-file", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--validation-file", type=str, required=True, help="Path to validation CSV file")
    parser.add_argument("--text-column", type=str, default="text", help="Text column in CSV")
    parser.add_argument("--label-column", type=str, default="label", help="Label column in CSV")
    parser.add_argument("--output-dir", type=str, default="models/bert-fake-news", help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_label(value) -> int:
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"fake", "1", "true"}:
            return 1
        if token in {"real", "0", "false"}:
            return 0
    normalized = int(value)
    if normalized not in {0, 1}:
        raise ValueError(f"Label must be 0/1 or real/fake, got: {value}")
    return normalized


def prepare_dataset(args) -> DatasetDict:
    files = {"train": args.train_file, "validation": args.validation_file}
    dataset = load_dataset("csv", data_files=files)

    required = {args.text_column, args.label_column}
    train_cols = set(dataset["train"].column_names)
    if not required.issubset(train_cols):
        raise ValueError(
            f"Missing required columns. Needed={required}, found={train_cols} in train CSV."
        )

    def remap(example):
        return {
            "text": str(example[args.text_column]),
            "label": normalize_label(example[args.label_column]),
        }

    return dataset.map(remap, remove_columns=dataset["train"].column_names)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    clean_metrics = {
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
        "precision": float(metrics.get("eval_precision", 0.0)),
        "recall": float(metrics.get("eval_recall", 0.0)),
        "f1": float(metrics.get("eval_f1", 0.0)),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(clean_metrics, handle, indent=2)

    print("Training complete. Validation metrics:")
    for key, value in clean_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
