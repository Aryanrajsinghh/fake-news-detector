import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT fake news classifier")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--validation-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models/bert-fake-news")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def normalize_label(v) -> int:
    # Your dataset uses: real = 1 (REAL), 0 (FAKE)
    return int(v)


def prepare_dataset(train_file, validation_file) -> DatasetDict:
    files = {"train": train_file, "validation": validation_file}
    dataset = load_dataset("csv", data_files=files)

    def remap(example):
        return {
            # Using correct columns from your CSV
            "text": str(example["title"]) + " " + str(example["source_domain"]),
            "label": normalize_label(example["real"]),
        }

    dataset = dataset.map(remap, remove_columns=dataset["train"].column_names)
    return dataset


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Loading dataset...")
    dataset = prepare_dataset(args.train_file, args.validation_file)

    print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    print("🧠 Tokenizing dataset...")
    encoded = dataset.map(tokenize, batched=True)

    # ⭐ CRITICAL FIX: dynamic padding (prevents tensor length crash)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("📦 Loading BERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
    )

    # GPU Detection (SAFE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🔥 Using device: {device}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # Uses GPU FP16 automatically
        logging_steps=50,
        save_steps=500,
        report_to="none",
    )

    print("🏋️ Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=data_collator,  # ⭐ FIXES YOUR LAST ERROR
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("📊 Evaluating model...")
    metrics = trainer.evaluate()

    print("💾 Saving model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ TRAINING COMPLETE")
    print(metrics)


if __name__ == "__main__":
    main()