import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .utils import ensure_dir, save_json, save_markdown


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = (preds == labels).mean()

    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 4: Transformer Fine-tuning")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_samples", type=int, default=800)
    parser.add_argument("--eval_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    a4_dir = base_dir.parent
    outputs_dir = a4_dir / "outputs"
    ensure_dir(outputs_dir)

    dataset = load_dataset("glue", "sst2")

    train_dataset = dataset["train"].shuffle(seed=args.seed)
    eval_dataset = dataset["validation"].shuffle(seed=args.seed)

    if args.train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.train_samples, len(train_dataset))))
    if args.eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.eval_samples, len(eval_dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_batch(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=args.max_length)

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    eval_dataset = eval_dataset.map(tokenize_batch, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # Transformers v5 uses eval_strategy instead of evaluation_strategy.
    training_args = TrainingArguments(
        output_dir=str(outputs_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=args.seed,
        report_to=[],
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    payload = {
        "settings": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        "metrics": metrics,
    }

    save_json(outputs_dir / "metrics.json", payload)
    save_json(outputs_dir / "training_args.json", payload["settings"])

    md_lines = ["| Metric | Value |", "| --- | --- |"]
    for key in ["eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]:
        if key in metrics:
            md_lines.append(f"| {key} | {metrics[key]:.4f} |")
    save_markdown(outputs_dir / "metrics.md", "\n".join(md_lines) + "\n")

    print("Assignment 4 complete.")
    print(f"Metrics: {outputs_dir / 'metrics.json'}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
