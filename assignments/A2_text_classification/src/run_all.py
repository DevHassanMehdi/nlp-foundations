import argparse
from pathlib import Path

import numpy as np

from .evaluate import classification_metrics, train_test_split
from .load_data import load_movie_reviews
from .models import LogisticRegression, MultinomialNB
from .preprocess import build_vocab, tokenize_text, to_dense, vectorize_counts
from .utils import ensure_dir, save_json, save_markdown, set_seed


def _write_metrics(outputs_dir: Path, payload: dict) -> None:
    save_json(outputs_dir / "metrics.json", payload)

    lines = ["| Model | Accuracy | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- |"]
    for row in payload["results"]:
        lines.append(
            "| {model} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(**row)
        )
    save_markdown(outputs_dir / "metrics.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 2: Text Classification")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    a2_dir = base_dir.parent
    outputs_dir = a2_dir / "outputs"
    ensure_dir(outputs_dir)

    data = load_movie_reviews()
    if not data:
        return

    if args.max_docs is not None:
        data = data[: args.max_docs]

    train, test = train_test_split(data, test_ratio=args.test_ratio, seed=args.seed)

    train_texts = [t for t, _ in train]
    train_labels = np.array([y for _, y in train], dtype=int)
    test_texts = [t for t, _ in test]
    test_labels = np.array([y for _, y in test], dtype=int)

    train_tokens = [tokenize_text(text) for text in train_texts]
    test_tokens = [tokenize_text(text) for text in test_texts]

    vocab = build_vocab(train_tokens, min_count=args.min_count, max_vocab=args.max_vocab)
    save_json(outputs_dir / "vocab.json", {"size": len(vocab), "vocab": vocab})

    train_indices, train_counts = vectorize_counts(train_tokens, vocab)
    test_indices, test_counts = vectorize_counts(test_tokens, vocab)

    X_train = np.array(to_dense(train_indices, train_counts, len(vocab))) # type: ignore
    X_test = np.array(to_dense(test_indices, test_counts, len(vocab))) # type: ignore

    nb = MultinomialNB(alpha=args.alpha)
    nb.fit(X_train, train_labels)
    nb_preds = nb.predict(X_test)
    nb_metrics = classification_metrics(test_labels, nb_preds)

    lr = LogisticRegression(lr=args.lr, epochs=args.epochs, l2=args.l2)
    lr.fit(X_train, train_labels)
    lr_preds = lr.predict(X_test)
    lr_metrics = classification_metrics(test_labels, lr_preds)

    results = [
        {"model": "naive_bayes", **{k: nb_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]}},
        {
            "model": "logistic_regression",
            **{k: lr_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]},
        },
    ]

    payload = {
        "settings": {
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "min_count": args.min_count,
            "max_vocab": args.max_vocab,
            "alpha": args.alpha,
            "lr": args.lr,
            "epochs": args.epochs,
            "l2": args.l2,
            "max_docs": args.max_docs,
        },
        "results": results,
        "confusion": {
            "naive_bayes": nb_metrics,
            "logistic_regression": lr_metrics,
        },
    }

    _write_metrics(outputs_dir, payload)

    print("Assignment 2 complete.")
    print(f"Metrics: {outputs_dir / 'metrics.json'}")
    print(f"Vocab: {outputs_dir / 'vocab.json'}")


if __name__ == "__main__":
    main()
