import argparse
from pathlib import Path

from .evaluate import (
    cross_entropy_bigram,
    cross_entropy_unigram,
    perplexity_from_cross_entropy,
    train_test_split,
)
from .load_data import load_gutenberg_text
from .ngram_lm import BigramLM, UnigramLM
from .tokenize_stats import BOUNDARY_TOKENS, compute_stats, tokenize_text
from .utils import ensure_dir, save_json, save_markdown, set_seed


def _write_results(outputs_dir: Path, results: list) -> None:
    results_json = {"results": results}
    save_json(outputs_dir / "results.json", results_json)

    lines = ["| Model | add_k | cross_entropy | perplexity |", "| --- | --- | --- | --- |"]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['add_k']} | {row['cross_entropy']:.4f} | {row['perplexity']:.4f} |"
        )
    save_markdown(outputs_dir / "results.md", "\n".join(lines) + "\n")

    csv_lines = ["model,add_k,cross_entropy,perplexity"]
    for row in results:
        csv_lines.append(
            f"{row['model']},{row['add_k']},{row['cross_entropy']},{row['perplexity']}"
        )
    save_markdown(outputs_dir / "results.csv", "\n".join(csv_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 1: Foundations of NLP")
    parser.add_argument("--file_id", default="austen-emma.txt")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--add_k", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    a1_dir = base_dir.parent
    outputs_dir = a1_dir / "outputs"
    ensure_dir(outputs_dir)

    text = load_gutenberg_text(args.file_id)
    if text is None:
        return

    tokens = tokenize_text(text)
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]

    train_tokens, test_tokens = train_test_split(
        tokens, test_ratio=args.test_ratio, seed=args.seed
    )

    unigram = UnigramLM(min_count=args.min_count)
    train_tokens_unigram = [tok for tok in train_tokens if tok not in BOUNDARY_TOKENS]
    unigram.fit(train_tokens_unigram)

    bigram = BigramLM(min_count=args.min_count)
    bigram.fit(train_tokens)

    stats_full = compute_stats(tokens)
    stats_train = compute_stats(train_tokens)
    stats_payload = {"full": stats_full, "train": stats_train}
    save_json(outputs_dir / "stats.json", stats_payload)

    stats_summary = (
        "# Corpus Statistics\n\n"
        f"- Full tokens: {stats_full['num_tokens']}\n"
        f"- Full vocab size: {stats_full['vocab_size']}\n"
        f"- Train tokens: {stats_train['num_tokens']}\n"
        f"- Train vocab size: {stats_train['vocab_size']}\n"
    )
    save_markdown(outputs_dir / "stats_summary.md", stats_summary)

    results = []
    h_uni = cross_entropy_unigram(unigram, test_tokens)
    results.append(
        {
            "model": "unigram",
            "add_k": 0.0,
            "cross_entropy": h_uni,
            "perplexity": perplexity_from_cross_entropy(h_uni),
        }
    )

    h_bi_unsmoothed = cross_entropy_bigram(bigram, test_tokens, add_k=0.0)
    results.append(
        {
            "model": "bigram",
            "add_k": 0.0,
            "cross_entropy": h_bi_unsmoothed,
            "perplexity": perplexity_from_cross_entropy(h_bi_unsmoothed),
        }
    )

    h_bi_smoothed = cross_entropy_bigram(bigram, test_tokens, add_k=args.add_k)
    results.append(
        {
            "model": "bigram",
            "add_k": args.add_k,
            "cross_entropy": h_bi_smoothed,
            "perplexity": perplexity_from_cross_entropy(h_bi_smoothed),
        }
    )

    _write_results(outputs_dir, results)

    print("Assignment 1 complete.")
    print(f"Stats: {outputs_dir / 'stats.json'}")
    print(f"Results: {outputs_dir / 'results.json'}")


if __name__ == "__main__":
    main()
