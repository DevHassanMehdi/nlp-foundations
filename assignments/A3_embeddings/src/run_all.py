import argparse
from pathlib import Path

from .evaluate import compute_similarities, nearest_neighbors
from .load_data import load_gutenberg_text
from .preprocess import sentence_split, tokenize_text
from .train_embeddings import train_word2vec
from .utils import ensure_dir, save_json, save_markdown, set_seed
from .visualize import pca_words


def _write_similarities(outputs_dir: Path, results: list) -> None:
    save_json(outputs_dir / "similarities.json", {"pairs": results})
    lines = ["| word1 | word2 | similarity | status |", "| --- | --- | --- | --- |"]
    for row in results:
        sim = "" if row["similarity"] is None else f"{row['similarity']:.4f}"
        lines.append(f"| {row['word1']} | {row['word2']} | {sim} | {row['status']} |")
    save_markdown(outputs_dir / "similarities.md", "\n".join(lines) + "\n")


def _write_neighbors(outputs_dir: Path, neighbors: dict) -> None:
    save_json(outputs_dir / "neighbors.json", neighbors)
    lines = ["# Nearest Neighbors\n"]
    for word, items in neighbors.items():
        lines.append(f"## {word}")
        if not items:
            lines.append("- OOV or no neighbors\n")
            continue
        for item in items:
            lines.append(f"- {item['word']}: {item['score']:.4f}")
        lines.append("")
    save_markdown(outputs_dir / "neighbors.md", "\n".join(lines).strip() + "\n")


def _write_pca(outputs_dir: Path, coords: list) -> None:
    header = "word,x,y"
    lines = [header]
    for word, x, y in coords:
        lines.append(f"{word},{x},{y}")
    save_markdown(outputs_dir / "pca_coords.csv", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 3: Word Embedding Analysis")
    parser.add_argument("--file_id", default="austen-emma.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vector_size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--pca_words", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    a3_dir = base_dir.parent
    outputs_dir = a3_dir / "outputs"
    ensure_dir(outputs_dir)

    text = load_gutenberg_text(args.file_id)
    if text is None:
        return

    tokens = tokenize_text(text)
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]

    sentences = sentence_split(tokens)
    if not sentences:
        print("No sentences found. Check tokenizer or input text.")
        return

    model = train_word2vec(
        sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        seed=args.seed,
    )

    pairs = [
        ("good", "bad"),
        ("man", "woman"),
        ("mr", "mrs"),
        ("love", "hate"),
        ("emma", "harriet"),
        ("friend", "enemy"),
    ]
    similarities = compute_similarities(model, pairs)
    _write_similarities(outputs_dir, similarities)

    neighbor_words = ["emma", "mr", "mrs", "good", "bad", "love", "friend"]
    neighbors = nearest_neighbors(model, neighbor_words, topn=args.topn)
    _write_neighbors(outputs_dir, neighbors)

    vocab_words = list(model.wv.index_to_key)[: args.pca_words]
    coords = pca_words(model, vocab_words)
    _write_pca(outputs_dir, coords)

    print("Assignment 3 complete.")
    print(f"Similarities: {outputs_dir / 'similarities.json'}")
    print(f"Neighbors: {outputs_dir / 'neighbors.json'}")
    print(f"PCA coords: {outputs_dir / 'pca_coords.csv'}")


if __name__ == "__main__":
    main()
