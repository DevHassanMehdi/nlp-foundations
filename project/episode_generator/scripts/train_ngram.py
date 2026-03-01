from pathlib import Path

from .ngram_model import BigramLM, save_model

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "corpus_words.txt"
MODEL_PATH = BASE_DIR / "models" / "ngram_model.json"


def main() -> None:
    if not DATA_PATH.exists():
        print("Corpus not found. Run data_prep first.")
        return

    tokens = DATA_PATH.read_text(encoding="utf-8").split()
    model = BigramLM(add_k=0.1)
    model.fit(tokens)
    save_model(MODEL_PATH, model)
    print(f"Saved n-gram model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
