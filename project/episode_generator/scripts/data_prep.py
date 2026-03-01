import re
from pathlib import Path
from typing import List

from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Cleaner prose corpora improve generation readability vs drama scripts.
DEFAULT_FILES = [
    "austen-emma.txt",
    "austen-sense.txt",
    "austen-persuasion.txt",
]


def load_texts(file_ids: List[str]) -> str:
    chunks = []
    for fid in file_ids:
        chunks.append(gutenberg.raw(fid))
    return "\n".join(chunks)


def clean_for_words(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.,;:!?\-']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_chars(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.,;:!?\-']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        raw = load_texts(DEFAULT_FILES)
    except LookupError:
        print("NLTK data not found. Run: python -m nltk.downloader punkt punkt_tab gutenberg")
        return

    cleaned = clean_for_words(raw)
    tokens = word_tokenize(cleaned)
    word_text = " ".join(tokens)
    (DATA_DIR / "corpus_words.txt").write_text(word_text, encoding="utf-8")

    char_text = clean_for_chars(raw)
    (DATA_DIR / "corpus_chars.txt").write_text(char_text, encoding="utf-8")

    print("Data prep complete.")
    print(f"Words: {DATA_DIR / 'corpus_words.txt'}")
    print(f"Chars: {DATA_DIR / 'corpus_chars.txt'}")


if __name__ == "__main__":
    main()
