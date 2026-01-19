import string
from collections import Counter
from typing import Dict, List, Tuple

import nltk
from nltk.tokenize import WordPunctTokenizer

BOUNDARY_TOKENS = {".", "!", "?"}


def tokenize_text(text: str) -> List[str]:
    """Tokenize text with lowercasing and simple punctuation filtering.

    Keeps sentence boundary tokens (., !, ?) for later sentence splitting.
    """
    try:
        raw_tokens = nltk.word_tokenize(text.lower())
    except LookupError:
        print(
            "NLTK tokenizer data not found; falling back to WordPunctTokenizer. "
            "To use punkt, run: python -m nltk.downloader punkt punkt_tab"
        )
        raw_tokens = WordPunctTokenizer().tokenize(text.lower())
    tokens: List[str] = []
    for tok in raw_tokens:
        if all(ch in string.punctuation for ch in tok):
            if tok in BOUNDARY_TOKENS:
                tokens.append(tok)
            continue
        tokens.append(tok)
    return tokens


def _filter_boundary_tokens(tokens: List[str]) -> List[str]:
    return [tok for tok in tokens if tok not in BOUNDARY_TOKENS]


def compute_stats(tokens: List[str]) -> Dict[str, object]:
    """Compute corpus statistics from tokens (excluding boundary markers)."""
    filtered = _filter_boundary_tokens(tokens)
    counts = Counter(filtered)
    top_20: List[Tuple[str, int]] = counts.most_common(20)
    stats: Dict[str, object] = {
        "num_tokens": len(filtered),
        "vocab_size": len(counts),
        "top_20": top_20,
    }
    return stats
