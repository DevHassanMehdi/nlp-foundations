import string
from collections import Counter
from typing import Dict, List, Tuple

import nltk
from nltk.tokenize import WordPunctTokenizer


def tokenize_text(text: str) -> List[str]:
    """Lowercase and tokenize text, removing punctuation-only tokens."""
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
            continue
        tokens.append(tok)
    return tokens


def build_vocab(
    token_lists: List[List[str]], min_count: int = 2, max_vocab: int = 10000
) -> Dict[str, int]:
    """Build a vocabulary from tokenized documents."""
    counts = Counter(tok for doc in token_lists for tok in doc)
    vocab_tokens = [tok for tok, cnt in counts.items() if cnt >= min_count]
    vocab_tokens.sort(key=lambda t: counts[t], reverse=True)
    vocab_tokens = vocab_tokens[: max_vocab - 1]

    vocab = {"<UNK>": 0}
    for tok in vocab_tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def vectorize_counts(token_lists: List[List[str]], vocab: Dict[str, int]) -> Tuple[List[int], List[List[int]]]:
    """Vectorize documents as sparse count vectors (indices and counts)."""
    indices_list: List[List[int]] = []
    counts_list: List[List[int]] = []
    for doc in token_lists:
        counts = Counter(vocab.get(tok, 0) for tok in doc)
        indices = list(counts.keys())
        values = [counts[idx] for idx in indices]
        indices_list.append(indices)
        counts_list.append(values)
    return indices_list, counts_list # type: ignore



def to_dense(
    indices_list: List[List[int]], counts_list: List[List[int]], vocab_size: int
) -> List[List[float]]:
    """Convert sparse vectors to dense list of floats."""
    dense: List[List[float]] = []
    for indices, counts in zip(indices_list, counts_list):
        row = [0.0] * vocab_size
        for idx, val in zip(indices, counts):
            row[idx] = float(val)
        dense.append(row)
    return dense
