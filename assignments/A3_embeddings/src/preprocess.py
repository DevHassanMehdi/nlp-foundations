import string
from typing import List

import nltk
from nltk.tokenize import WordPunctTokenizer

BOUNDARY_TOKENS = {".", "!", "?"}


def tokenize_text(text: str) -> List[str]:
    """Tokenize text with lowercasing and punctuation filtering."""
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


def sentence_split(tokens: List[str]) -> List[List[str]]:
    """Split tokens into sentences using boundary tokens."""
    sentences: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        if tok in BOUNDARY_TOKENS:
            if current:
                sentences.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        sentences.append(current)
    return sentences
