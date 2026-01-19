import math
import random
from typing import List, Tuple

from .ngram_lm import START_TOKEN, UNK_TOKEN
from .tokenize_stats import BOUNDARY_TOKENS


def _sentence_split(tokens: List[str]) -> List[List[str]]:
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


def train_test_split(
    tokens: List[str], test_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Split tokens into train/test by shuffling sentences for stability."""
    sentences = _sentence_split(tokens)
    rng = random.Random(seed)
    rng.shuffle(sentences)

    split_idx = int(len(sentences) * (1 - test_ratio))
    train_sents = sentences[:split_idx]
    test_sents = sentences[split_idx:]

    def _flatten(sents: List[List[str]]) -> List[str]:
        flat: List[str] = []
        for sent in sents:
            if flat:
                flat.append(".")
            flat.extend(sent)
        return flat

    return _flatten(train_sents), _flatten(test_sents)


def cross_entropy_unigram(model, test_tokens: List[str]) -> float:
    filtered = [tok for tok in test_tokens if tok not in BOUNDARY_TOKENS]
    if not filtered:
        return float("inf")

    log_sum = 0.0
    for tok in filtered:
        if tok not in model.vocab:
            tok = UNK_TOKEN
        lp = model.log_prob(tok)
        if lp == float("-inf"):
            return float("inf")
        log_sum += lp

    return -log_sum / len(filtered)


def cross_entropy_bigram(model, test_tokens: List[str], add_k: float) -> float:
    sentences = _sentence_split(test_tokens)
    if not sentences:
        return float("inf")

    log_sum = 0.0
    count = 0
    for sent in sentences:
        prev = START_TOKEN
        for tok in sent:
            if tok not in model.vocab:
                tok = UNK_TOKEN
            lp = model.log_prob(prev, tok, add_k=add_k)
            if lp == float("-inf"):
                return float("inf")
            log_sum += lp
            count += 1
            prev = tok

    if count == 0:
        return float("inf")
    return -log_sum / count


def perplexity_from_cross_entropy(h: float) -> float:
    if math.isinf(h):
        return float("inf")
    return 2 ** h
