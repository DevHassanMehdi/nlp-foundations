from collections import Counter
from typing import Dict, Iterable, List, Tuple

from .tokenize_stats import BOUNDARY_TOKENS
from .utils import safe_log

UNK_TOKEN = "<UNK>"
START_TOKEN = "<s>"


def _sentence_split(tokens: List[str]) -> List[List[str]]:
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


def _replace_rare(tokens: Iterable[str], min_count: int) -> Tuple[List[str], Dict[str, int]]:
    """Replace rare tokens with <UNK> and return mapped tokens and counts."""
    counts = Counter(tokens)
    mapped: List[str] = []
    for tok in tokens:
        if counts[tok] < min_count:
            mapped.append(UNK_TOKEN)
        else:
            mapped.append(tok)
    return mapped, counts


class UnigramLM:
    def __init__(self, min_count: int = 2) -> None:
        self.min_count = min_count
        self.counts: Counter = Counter()
        self.total: int = 0
        self.vocab = set()

    def fit(self, tokens: List[str]) -> None:
        mapped_tokens, counts = _replace_rare(tokens, self.min_count)
        self.counts = Counter(mapped_tokens)
        self.total = sum(self.counts.values())
        self.vocab = {tok for tok, cnt in counts.items() if cnt >= self.min_count}
        self.vocab.add(UNK_TOKEN)

    def log_prob(self, token: str) -> float:
        if self.total == 0:
            return float("-inf")
        if token not in self.vocab:
            token = UNK_TOKEN
        return safe_log(self.counts[token] / self.total)


class BigramLM:
    def __init__(self, min_count: int = 2) -> None:
        self.min_count = min_count
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.vocab = set()

    def fit(self, tokens: List[str]) -> None:
        sentences = _sentence_split(tokens)
        flat_tokens = [tok for sent in sentences for tok in sent]
        mapped_tokens, counts = _replace_rare(flat_tokens, self.min_count)

        self.vocab = {tok for tok, cnt in counts.items() if cnt >= self.min_count}
        self.vocab.add(UNK_TOKEN)
        self.vocab.add(START_TOKEN)

        # Rebuild sentences with mapped tokens
        mapped_iter = iter(mapped_tokens)
        mapped_sentences: List[List[str]] = []
        for sent in sentences:
            mapped_sent = [next(mapped_iter) for _ in sent]
            mapped_sentences.append(mapped_sent)

        for sent in mapped_sentences:
            prev = START_TOKEN
            self.unigram_counts[prev] += 1
            for tok in sent:
                if tok not in self.vocab:
                    tok = UNK_TOKEN
                self.bigram_counts[(prev, tok)] += 1
                self.unigram_counts[tok] += 1
                prev = tok

    def log_prob(self, prev_token: str, token: str, add_k: float = 0.0) -> float:
        if prev_token not in self.vocab:
            prev_token = UNK_TOKEN
        if token not in self.vocab:
            token = UNK_TOKEN

        prev_count = self.unigram_counts[prev_token]
        bigram_count = self.bigram_counts[(prev_token, token)]

        if add_k == 0.0:
            return safe_log(bigram_count / prev_count) if prev_count > 0 else float("-inf")

        vocab_size = len(self.vocab)
        prob = (bigram_count + add_k) / (prev_count + add_k * vocab_size)
        return safe_log(prob)
