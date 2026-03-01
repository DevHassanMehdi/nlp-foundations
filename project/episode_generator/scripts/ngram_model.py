import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

PUNCT = {".", "!", "?", ",", ";", ":"}


def _is_allowed_token(token: str) -> bool:
    if token in PUNCT:
        return True
    return bool(re.fullmatch(r"[a-z][a-z']*", token))


class BigramLM:
    def __init__(self, add_k: float = 0.2, min_count: int = 4, top_k: int = 50) -> None:
        self.add_k = add_k
        self.min_count = min_count
        self.top_k = top_k
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.vocab: List[str] = []

    def fit(self, tokens: List[str]) -> None:
        raw_counts = Counter(t for t in tokens if _is_allowed_token(t))
        self.vocab = sorted([t for t, c in raw_counts.items() if c >= self.min_count])
        vocab_set = set(self.vocab)
        filtered = [t if t in vocab_set else "<unk>" for t in tokens if _is_allowed_token(t)]

        self.unigram_counts = Counter(filtered)
        self.bigram_counts = Counter(zip(filtered[:-1], filtered[1:]))
        if "<unk>" not in self.vocab:
            self.vocab.append("<unk>")

    def prob(self, prev: str, token: str) -> float:
        vocab_size = len(self.vocab)
        prev_count = self.unigram_counts.get(prev, 0)
        bigram_count = self.bigram_counts.get((prev, token), 0)
        return (bigram_count + self.add_k) / (prev_count + self.add_k * vocab_size)

    def _sample_from_probs(self, items: List[Tuple[str, float]], rng: random.Random) -> str:
        total = sum(p for _, p in items)
        if total <= 0:
            return rng.choice(self.vocab)
        r = rng.random() * total
        acc = 0.0
        for tok, p in items:
            acc += p
            if acc >= r:
                return tok
        return items[-1][0]

    def sample_next(self, prev: str, rng: random.Random) -> str:
        prev = prev if prev in self.unigram_counts else "<unk>"
        scored = [(tok, self.prob(prev, tok)) for tok in self.vocab]
        scored.sort(key=lambda x: x[1], reverse=True)
        shortlist = scored[: self.top_k]
        return self._sample_from_probs(shortlist, rng)

    def _detok(self, tokens: List[str]) -> str:
        out = []
        for tok in tokens:
            if tok in {"<unk>", "<s>", "</s>"}:
                continue
            if tok in PUNCT and out:
                out[-1] += tok
            else:
                out.append(tok)
        text = " ".join(out)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def generate(self, prompt: str, max_words: int = 140, seed: int = 42) -> str:
        rng = random.Random(seed)
        prompt_tokens = [t.lower() for t in re.findall(r"[a-z']+|[.!?,;:]", prompt)]
        tokens = [t if t in self.vocab else "<unk>" for t in prompt_tokens if _is_allowed_token(t)]
        if not tokens:
            tokens = [rng.choice([t for t in self.vocab if t not in PUNCT and t != "<unk>"])]

        while len([t for t in tokens if t not in PUNCT]) < max_words:
            prev = tokens[-1]
            next_tok = self.sample_next(prev, rng)
            tokens.append(next_tok)

        return self._detok(tokens)

    def to_json(self) -> Dict[str, object]:
        return {
            "add_k": self.add_k,
            "min_count": self.min_count,
            "top_k": self.top_k,
            "unigram_counts": dict(self.unigram_counts),
            "bigram_counts": {"\t".join(k): v for k, v in self.bigram_counts.items()},
            "vocab": self.vocab,
        }

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> "BigramLM":
        model = cls(
            add_k=float(data.get("add_k", 0.2)),
            min_count=int(data.get("min_count", 4)),
            top_k=int(data.get("top_k", 50)),
        )
        model.unigram_counts = Counter(data["unigram_counts"])
        model.bigram_counts = Counter(
            {tuple(k.split("\t")): v for k, v in data["bigram_counts"].items()}
        )
        model.vocab = list(data["vocab"])
        return model


def save_model(path: Path, model: BigramLM) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(model.to_json(), f, indent=2, ensure_ascii=True)


def load_model(path: Path) -> BigramLM:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return BigramLM.from_json(data)
