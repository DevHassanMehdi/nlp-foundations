from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn

TOKEN_RE = re.compile(r"[a-z']+|[.!?,;:]")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens: Sequence[str]) -> str:
    out: List[str] = []
    for tok in tokens:
        if tok in {"<pad>", "<bos>", "<eos>", "<unk>"}:
            continue
        if tok in {".", "!", "?", ",", ";", ":"} and out:
            out[-1] += tok
        else:
            out.append(tok)
    text = " ".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class WordVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, tokens: List[str], min_count: int = 2, max_vocab: int = 20000) -> "WordVocab":
        from collections import Counter

        counts = Counter(tokens)
        kept = [t for t, c in counts.items() if c >= min_count]
        kept.sort(key=lambda t: counts[t], reverse=True)
        kept = kept[: max_vocab - 4]

        itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + kept
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] for i in ids]


class WordLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 192,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def build_model_from_checkpoint(checkpoint: Dict[str, object], vocab_size: int) -> "WordLSTM":
    cfg = checkpoint.get("model_config", {})
    model = WordLSTM(
        vocab_size=vocab_size,
        emb_dim=int(cfg.get("emb_dim", 192)),
        hidden_size=int(cfg.get("hidden_size", 384)),
        num_layers=int(cfg.get("num_layers", 2)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    return model


def batch_iter(data: List[int], seq_len: int, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    step = seq_len * batch_size
    upper = len(data) - seq_len - 1
    for i in range(0, upper, step):
        x_batch: List[List[int]] = []
        y_batch: List[List[int]] = []
        for b in range(batch_size):
            s = i + b * seq_len
            e = s + seq_len
            if e + 1 >= len(data):
                break
            x_batch.append(data[s:e])
            y_batch.append(data[s + 1 : e + 1])
        if x_batch:
            yield torch.tensor(x_batch, dtype=torch.long), torch.tensor(y_batch, dtype=torch.long)


def generate(
    model: WordLSTM,
    vocab: WordVocab,
    prompt: str,
    max_words: int = 180,
    temperature: float = 0.75,
    top_k: int = 20,
    repetition_penalty: float = 1.12,
    no_repeat_ngram_size: int = 3,
    seed: int | None = None,
) -> str:
    model.eval()
    device = next(model.parameters()).device

    prompt_tokens = tokenize(prompt)
    ids = vocab.encode(prompt_tokens)
    if not ids:
        ids = [vocab.stoi["<bos>"]]

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    hidden = None

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    with torch.no_grad():
        _, hidden = model(x, hidden)
        cur = x[:, -1:]
        generated = ids[:]

        for _ in range(max_words):
            logits, hidden = model(cur, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            # Light repetition penalty on recent tokens to reduce loops.
            recent = generated[-50:]
            if recent:
                for tok_id in set(recent):
                    logits[0, tok_id] /= repetition_penalty

            # Ban candidates that would recreate any seen n-gram.
            if no_repeat_ngram_size >= 2 and len(generated) >= no_repeat_ngram_size - 1:
                prefix = tuple(generated[-(no_repeat_ngram_size - 1) :])
                seen = set()
                for i in range(0, len(generated) - no_repeat_ngram_size + 1):
                    ng = tuple(generated[i : i + no_repeat_ngram_size])
                    if ng[:-1] == prefix:
                        seen.add(ng[-1])
                if seen:
                    logits[0, list(seen)] = float("-inf")

            if 0 < top_k < logits.size(-1):
                vals, idxs = torch.topk(logits, k=top_k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                sample = torch.multinomial(probs, num_samples=1, generator=generator)
                next_id = idxs.gather(-1, sample)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)

            nid = int(next_id.item())
            generated.append(nid)
            cur = next_id
            if nid == vocab.stoi["<eos>"]:
                break

    return detokenize(vocab.decode(generated))
