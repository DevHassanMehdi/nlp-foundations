from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, text: str) -> "Vocab":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = chars
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int = 96, hidden_size: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, num_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def batch_iter(data: List[int], seq_len: int, batch_size: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    stride = seq_len
    upper = len(data) - seq_len - 1
    for i in range(0, upper, stride * batch_size):
        x_batch = []
        y_batch = []
        for b in range(batch_size):
            start = i + b * stride
            end = start + seq_len
            if end + 1 >= len(data):
                break
            x_batch.append(data[start:end])
            y_batch.append(data[start + 1 : end + 1])
        if x_batch:
            yield torch.tensor(x_batch, dtype=torch.long), torch.tensor(y_batch, dtype=torch.long)


def generate(
    model: CharRNN,
    vocab: Vocab,
    prompt: str,
    length: int,
    temperature: float = 0.8,
    top_k: int = 20,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    ids = vocab.encode(prompt)
    if not ids:
        ids = [0]

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    hidden = None

    with torch.no_grad():
        _, hidden = model(input_ids, hidden)
        last_id = input_ids[:, -1:]
        generated = ids[:]

        for _ in range(length):
            logits, hidden = model(last_id, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0 and top_k < logits.size(-1):
                values, indices = torch.topk(logits, k=top_k, dim=-1)
                probs = torch.softmax(values, dim=-1)
                next_local = torch.multinomial(probs, num_samples=1)
                next_id = indices.gather(-1, next_local)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            generated.append(int(next_id.item()))
            last_id = next_id

    return vocab.decode(generated)
