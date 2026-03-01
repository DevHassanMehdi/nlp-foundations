from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .char_rnn import CharRNN, Vocab, batch_iter

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "corpus_chars.txt"
MODEL_PATH = BASE_DIR / "models" / "char_rnn.pt"


def _estimate_steps(data_len: int, seq_len: int, batch_size: int) -> int:
    upper = max(0, data_len - seq_len - 1)
    step = seq_len * batch_size
    return max(1, upper // step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train char-level RNN")
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_chars", type=int, default=350000)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print("Corpus not found. Run data_prep first.")
        return

    text = DATA_PATH.read_text(encoding="utf-8")
    if args.max_chars is not None and len(text) > args.max_chars:
        text = text[: args.max_chars]

    vocab = Vocab.build(text)
    data = vocab.encode(text)

    model = CharRNN(vocab_size=len(vocab.itos))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    expected_steps = _estimate_steps(len(data), args.seq_len, args.batch_size)
    print(f"Training on {len(data)} chars, vocab={len(vocab.itos)}, approx_steps/epoch={expected_steps}")

    if expected_steps <= 0:
        print("Not enough data to train.")
        return

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        steps = 0
        for x, y in batch_iter(data, args.seq_len, args.batch_size):
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} steps={steps}")

    torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos}, MODEL_PATH)
    print(f"Saved char RNN model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
