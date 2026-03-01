from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch import amp

from .word_lstm import WordLSTM, WordVocab, batch_iter, tokenize

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "corpus_words.txt"
MODEL_PATH = BASE_DIR / "models" / "word_lstm.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train word-level LSTM")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_tokens", type=int, default=140000)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=16000)
    parser.add_argument("--emb_dim", type=int, default=192)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print("Corpus not found. Run data_prep first.")
        return

    text = DATA_PATH.read_text(encoding="utf-8")
    tokens = tokenize(text)
    if args.max_tokens is not None and len(tokens) > args.max_tokens:
        tokens = tokens[: args.max_tokens]

    vocab = WordVocab.build(tokens, min_count=args.min_count, max_vocab=args.max_vocab)
    ids = [vocab.stoi["<bos>"]] + vocab.encode(tokens) + [vocab.stoi["<eos>"]]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    use_amp = device == "cuda"
    model = WordLSTM(
        vocab_size=len(vocab.itos),
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"Training on tokens={len(tokens)}, vocab={len(vocab.itos)}, "
        f"device={device}, amp={use_amp}, grad_accum={args.grad_accum_steps}"
    )
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for x, y in batch_iter(ids, args.seq_len, args.batch_size):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (steps + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total += loss.item() * args.grad_accum_steps
            steps += 1
        if steps % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        avg = total / max(steps, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: loss={avg:.4f} steps={steps} lr={lr_now:.6f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab.itos,
            "model_config": {
                "emb_dim": args.emb_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
        },
        MODEL_PATH,
    )
    print(f"Saved word LSTM model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
