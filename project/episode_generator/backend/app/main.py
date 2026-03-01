from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from project.episode_generator.scripts.generate import (
    generate_lstm_structured_with_model,
    generate_ngram_structured_with_model,
    generate_pretrained_structured_with_model,
)
from project.episode_generator.scripts.ngram_model import load_model
from project.episode_generator.scripts.pretrained_model import (
    DEFAULT_MODEL_NAME,
    load_pretrained_generator,
)
from project.episode_generator.scripts.word_lstm import (
    WordVocab,
    build_model_from_checkpoint,
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="Episode Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.72
    seed: int = 42


class GenerateResponse(BaseModel):
    prompt: str
    ngram: str
    word_lstm: str
    pretrained: str


ngram_model = None
lstm_model = None
lstm_vocab = None
pretrained_generator = None


@app.on_event("startup")
def _startup() -> None:
    global ngram_model, lstm_model, lstm_vocab, pretrained_generator
    ngram_path = MODELS_DIR / "ngram_model.json"
    lstm_path = MODELS_DIR / "word_lstm.pt"

    if ngram_path.exists():
        ngram_model = load_model(ngram_path)
    if lstm_path.exists():
        checkpoint = torch.load(lstm_path, map_location="cpu")
        lstm_vocab = WordVocab(
            stoi={t: i for i, t in enumerate(checkpoint["vocab"])},
            itos=checkpoint["vocab"],
        )
        lstm_model = build_model_from_checkpoint(checkpoint, vocab_size=len(lstm_vocab.itos))
        lstm_model.load_state_dict(checkpoint["state_dict"])

    try:
        pretrained_generator = load_pretrained_generator(DEFAULT_MODEL_NAME)
    except Exception:
        pretrained_generator = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    prompt = req.prompt.strip()
    if not prompt:
        return GenerateResponse(prompt="", ngram="", word_lstm="", pretrained="")

    ngram_text = "(model missing)"
    if ngram_model is not None:
        ngram_text = generate_ngram_structured_with_model(
            ngram_model, prompt, req.temperature, req.seed
        )

    lstm_text = "(model missing)"
    if lstm_model is not None and lstm_vocab is not None:
        lstm_text = generate_lstm_structured_with_model(
            lstm_model, lstm_vocab, prompt, req.temperature, req.seed
        )

    pretrained_text = "(transformer model unavailable)"
    if pretrained_generator is not None:
        pretrained_text = generate_pretrained_structured_with_model(
            pretrained_generator, prompt, req.temperature, req.seed
        )

    return GenerateResponse(
        prompt=prompt, ngram=ngram_text, word_lstm=lstm_text, pretrained=pretrained_text
    )
