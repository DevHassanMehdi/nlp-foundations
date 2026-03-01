from __future__ import annotations

from typing import Tuple

from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def load_transformer():
    return pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME)


def predict(model, text: str) -> Tuple[str, float]:
    result = model(text)[0]
    label = result["label"].lower()
    score = float(result["score"])
    if label in {"positive", "negative"}:
        return label, score
    # Fallback if model returns labels like LABEL_1
    if label.endswith("1"):
        return "positive", score
    return "negative", score
