from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CACHE_DIR = Path(__file__).resolve().parent / "model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CACHE_DIR / "classic_model.joblib"


@dataclass
class ClassicSentimentModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    def predict(self, text: str) -> Tuple[str, float]:
        X = self.vectorizer.transform([text])
        prob = self.classifier.predict_proba(X)[0, 1]
        label = "positive" if prob >= 0.5 else "negative"
        score = float(prob if label == "positive" else 1 - prob)
        return label, score


def train_or_load(seed: int = 42, max_samples: int = 4000) -> ClassicSentimentModel:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    dataset = load_dataset("glue", "sst2")
    train = dataset["train"].shuffle(seed=seed)
    subset = train.select(range(min(max_samples, len(train))))

    texts = subset["sentence"]
    labels = subset["label"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(max_iter=200, random_state=seed)
    classifier.fit(X, labels)

    model = ClassicSentimentModel(vectorizer=vectorizer, classifier=classifier)
    joblib.dump(model, MODEL_PATH)
    return model
