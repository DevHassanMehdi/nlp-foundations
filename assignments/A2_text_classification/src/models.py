from typing import Dict, Tuple

import numpy as np


class MultinomialNB:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_log_prior: Dict[int, float] = {}
        self.feature_log_prob: Dict[int, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        classes = np.unique(y)
        n_docs = len(y)
        vocab_size = X.shape[1]
        for c in classes:
            X_c = X[y == c]
            class_count = X_c.shape[0]
            self.class_log_prior[int(c)] = np.log(class_count / n_docs)
            word_counts = X_c.sum(axis=0) + self.alpha
            denom = word_counts.sum()
            self.feature_log_prob[int(c)] = np.log(word_counts / denom)

    def predict(self, X: np.ndarray) -> np.ndarray:
        classes = sorted(self.class_log_prior.keys())
        log_probs = []
        for c in classes:
            log_prior = self.class_log_prior[c]
            log_likelihood = X @ self.feature_log_prob[c]
            log_probs.append(log_prior + log_likelihood)
        log_probs = np.vstack(log_probs).T
        preds = np.argmax(log_probs, axis=1)
        return preds


class LogisticRegression:
    def __init__(self, lr: float = 0.1, epochs: int = 10, l2: float = 0.0) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.epochs):
            logits = X @ self.weights + self.bias
            probs = self._sigmoid(logits)
            errors = probs - y
            grad_w = (X.T @ errors) / n_samples
            if self.l2 > 0:
                grad_w += self.l2 * self.weights
            grad_b = errors.mean()
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not fitted")
        logits = X @ self.weights + self.bias
        probs = self._sigmoid(logits)
        return (probs >= 0.5).astype(int)
