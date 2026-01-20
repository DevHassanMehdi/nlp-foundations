from typing import Dict, List, Tuple

import numpy as np


def train_test_split(
    data: List[Tuple[str, int]], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))
    split_idx = int(len(data) * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    return train, test


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
