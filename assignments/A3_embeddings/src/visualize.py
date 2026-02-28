from typing import List, Tuple

import numpy as np
from gensim.models import Word2Vec


def pca_reduce(vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce vectors to n_components using PCA (SVD-based)."""
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return centered @ components.T


def pca_words(model: Word2Vec, words: List[str]) -> List[Tuple[str, float, float]]:
    vectors = []
    kept_words = []
    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])
            kept_words.append(w)
    if not vectors:
        return []

    reduced = pca_reduce(np.array(vectors), n_components=2)
    return [(w, float(x), float(y)) for w, (x, y) in zip(kept_words, reduced)]
