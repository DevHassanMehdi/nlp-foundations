from typing import List

from gensim.models import Word2Vec


def train_word2vec(
    sentences: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 5,
    seed: int = 42,
) -> Word2Vec:
    """Train a Word2Vec model from tokenized sentences."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
        seed=seed,
        epochs=epochs,
    )
    return model
