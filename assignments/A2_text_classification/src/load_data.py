from typing import List, Tuple

import nltk
from nltk.corpus import movie_reviews


def load_movie_reviews() -> List[Tuple[str, int]]:
    """Load NLTK movie reviews as (text, label) where label 1=pos, 0=neg."""
    try:
        data: List[Tuple[str, int]] = []
        for category in movie_reviews.categories():
            label = 1 if category == "pos" else 0
            for file_id in movie_reviews.fileids(category):
                text = movie_reviews.raw(file_id)
                data.append((text, label))
        return data
    except LookupError:
        print(
            "NLTK data not found. Please run: "
            "python -m nltk.downloader punkt punkt_tab movie_reviews"
        )
        return []
