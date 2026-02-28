from typing import Dict, List, Tuple

from gensim.models import Word2Vec


def compute_similarities(
    model: Word2Vec, pairs: List[Tuple[str, str]]
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for w1, w2 in pairs:
        if w1 in model.wv and w2 in model.wv:
            score = float(model.wv.similarity(w1, w2))
            status = "ok"
        else:
            score = None
            status = "oov"
        results.append({"word1": w1, "word2": w2, "similarity": score, "status": status})
    return results


def nearest_neighbors(
    model: Word2Vec, words: List[str], topn: int = 5
) -> Dict[str, List[Dict[str, object]]]:
    output: Dict[str, List[Dict[str, object]]] = {}
    for w in words:
        if w not in model.wv:
            output[w] = []
            continue
        neighbors = model.wv.most_similar(w, topn=topn)
        output[w] = [{"word": n, "score": float(s)} for n, s in neighbors]
    return output
