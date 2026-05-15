import json
import os
import numpy as np


def _tfidf_embeddings(texts, dim=768, seed=0):
    """Fallback: character n-gram TF-IDF projected to `dim` via random Gaussian matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    # character 2-4 grams work well for Chinese
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=4096)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    rng = np.random.RandomState(seed)
    proj = rng.randn(X.shape[1], dim).astype(np.float32) / np.sqrt(dim)
    embs = X @ proj
    norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
    return (embs / norms).astype(np.float32)


def gen_embeddings(segments_path, out_path, use_transformer=True):
    with open(segments_path, encoding="utf-8") as f:
        segs = json.load(f)
    texts = [s["text"] for s in segs]

    if use_transformer:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        except Exception as e:
            print(f"Transformer unavailable ({e}), falling back to TF-IDF projection.")
            embs = _tfidf_embeddings(texts)
    else:
        embs = _tfidf_embeddings(texts)

    np.save(out_path, embs)
    print(f"Embeddings saved: {embs.shape} -> {out_path}")
    return embs


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    gen_embeddings(
        os.path.join(base, "data/segments.json"),
        os.path.join(base, "data/embeddings.npy"),
    )
