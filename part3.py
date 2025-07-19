from __future__ import annotations

import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import feedparser
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Prevent joblib from querying physical CPU cores when running on systems
# where utilities like ``wmic`` are unavailable (e.g. some Windows setups).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def fetch_documents(num_per_category: int = 10) -> Tuple[List[str], np.ndarray, List[str]]:
    """Fetch articles from CNNIndonesia RSS feeds.

    Parameters
    ----------
    num_per_category:
        Number of articles to collect from each category.

    Returns
    -------
    docs:
        List of article texts composed of title and summary.
    labels:
        Array of integer labels corresponding to categories.
    categories:
        Names of the categories used.
    """

    feeds = {
        "olahraga": "https://www.cnnindonesia.com/olahraga/rss",
        "teknologi": "https://www.cnnindonesia.com/teknologi/rss",
    }

    docs: List[str] = []
    labels = []
    categories = list(feeds.keys())
    for label_idx, (name, url) in enumerate(feeds.items()):
        feed = feedparser.parse(url)
        for entry in feed.entries[:num_per_category]:
            text = f"{entry.get('title', '')} {entry.get('summary', '')}"
            docs.append(text)
            labels.append(label_idx)

    return docs, np.array(labels), categories


def compute_tfidf_vectors(documents: List[str]) -> np.ndarray:
    """Compute TF-IDF vectors for documents."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    return vectors.toarray()


def compute_word2vec_vectors(documents: List[str]) -> np.ndarray:
    """Train Word2Vec and obtain average document vectors."""
    tokenized = [re.sub(r"[^\w\s]", "", doc.lower()).split() for doc in documents]
    model = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=1, workers=1, sg=1)

    vectors = []
    for tokens in tokenized:
        if not tokens:
            vectors.append(np.zeros(model.vector_size))
            continue
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        vectors.append(np.mean(word_vectors, axis=0))
    return np.vstack(vectors)


def similarity_stats(vectors: np.ndarray, labels: np.ndarray, method: str) -> Tuple[pd.Series, np.ndarray]:
    """Calculate average similarities within and across categories."""
    sim_matrix = cosine_similarity(vectors)
    sports = labels == 0
    tech = labels == 1

    def avg_sim(mat: np.ndarray) -> float:
        if mat.size == 0:
            return 0.0
        triu = mat[np.triu_indices_from(mat, k=1)]
        return float(np.mean(triu)) if triu.size else 0.0

    results = pd.Series(
        {
            "Method": method,
            "Sports vs Sports": avg_sim(sim_matrix[np.ix_(sports, sports)]),
            "Tech vs Tech": avg_sim(sim_matrix[np.ix_(tech, tech)]),
            "Sports vs Tech": float(np.mean(sim_matrix[np.ix_(sports, tech)])),
        }
    )
    return results, sim_matrix


def plot_tsne(vectors: np.ndarray, labels: np.ndarray, categories: List[str], filename: str) -> None:
    """Save a 2D t-SNE plot of document vectors.

    Parameters
    ----------
    vectors:
        Vector representations of the documents.
    labels:
        Integer category labels for each document.
    categories:
        Names corresponding to ``labels``.
    filename:
        Output filename for the saved plot.
    """
    n_samples = len(vectors)
    # Perplexity must be less than the number of samples. Pick a value that
    # works for small corpora while keeping a sensible default for larger ones.
    perplexity = max(5, min(30, n_samples - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    points = tsne.fit_transform(vectors)

    plt.figure(figsize=(6, 5))
    for idx, category in enumerate(categories):
        subset = points[labels == idx]
        plt.scatter(subset[:, 0], subset[:, 1], label=category, alpha=0.7)
    plt.legend()
    plt.title(filename.replace(".png", ""))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main() -> None:
    docs, labels, categories = fetch_documents()

    tfidf_vectors = compute_tfidf_vectors(docs)
    word2vec_vectors = compute_word2vec_vectors(docs)

    tfidf_stats, _ = similarity_stats(tfidf_vectors, labels, "TF-IDF")
    w2v_stats, _ = similarity_stats(word2vec_vectors, labels, "Word2Vec")

    summary = pd.DataFrame([tfidf_stats, w2v_stats])
    print("\nAverage Cosine Similarities:\n")
    print(summary.to_string(index=False))

    plot_tsne(tfidf_vectors, labels, categories, "tfidf_tsne.png")
    plot_tsne(word2vec_vectors, labels, categories, "word2vec_tsne.png")

    print("\nPlots saved as 'tfidf_tsne.png' and 'word2vec_tsne.png'.")


if __name__ == "__main__":
    main()