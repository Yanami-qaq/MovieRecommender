"""
Evaluation metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K, Coverage, Diversity.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for m in top_k if m in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, m in enumerate(top_k) if m in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_recommendations: list, n_items: int) -> float:
    """Fraction of items ever recommended."""
    recommended_items = set(m for rec in all_recommendations for m in rec)
    return len(recommended_items) / n_items if n_items > 0 else 0.0


def intra_list_diversity(recommended: list, content_features: pd.DataFrame) -> float:
    """Average pairwise dissimilarity within a recommendation list."""
    feats = [content_features.loc[m].values
             for m in recommended if m in content_features.index]
    if len(feats) < 2:
        return 0.0
    mat = np.array(feats)
    sim_matrix = cosine_similarity(mat)
    n = len(feats)
    total = (sim_matrix.sum() - n) / (n * (n - 1))  # average off-diagonal sim
    return float(1.0 - total)  # diversity = 1 - similarity
