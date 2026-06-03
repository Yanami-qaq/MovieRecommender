"""
Evaluator: runs rating-prediction and ranking evaluation for a recommender.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, coverage, intra_list_diversity
from src.models.base import BaseRecommender


RATING_THRESHOLD = 4.0  # rating >= this is considered "relevant"


def evaluate_rating_prediction(model: BaseRecommender,
                                test: pd.DataFrame) -> dict:
    """Compute RMSE and MAE on held-out ratings."""
    y_true, y_pred = [], []
    for _, row in test.iterrows():
        pred = model.predict(int(row["user_id"]), int(row["movie_id"]))
        y_true.append(row["rating"])
        y_pred.append(pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}


def evaluate_ranking(model: BaseRecommender,
                     train: pd.DataFrame,
                     test: pd.DataFrame,
                     movies: pd.DataFrame,
                     k: int = 10,
                     n_users: int = 200,
                     content_features: pd.DataFrame = None,
                     seed: int = 42) -> dict:
    """
    Sample n_users from test set and compute ranking metrics.
    'Relevant' = items in the test set with rating >= RATING_THRESHOLD.
    """
    rng = np.random.RandomState(seed)
    test_users = test["user_id"].unique()
    sampled = rng.choice(test_users, size=min(n_users, len(test_users)), replace=False)

    train_rated = train.groupby("user_id")["movie_id"].apply(set).to_dict()

    all_recs, p_scores, r_scores, ndcg_scores, div_scores = [], [], [], [], []

    for uid in tqdm(sampled, desc=f"  Ranking eval ({model.name})", leave=False):
        user_test = test[test["user_id"] == uid]
        relevant = set(user_test[user_test["rating"] >= RATING_THRESHOLD]["movie_id"])
        if not relevant:
            continue

        rated = train_rated.get(uid, set())
        rec_df = model.recommend(uid, movies, rated_movie_ids=rated, top_k=k)
        rec_list = rec_df["movie_id"].tolist()

        all_recs.append(rec_list)
        p_scores.append(precision_at_k(rec_list, relevant, k))
        r_scores.append(recall_at_k(rec_list, relevant, k))
        ndcg_scores.append(ndcg_at_k(rec_list, relevant, k))
        if content_features is not None:
            div_scores.append(intra_list_diversity(rec_list, content_features))

    n_items = movies["movie_id"].nunique()
    result = {
        f"Precision@{k}": float(np.mean(p_scores)) if p_scores else 0.0,
        f"Recall@{k}":    float(np.mean(r_scores)) if r_scores else 0.0,
        f"NDCG@{k}":      float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "Coverage":       coverage(all_recs, n_items),
    }
    if div_scores:
        result["Diversity"] = float(np.mean(div_scores))
    return result


def full_evaluate(model: BaseRecommender,
                  train: pd.DataFrame,
                  test: pd.DataFrame,
                  movies: pd.DataFrame,
                  k: int = 10,
                  n_users: int = 200,
                  content_features: pd.DataFrame = None) -> dict:
    """Run both rating and ranking evaluation, return combined dict."""
    print(f"\nEvaluating {model.name} ...")
    rating_metrics = evaluate_rating_prediction(model, test)
    ranking_metrics = evaluate_ranking(model, train, test, movies, k=k,
                                       n_users=n_users,
                                       content_features=content_features)
    return {**rating_metrics, **ranking_metrics}
