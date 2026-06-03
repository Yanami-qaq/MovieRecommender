"""
Item-Based Collaborative Filtering (ItemCF).
Predicts rating by weighted average of similar items the user has already rated.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender


class ItemCF(BaseRecommender):
    name = "ItemCF"

    def __init__(self, n_neighbors: int = 30):
        self.n_neighbors = n_neighbors

    def fit(self, train: pd.DataFrame, **kwargs):
        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self.user2idx_ = {u: i for i, u in enumerate(users)}
        self.movie2idx_ = {m: i for i, m in enumerate(movies)}
        self.idx2movie_ = {i: m for m, i in self.movie2idx_.items()}

        n_users, n_movies = len(users), len(movies)
        self.matrix_ = np.zeros((n_users, n_movies), dtype=np.float32)
        for _, row in train.iterrows():
            ui = self.user2idx_.get(row["user_id"])
            mi = self.movie2idx_.get(row["movie_id"])
            if ui is not None and mi is not None:
                self.matrix_[ui, mi] = row["rating"]

        # Item similarity: columns are items, so transpose first
        # cosine_similarity expects (n_samples, n_features)
        self.item_sim_ = cosine_similarity(self.matrix_.T)
        np.fill_diagonal(self.item_sim_, 0)

        self.user_movie_rating_ = (
            train.set_index(["user_id", "movie_id"])["rating"].to_dict()
        )
        self.user_rated_ = (
            train.groupby("user_id")["movie_id"].apply(list).to_dict()
        )
        self.global_mean_ = float(train["rating"].mean())
        return self

    def predict_batch(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        if user_id not in self.user2idx_:
            return np.full(len(movie_ids), self.global_mean_, dtype=np.float32)
        rated_movies = self.user_rated_.get(user_id, [])
        if not rated_movies:
            return np.full(len(movie_ids), self.global_mean_, dtype=np.float32)

        # Build rated index array and rating array once
        rated_idxs = np.array([self.movie2idx_[rm] for rm in rated_movies
                                if rm in self.movie2idx_], dtype=np.int32)
        rated_ratings = np.array([self.user_movie_rating_.get((user_id, rm), 0)
                                   for rm in rated_movies if rm in self.movie2idx_],
                                  dtype=np.float32)
        if len(rated_idxs) == 0:
            return np.full(len(movie_ids), self.global_mean_, dtype=np.float32)

        results = np.empty(len(movie_ids), dtype=np.float32)
        for k, movie_id in enumerate(movie_ids):
            if movie_id not in self.movie2idx_:
                results[k] = self.global_mean_
                continue
            mi = self.movie2idx_[int(movie_id)]
            sims = self.item_sim_[mi, rated_idxs]
            top_k_mask = np.argsort(sims)[::-1][:self.n_neighbors]
            s = sims[top_k_mask]
            r = rated_ratings[top_k_mask]
            denom = np.abs(s).sum()
            results[k] = float(np.clip((s * r).sum() / denom, 1.0, 5.0)) if denom > 0 else self.global_mean_
        return results

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user2idx_ or movie_id not in self.movie2idx_:
            return self.global_mean_

        mi = self.movie2idx_[movie_id]
        rated_movies = self.user_rated_.get(user_id, [])

        sims = []
        for rm in rated_movies:
            if rm == movie_id or rm not in self.movie2idx_:
                continue
            rmi = self.movie2idx_[rm]
            s = self.item_sim_[mi, rmi]
            r = self.user_movie_rating_.get((user_id, rm), 0)
            sims.append((s, r))

        # Use top-k most similar
        sims.sort(key=lambda x: x[0], reverse=True)
        sims = sims[:self.n_neighbors]

        denom = sum(abs(s) for s, _ in sims)
        if denom == 0:
            return self.global_mean_

        pred = sum(s * r for s, r in sims) / denom
        return float(np.clip(pred, 1.0, 5.0))
