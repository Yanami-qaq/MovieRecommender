"""
User-Based Collaborative Filtering (UserCF).
Predicts rating by weighted average of similar users' ratings.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender


class UserCF(BaseRecommender):
    name = "UserCF"

    def __init__(self, n_neighbors: int = 30, min_common: int = 3):
        self.n_neighbors = n_neighbors
        self.min_common = min_common

    def fit(self, train: pd.DataFrame, **kwargs):
        self.train_ = train
        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self.user2idx_ = {u: i for i, u in enumerate(users)}
        self.movie2idx_ = {m: i for i, m in enumerate(movies)}
        self.idx2user_ = {i: u for u, i in self.user2idx_.items()}

        # Build dense mean-centered matrix for similarity
        n_users, n_movies = len(users), len(movies)
        self.matrix_ = np.zeros((n_users, n_movies), dtype=np.float32)
        for _, row in train.iterrows():
            ui = self.user2idx_.get(row["user_id"])
            mi = self.movie2idx_.get(row["movie_id"])
            if ui is not None and mi is not None:
                self.matrix_[ui, mi] = row["rating"]

        # Mean-center per user (ignore zeros)
        self.user_means_ = np.zeros(n_users)
        for i in range(n_users):
            rated = self.matrix_[i] != 0
            if rated.sum() > 0:
                self.user_means_[i] = self.matrix_[i, rated].mean()

        centered = self.matrix_.copy()
        for i in range(n_users):
            mask = centered[i] != 0
            centered[i, mask] -= self.user_means_[i]

        # Precompute cosine similarity
        self.sim_ = cosine_similarity(centered)
        np.fill_diagonal(self.sim_, 0)

        # Build user -> rated movies lookup
        self.user_rated_ = (
            train.groupby("user_id")["movie_id"].apply(set).to_dict()
        )
        self.user_movie_rating_ = (
            train.set_index(["user_id", "movie_id"])["rating"].to_dict()
        )
        return self

    def predict_batch(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        if user_id not in self.user2idx_:
            return np.full(len(movie_ids), self.user_means_.mean())
        ui = self.user2idx_[user_id]
        sims = self.sim_[ui]
        neighbor_idxs = np.argsort(sims)[::-1][:self.n_neighbors * 3]

        results = np.empty(len(movie_ids), dtype=np.float32)
        for k, movie_id in enumerate(movie_ids):
            results[k] = self._predict_with_ui(ui, int(movie_id), sims, neighbor_idxs)
        return results

    def _predict_with_ui(self, ui, movie_id, sims, neighbor_idxs):
        if movie_id not in self.movie2idx_:
            return self.user_means_[ui] if self.user_means_[ui] != 0 else 3.5
        mi = self.movie2idx_[movie_id]
        numerator, denominator, count = 0.0, 0.0, 0
        for ni in neighbor_idxs:
            if count >= self.n_neighbors:
                break
            neighbor_id = self.idx2user_[ni]
            r = self.user_movie_rating_.get((neighbor_id, movie_id))
            if r is None:
                continue
            common = (self.matrix_[ui] != 0) & (self.matrix_[ni] != 0)
            if common.sum() < self.min_common:
                continue
            w = sims[ni]
            numerator += w * (r - self.user_means_[ni])
            denominator += abs(w)
            count += 1
        if denominator == 0:
            return float(self.user_means_[ui]) if self.user_means_[ui] != 0 else 3.5
        pred = self.user_means_[ui] + numerator / denominator
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user2idx_:
            return self.user_means_.mean() if len(self.user_means_) else 3.5
        ui = self.user2idx_[user_id]
        sims = self.sim_[ui]
        neighbor_idxs = np.argsort(sims)[::-1][:self.n_neighbors * 3]
        return self._predict_with_ui(ui, movie_id, sims, neighbor_idxs)
