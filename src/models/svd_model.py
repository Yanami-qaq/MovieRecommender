"""
FunkSVD: matrix factorization with user/item biases trained via SGD.
Inspired by the Netflix Prize winning approach.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from .base import BaseRecommender


class FunkSVD(BaseRecommender):
    name = "FunkSVD"

    def __init__(self, n_factors: int = 50, n_epochs: int = 20,
                 lr: float = 0.005, reg: float = 0.02, seed: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, train: pd.DataFrame, **kwargs):
        rng = np.random.RandomState(self.seed)
        users = sorted(train["user_id"].unique())
        movies = sorted(train["movie_id"].unique())
        self.user2idx_ = {u: i for i, u in enumerate(users)}
        self.movie2idx_ = {m: i for i, m in enumerate(movies)}

        n_users, n_movies = len(users), len(movies)
        self.global_mean_ = float(train["rating"].mean())

        # Latent factors
        self.P_ = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.Q_ = rng.normal(0, 0.1, (n_movies, self.n_factors)).astype(np.float32)
        # Biases
        self.bu_ = np.zeros(n_users, dtype=np.float32)
        self.bi_ = np.zeros(n_movies, dtype=np.float32)

        # Map user/movie ids to indices once
        uid_arr = np.array([self.user2idx_[u] for u in train["user_id"]], dtype=np.int32)
        mid_arr = np.array([self.movie2idx_[m] for m in train["movie_id"]], dtype=np.int32)
        r_arr   = train["rating"].values.astype(np.float32)
        n       = len(r_arr)
        lr, reg = self.lr, self.reg

        for epoch in range(self.n_epochs):
            # Shuffle
            perm = rng.permutation(n)
            uid_s, mid_s, r_s = uid_arr[perm], mid_arr[perm], r_arr[perm]

            total_loss = 0.0
            # Mini-batch by chunk to avoid pure-Python overhead
            CHUNK = 4096
            for start in range(0, n, CHUNK):
                ui = uid_s[start:start+CHUNK]
                mi = mid_s[start:start+CHUNK]
                r  = r_s[start:start+CHUNK]

                # (chunk, factors)
                p = self.P_[ui]
                q = self.Q_[mi]

                pred = self.global_mean_ + self.bu_[ui] + self.bi_[mi] + (p * q).sum(axis=1)
                err  = r - pred
                total_loss += (err ** 2).sum()

                # Gradient updates (vectorized)
                p_new = p + lr * (err[:, None] * q - reg * p)
                q_new = q + lr * (err[:, None] * p - reg * q)
                np.add.at(self.bu_, ui, lr * (err - reg * self.bu_[ui]))
                np.add.at(self.bi_, mi, lr * (err - reg * self.bi_[mi]))
                # Write back latent factors (last write wins for repeated indices)
                self.P_[ui] = p_new
                self.Q_[mi] = q_new

            rmse_val = np.sqrt(total_loss / n)
            if (epoch + 1) % 5 == 0:
                print(f"  [FunkSVD] Epoch {epoch+1}/{self.n_epochs}  RMSE={rmse_val:.4f}")

        return self

    def predict_batch(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        ui = self.user2idx_.get(user_id)
        if ui is None:
            return np.full(len(movie_ids), self.global_mean_, dtype=np.float32)
        mid_arr = np.array([self.movie2idx_.get(int(m), -1) for m in movie_ids], dtype=np.int32)
        known = mid_arr >= 0
        scores = np.full(len(movie_ids), self.global_mean_, dtype=np.float32)
        if known.any():
            q = self.Q_[mid_arr[known]]
            scores[known] = (self.global_mean_ + self.bu_[ui]
                             + self.bi_[mid_arr[known]]
                             + q @ self.P_[ui])
        return np.clip(scores, 1.0, 5.0)

    def predict(self, user_id: int, movie_id: int) -> float:
        ui = self.user2idx_.get(user_id)
        mi = self.movie2idx_.get(movie_id)

        if ui is None or mi is None:
            return self.global_mean_

        pred = (self.global_mean_ + self.bu_[ui] + self.bi_[mi]
                + np.dot(self.P_[ui], self.Q_[mi]))
        return float(np.clip(pred, 1.0, 5.0))
