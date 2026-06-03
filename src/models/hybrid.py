"""
Hybrid Recommender: weighted combination of ItemCF and ContentBased.
Falls back gracefully when one component cannot predict.
"""
import numpy as np
import pandas as pd
from .base import BaseRecommender
from .item_cf import ItemCF
from .content_based import ContentBasedRecommender


class HybridRecommender(BaseRecommender):
    name = "Hybrid"

    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4,
                 n_neighbors: int = 30):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.n_neighbors = n_neighbors

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame = None,
            content_features: pd.DataFrame = None, **kwargs):
        self.cf_ = ItemCF(n_neighbors=self.n_neighbors)
        self.cf_.fit(train)

        self.cb_ = ContentBasedRecommender()
        self.cb_.fit(train, movies=movies, content_features=content_features)

        self.global_mean_ = float(train["rating"].mean())
        return self

    def predict_batch(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        cf_scores = self.cf_.predict_batch(user_id, movie_ids)
        cb_scores = self.cb_.predict_batch(user_id, movie_ids)
        user_n = len(self.cf_.user_rated_.get(user_id, []))
        w_cf, w_cb = (0.2, 0.8) if user_n < 5 else (self.cf_weight, self.cb_weight)
        return np.clip(w_cf * cf_scores + w_cb * cb_scores, 1.0, 5.0)

    def predict(self, user_id: int, movie_id: int) -> float:
        cf_pred = self.cf_.predict(user_id, movie_id)
        cb_pred = self.cb_.predict(user_id, movie_id)

        # Adaptive weighting: trust CF more when user has enough ratings
        user_n = len(self.cf_.user_rated_.get(user_id, []))
        if user_n < 5:
            # Cold-start: rely more on content
            w_cf, w_cb = 0.2, 0.8
        else:
            w_cf, w_cb = self.cf_weight, self.cb_weight

        pred = w_cf * cf_pred + w_cb * cb_pred
        return float(np.clip(pred, 1.0, 5.0))
