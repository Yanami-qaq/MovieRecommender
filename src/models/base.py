"""Base class for all recommenders."""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    name: str = "BaseRecommender"

    @abstractmethod
    def fit(self, train: pd.DataFrame, **kwargs):
        """Train the model on rating data."""

    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating for a (user, movie) pair."""

    def predict_batch(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        """Vectorized prediction for many movies. Override in subclasses for speed."""
        return np.array([self.predict(user_id, int(mid)) for mid in movie_ids])

    def recommend(self, user_id: int, movies: pd.DataFrame,
                  rated_movie_ids: set, top_k: int = 10) -> pd.DataFrame:
        """Return top-k recommended movies, uses predict_batch for efficiency."""
        candidates = movies[~movies["movie_id"].isin(rated_movie_ids)]["movie_id"].values
        scores = self.predict_batch(user_id, candidates)
        top_idx = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return pd.DataFrame({
            "movie_id": candidates[top_idx],
            "predicted_rating": scores[top_idx],
        })
