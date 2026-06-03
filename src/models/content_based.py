"""
Content-Based Filtering.
Builds a user profile from the weighted average of rated movie feature vectors,
then ranks unseen movies by cosine similarity to that profile.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    name = "ContentBased"

    def __init__(self):
        self.content_features_: pd.DataFrame = None

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame = None,
            content_features: pd.DataFrame = None, **kwargs):
        """
        Parameters
        ----------
        train : ratings DataFrame
        movies : movies DataFrame (used to build default genre features if content_features is None)
        content_features : pre-built feature DataFrame indexed by movie_id
        """
        if content_features is not None:
            self.content_features_ = content_features
        elif movies is not None:
            from src.data.preprocessor import build_content_features
            self.content_features_ = build_content_features(movies)
        else:
            raise ValueError("Provide either content_features or movies.")

        self.train_ = train
        self.global_mean_ = float(train["rating"].mean())

        # Build user profiles: weighted sum of movie feature vectors
        self.user_profiles_ = {}
        for user_id, group in train.groupby("user_id"):
            vecs, weights = [], []
            for _, row in group.iterrows():
                mid = row["movie_id"]
                if mid in self.content_features_.index:
                    vecs.append(self.content_features_.loc[mid].values)
                    weights.append(row["rating"])
            if vecs:
                w = np.array(weights)
                w = w / w.sum()
                profile = np.average(np.array(vecs), axis=0, weights=w)
                self.user_profiles_[user_id] = profile

        self.user_movie_rating_ = (
            train.set_index(["user_id", "movie_id"])["rating"].to_dict()
        )
        return self

    def predict_batch(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        profile = self.user_profiles_.get(user_id)
        if profile is None:
            return np.full(len(movie_ids), self.global_mean_, dtype=np.float32)
        norm_p = np.linalg.norm(profile)
        if norm_p == 0:
            return np.full(len(movie_ids), self.global_mean_, dtype=np.float32)

        user_ratings = self.train_[self.train_["user_id"] == user_id]["rating"]
        user_mean = float(user_ratings.mean()) if len(user_ratings) else self.global_mean_

        results = np.empty(len(movie_ids), dtype=np.float32)
        for k, mid in enumerate(movie_ids):
            mid = int(mid)
            if mid not in self.content_features_.index:
                results[k] = self.global_mean_
                continue
            v = self.content_features_.loc[mid].values
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                results[k] = self.global_mean_
                continue
            sim = max(0.0, np.dot(profile, v) / (norm_p * norm_v))
            results[k] = np.clip(user_mean + (sim - 0.5) * 2.0, 1.0, 5.0)
        return results

    def predict(self, user_id: int, movie_id: int) -> float:
        profile = self.user_profiles_.get(user_id)
        if profile is None:
            return self.global_mean_
        if movie_id not in self.content_features_.index:
            return self.global_mean_

        movie_vec = self.content_features_.loc[movie_id].values
        # cosine similarity in [0, 1] -> scale to [1, 5]
        norm_p = np.linalg.norm(profile)
        norm_m = np.linalg.norm(movie_vec)
        if norm_p == 0 or norm_m == 0:
            return self.global_mean_
        sim = np.dot(profile, movie_vec) / (norm_p * norm_m)
        sim = max(0.0, sim)
        # Map similarity to rating scale using user mean as anchor
        user_ratings = self.train_[self.train_["user_id"] == user_id]["rating"]
        user_mean = float(user_ratings.mean()) if len(user_ratings) else self.global_mean_
        pred = user_mean + (sim - 0.5) * 2.0  # center around user mean
        return float(np.clip(pred, 1.0, 5.0))
