"""
Data preprocessing: train/test split, user-item matrix, content features.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def split_data(ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Random 80/20 split stratified by user."""
    train, test = train_test_split(ratings, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_user_item_matrix(ratings: pd.DataFrame):
    """Return sparse user-item matrix and index mappings."""
    users = ratings["user_id"].unique()
    movies = ratings["movie_id"].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    movie2idx = {m: i for i, m in enumerate(movies)}

    row = ratings["user_id"].map(user2idx).values
    col = ratings["movie_id"].map(movie2idx).values
    data = ratings["rating"].values

    matrix = csr_matrix((data, (row, col)), shape=(len(users), len(movies)))
    return matrix, user2idx, movie2idx


def build_genre_features(movies: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode genres and return feature matrix indexed by movie_id."""
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies["genre_list"].fillna("").apply(
        lambda x: x if isinstance(x, list) else []
    ))
    feat_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movies["movie_id"])
    return feat_df


def build_content_features(movies: pd.DataFrame) -> pd.DataFrame:
    """
    TF-IDF on combined genre + title text for richer content similarity.
    Returns a DataFrame indexed by movie_id.
    """
    movies = movies.copy()
    movies["content"] = (
        movies["genres"].str.replace("|", " ", regex=False) + " " +
        movies["title_clean"].fillna("")
    )
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    mat = tfidf.fit_transform(movies["content"])
    feat_df = pd.DataFrame(mat.toarray(), index=movies["movie_id"],
                           columns=tfidf.get_feature_names_out())
    return feat_df


def get_user_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-user statistics for cold-start and analysis."""
    return ratings.groupby("user_id").agg(
        n_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
    ).reset_index()


def get_movie_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-movie statistics."""
    return ratings.groupby("movie_id").agg(
        n_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
    ).reset_index()


def filter_sparse(ratings: pd.DataFrame,
                  min_user_ratings: int = 20,
                  min_movie_ratings: int = 10) -> pd.DataFrame:
    """Remove very sparse users/movies to improve model quality."""
    user_counts = ratings["user_id"].value_counts()
    movie_counts = ratings["movie_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    filtered = ratings[
        ratings["user_id"].isin(valid_users) & ratings["movie_id"].isin(valid_movies)
    ]
    return filtered.reset_index(drop=True)
