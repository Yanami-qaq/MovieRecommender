"""
Data loading module: downloads MovieLens 1M and loads into DataFrames.
"""
import io
import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def _download(url: str, dest: str) -> None:
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            buf.write(chunk)
            bar.update(len(chunk))
    with zipfile.ZipFile(buf) as z:
        z.extractall(dest)
    print("Download complete.")


def ensure_data(raw_dir: str = RAW_DIR) -> str:
    """Return path to ml-1m directory, downloading if needed."""
    ml_dir = os.path.join(raw_dir, "ml-1m")
    if not os.path.isdir(ml_dir):
        os.makedirs(raw_dir, exist_ok=True)
        _download(ML1M_URL, raw_dir)
    return ml_dir


def load_ratings(ml_dir: str) -> pd.DataFrame:
    path = os.path.join(ml_dir, "ratings.dat")
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    df["rating"] = df["rating"].astype(float)
    return df


def load_movies(ml_dir: str) -> pd.DataFrame:
    path = os.path.join(ml_dir, "movies.dat")
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
    df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
    df["genre_list"] = df["genres"].str.split("|")
    return df


def load_users(ml_dir: str) -> pd.DataFrame:
    path = os.path.join(ml_dir, "users.dat")
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    return df


def load_all(raw_dir: str = RAW_DIR):
    """Download if needed and return (ratings, movies, users) DataFrames."""
    ml_dir = ensure_data(raw_dir)
    ratings = load_ratings(ml_dir)
    movies = load_movies(ml_dir)
    users = load_users(ml_dir)
    return ratings, movies, users
