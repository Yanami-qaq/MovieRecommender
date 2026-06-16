"""
Main pipeline: load data -> preprocess -> train 5 models -> evaluate -> save results.
Run: python main.py
"""
import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import load_all
from src.data.preprocessor import (
    filter_sparse, split_data, build_content_features
)
from src.models.user_cf import UserCF
from src.models.item_cf import ItemCF
from src.models.svd_model import FunkSVD
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.evaluator import full_evaluate
from src.utils.helpers import save_results, results_to_dataframe

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_DIR = os.path.join(PROCESSED_DIR, "models")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Loading data")
    ratings, movies, users = load_all()
    print(f"  Ratings: {len(ratings):,}  |  Movies: {len(movies):,}  |  Users: {len(users):,}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\nStep 2: Preprocessing")
    ratings = filter_sparse(ratings, min_user_ratings=20, min_movie_ratings=10)
    print(f"  After filtering: {len(ratings):,} ratings")

    train, test = split_data(ratings, test_size=0.2)
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

    content_features = build_content_features(movies)
    movies_pkl = os.path.join(PROCESSED_DIR, "movies.pkl")
    movies.to_pickle(movies_pkl)
    content_features.to_pickle(os.path.join(PROCESSED_DIR, "content_features.pkl"))
    train.to_pickle(os.path.join(PROCESSED_DIR, "train.pkl"))
    test.to_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    print("  Preprocessed data saved.")

    # ── 3. Train models ───────────────────────────────────────────────────────
    print("\nStep 3: Training models")

    models = {
        "UserCF":      UserCF(n_neighbors=30),
        "ItemCF":      ItemCF(n_neighbors=30),
        "FunkSVD":     FunkSVD(n_factors=50, n_epochs=20),
        "ContentBased": ContentBasedRecommender(),
        "Hybrid":      HybridRecommender(cf_weight=0.6, cb_weight=0.4),
    }

    for name, model in models.items():
        print(f"\n  Training {name} ...")
        if name in ("ContentBased", "Hybrid"):
            model.fit(train, movies=movies, content_features=content_features)
        else:
            model.fit(train)
        pkl_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  {name} saved to {pkl_path}")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\nStep 4: Evaluating all models (K=10, 200 sampled users)")
    results = {}
    for name, model in models.items():
        metrics = full_evaluate(
            model, train, test, movies,
            k=10, n_users=200,
            content_features=content_features,
        )
        results[name] = metrics
        print(f"  {name}: {metrics}")

    # ── 5. Save & display ─────────────────────────────────────────────────────
    print("\nStep 5: Saving results")
    save_results(results)
    df = results_to_dataframe(results)
    print("\n" + "=" * 60)
    print("Final Results:")
    print(df.to_string())
    print("=" * 60)
    print("Pipeline complete. Run `streamlit run app/streamlit_app.py` to launch the demo.")


if __name__ == "__main__":
    main()
