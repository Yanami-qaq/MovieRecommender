"""
Grid search for FunkSVD hyperparameters.
Run: python -m src.analysis.hyperparam_search
"""
import os
import sys
import itertools

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import load_all
from src.data.preprocessor import filter_sparse, split_data
from src.models.svd_model import FunkSVD
from src.evaluation.metrics import rmse, mae
import numpy as np


PROCESSED = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def evaluate_svd_on_subset(model, test: pd.DataFrame, max_samples: int = 5000) -> dict:
    """Quick RMSE/MAE on a random subset of test ratings."""
    subset = test.sample(n=min(max_samples, len(test)), random_state=42)
    y_true, y_pred = [], []
    for _, row in subset.iterrows():
        y_pred.append(model.predict(int(row["user_id"]), int(row["movie_id"])))
        y_true.append(row["rating"])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}


def search_funksvd(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_factors_list: list = None,
    lr_list: list = None,
    reg_list: list = None,
    n_epochs: int = 10,
) -> pd.DataFrame:
    """Grid search over FunkSVD hyperparameters."""
    n_factors_list = n_factors_list or [20, 50, 80]
    lr_list = lr_list or [0.003, 0.005, 0.01]
    reg_list = reg_list or [0.01, 0.02, 0.05]

    rows = []
    combos = list(itertools.product(n_factors_list, lr_list, reg_list))
    print(f"Searching {len(combos)} hyperparameter combinations ...")

    for nf, lr, reg in combos:
        print(f"  n_factors={nf}, lr={lr}, reg={reg}")
        model = FunkSVD(n_factors=nf, n_epochs=n_epochs, lr=lr, reg=reg)
        model.fit(train)
        metrics = evaluate_svd_on_subset(model, test)
        rows.append({
            "n_factors": nf, "lr": lr, "reg": reg,
            **metrics,
        })

    df = pd.DataFrame(rows).sort_values("RMSE")
    return df


def main():
    print("Loading data for hyperparameter search ...")
    ratings, _, _ = load_all()
    ratings = filter_sparse(ratings, min_user_ratings=20, min_movie_ratings=10)
    train, test = split_data(ratings, test_size=0.2)

    results = search_funksvd(train, test, n_epochs=10)
    out_path = os.path.join(PROCESSED, "svd_hyperparam_search.csv")
    os.makedirs(PROCESSED, exist_ok=True)
    results.to_csv(out_path, index=False)

    print("\nTop-5 configurations:")
    print(results.head().to_string(index=False))
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
