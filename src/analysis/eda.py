"""
Exploratory data analysis: compute stats and save figures.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def rating_distribution(ratings: pd.DataFrame, save_path: str = None) -> pd.Series:
    """Plot rating value counts and return the distribution."""
    counts = ratings["rating"].value_counts().sort_index()
    if save_path:
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=ax, color="#2c5f8a", edgecolor="white")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title("Rating Distribution")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return counts


def user_activity_stats(ratings: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """Per-user rating count distribution."""
    user_counts = ratings.groupby("user_id").size().reset_index(name="n_ratings")
    if save_path:
        fig, ax = plt.subplots(figsize=(7, 4))
        cutoff = user_counts["n_ratings"].quantile(0.95)
        sns.histplot(
            user_counts[user_counts["n_ratings"] <= cutoff]["n_ratings"],
            bins=40, ax=ax, color="#2c5f8a",
        )
        ax.set_xlabel("Ratings per user")
        ax.set_ylabel("Number of users")
        ax.set_title("User Activity Distribution (95th percentile)")
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return user_counts.describe()


def genre_stats(movies: pd.DataFrame, save_path: str = None) -> pd.Series:
    """Genre frequency across the movie catalog."""
    genre_counts = movies["genre_list"].explode().value_counts()
    if save_path:
        top = genre_counts.head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        top.plot(kind="barh", ax=ax, color="#2c5f8a")
        ax.set_xlabel("Number of movies")
        ax.set_title("Top-15 Genres")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return genre_counts


def user_demographics_analysis(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    save_path: str = None,
) -> pd.DataFrame:
    """Average rating by gender and age group."""
    merged = ratings.merge(users, on="user_id", how="left")
    by_gender = merged.groupby("gender")["rating"].agg(["mean", "count"]).round(3)
    by_age = merged.groupby("age")["rating"].agg(["mean", "count"]).round(3)

    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        by_gender["mean"].plot(kind="bar", ax=axes[0], color=["#4a90d9", "#d94a4a"])
        axes[0].set_title("Mean Rating by Gender")
        axes[0].set_ylabel("Mean rating")
        axes[0].tick_params(axis="x", rotation=0)

        by_age["mean"].plot(kind="bar", ax=axes[1], color="#2c5f8a")
        axes[1].set_title("Mean Rating by Age Group")
        axes[1].set_ylabel("Mean rating")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    return pd.concat(
        {"gender": by_gender, "age": by_age},
        names=["dimension"],
    )


def run_all_eda(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Run all EDA functions and save figures. Returns summary stats."""
    os.makedirs(output_dir, exist_ok=True)
    print("  Running EDA ...")

    summary = {
        "rating_distribution": rating_distribution(
            ratings, os.path.join(output_dir, "rating_distribution.png")
        ).to_dict(),
        "user_activity": user_activity_stats(
            ratings, os.path.join(output_dir, "user_activity.png")
        ).to_dict(),
        "top_genres": genre_stats(
            movies, os.path.join(output_dir, "genre_stats.png")
        ).head(10).to_dict(),
        "demographics": user_demographics_analysis(
            ratings, users, os.path.join(output_dir, "user_demographics.png")
        ),
    }
    print(f"  EDA figures saved to {output_dir}")
    return summary


def main():
    """Standalone EDA — no model training required."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.loader import load_all
    from config import FIGURES_DIR

    ratings, movies, users = load_all()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", FIGURES_DIR)
    run_all_eda(ratings, movies, users, out_dir)


if __name__ == "__main__":
    main()
