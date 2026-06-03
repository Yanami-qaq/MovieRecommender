"""Utility helpers: saving/loading results, plotting comparison charts."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def save_results(results: dict, path: str = None) -> None:
    if path is None:
        path = os.path.join(RESULTS_DIR, "eval_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


def load_results(path: str = None) -> dict:
    if path is None:
        path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_comparison(results: dict, save_path: str = None) -> plt.Figure:
    """Bar chart comparing all models across all metrics."""
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    palette = sns.color_palette("Blues_d", len(models))

    for ax, metric in zip(axes, metrics):
        values = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(models, values, color=palette)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def results_to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.round(4)
