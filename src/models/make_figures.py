"""Create publication-ready figures from the saved model results.

Run:
    python src/models/make_figures.py

Every plotted value is read from results/models/*.json. The script creates
PNG and vector PDF copies in results/figures/.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODEL_DIR = os.path.join(REPO, "results", "models")
FIG_DIR = os.path.join(REPO, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

APPROACHES = ["A", "B", "C", "D"]
APPROACH_NAMES = {
    "A": "A: Random resample",
    "B": "B: Random +/-1",
    "C": "C: Availability (+1)",
    "D": "D: Stealth (-1)",
}
MODEL_ORDER = [
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
    "SVM",
    "KNN",
    "Neural Network",
]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            model_records = json.load(handle)
        rows.extend(
            {key: value for key, value in result.items() if key != "confusion_matrix"}
            for result in model_records.values()
        )
    if not rows:
        raise SystemExit(f"No result files found in {MODEL_DIR}. Run the model scripts first.")
    frame = pd.DataFrame(rows)
    models = [model for model in MODEL_ORDER if model in set(frame.model)]
    models += sorted(set(frame.model) - set(models))
    frame["model"] = pd.Categorical(frame["model"], categories=models, ordered=True)
    return frame.sort_values(["approach", "poison_pct", "model"]), models


def save(fig, number, title):
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.set_layout_engine("constrained")
    stem = os.path.join(FIG_DIR, f"fig_{number:02d}_{title.lower().replace(' ', '_')[:45]}")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def metric_lines(df, metric, ylabel, number):
    levels = sorted(df.poison_pct.unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, approach in zip(axes.flat, APPROACHES):
        subset = df[df.approach == approach]
        for color, model in zip(COLORS, df.model.cat.categories):
            values = subset[subset.model == model].set_index("poison_pct")[metric]
            ax.plot(levels, values.reindex(levels), marker="o", linewidth=2, color=color, label=model)
        ax.set_title(APPROACH_NAMES[approach])
        ax.set_xticks(levels)
        ax.set_xlabel("Poisoning level (%)")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[0, 1].legend(fontsize=8, loc="best")
    save(fig, number, f"{ylabel} by poisoning level")


def heatmaps(df, metric, number):
    levels = sorted(df.poison_pct.unique())
    models = list(df.model.cat.categories)
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    norm = Normalize(vmin=max(0.0, df[metric].min() - 0.02), vmax=1.0)
    image = None
    for ax, level in zip(axes, levels):
        matrix = df[df.poison_pct == level].pivot(index="model", columns="approach", values=metric)
        matrix = matrix.reindex(index=models, columns=APPROACHES)
        image = ax.imshow(matrix.to_numpy(), cmap="viridis", norm=norm, aspect="auto")
        ax.set_title(f"{level}%")
        ax.set_xticks(range(4), APPROACHES)
        ax.set_xlabel("Approach")
        ax.set_yticks(range(len(models)), models)
        for row in range(len(models)):
            for col in range(4):
                value = matrix.iloc[row, col]
                if pd.notna(value):
                    ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=axes, label=metric.upper(), shrink=0.85)
    save(fig, number, f"{metric.upper()} heatmaps")


def main():
    df, models = load_results()
    metric_lines(df, "accuracy", "Accuracy", 1)
    metric_lines(df, "auc", "AUC", 2)
    heatmaps(df, "accuracy", 3)
    heatmaps(df, "auc", 4)

    lowest = int(df.poison_pct.min())
    for metric, number in (("accuracy", 5), ("auc", 6)):
        subset = df[df.poison_pct == lowest]
        pivot = subset.pivot(index="approach", columns="model", values=metric).reindex(index=APPROACHES, columns=models)
        fig, ax = plt.subplots(figsize=(13, 5.5))
        pivot.plot.bar(ax=ax, width=0.82, color=COLORS[: len(models)])
        ax.set_xlabel("Poisoning approach")
        ax.set_ylabel(metric.upper())
        ax.set_xticklabels([APPROACH_NAMES[a] for a in APPROACHES], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Model", ncol=3)
        save(fig, number, f"{metric.upper()} comparison at lowest level")

    for metric, number in (("accuracy", 7), ("auc", 8)):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        data = [df[df.model == model][metric].dropna() for model in models]
        ax.boxplot(data, tick_labels=models, patch_artist=True)
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        save(fig, number, f"{metric.upper()} distribution across experiments")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, model in zip(COLORS, models):
        subset = df[df.model == model].groupby("poison_pct", observed=True).accuracy.mean()
        ax.plot(subset.index, 1 - subset, marker="o", linewidth=2, color=color, label=model)
    ax.set_xlabel("Poisoning level (%)")
    ax.set_ylabel("Mean classification error (1 - accuracy)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    save(fig, 9, "Mean classification error")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, model in zip(COLORS, models):
        subset = df[df.model == model].groupby("poison_pct", observed=True).seconds.mean()
        ax.plot(subset.index, subset, marker="o", linewidth=2, color=color, label=model)
    ax.set_xlabel("Poisoning level (%)")
    ax.set_ylabel("Mean training time (seconds)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    save(fig, 10, "Training time by poisoning level")

    hardest_key = f"B|{lowest}"
    fig, ax = plt.subplots(figsize=(8, 6))
    for color, model in zip(COLORS, models):
        path = os.path.join(MODEL_DIR, model.lower().replace(" ", "_") + ".json")
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle).get(hardest_key)
        if result and result.get("roc_fpr") and result.get("roc_tpr"):
            ax.plot(result["roc_fpr"], result["roc_tpr"], linewidth=2, color=color,
                    label=f"{model} (AUC={result['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, 11, f"ROC curves at approach B and {lowest}%")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, model in zip(axes.flat, models):
        path = os.path.join(MODEL_DIR, model.lower().replace(" ", "_") + ".json")
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle).get(hardest_key)
        matrix = np.asarray(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(model)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["Clean", "Poisoned"])
        ax.set_yticks([0, 1], ["Clean", "Poisoned"])
        for row in range(2):
            for col in range(2):
                ax.text(col, row, int(matrix[row, col]), ha="center", va="center")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    save(fig, 12, f"Confusion matrices at approach B and {lowest}%")

    print(f"12 figures saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
