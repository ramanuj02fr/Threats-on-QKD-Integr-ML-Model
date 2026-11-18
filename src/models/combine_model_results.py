"""Combine per-model JSON results into reproducible CSV tables and a text summary."""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
IN_DIR = os.path.join(REPO, "results", "models")
OUT_DIR = os.path.join(REPO, "results", "combined")
os.makedirs(OUT_DIR, exist_ok=True)

APPROACHES = ["A", "B", "C", "D"]
APPROACH_NAMES = {
    "A": "A (Random Resample)",
    "B": "B (Random +/-1)",
    "C": "C (+1 only, Availability)",
    "D": "D (-1 only, Stealth)",
}
ORDER = [
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
    "SVM",
    "KNN",
    "Neural Network",
]

rows = []
for path in sorted(glob.glob(os.path.join(IN_DIR, "*.json"))):
    with open(path, encoding="utf-8") as handle:
        rows.extend(
            {key: value for key, value in result.items() if key != "confusion_matrix"}
            for result in json.load(handle).values()
        )
if not rows:
    raise SystemExit(f"No result files found in {IN_DIR}. Run the model scripts first.")

df = pd.DataFrame(rows)
models = [model for model in ORDER if model in set(df.model)]
models += sorted(set(df.model) - set(models))
df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
df = df.sort_values(["approach", "poison_pct", "model"]).reset_index(drop=True)
df["approach_name"] = df["approach"].map(APPROACH_NAMES)
df.to_csv(os.path.join(OUT_DIR, "all_results_long.csv"), index=False)

for metric in ("accuracy", "auc"):
    table = df.pivot_table(
        index=["approach_name", "poison_pct"],
        columns="model",
        values=metric,
        observed=True,
    )
    table.round(4).to_csv(os.path.join(OUT_DIR, f"{metric}_table.csv"))

summary = [
    f"Models found: {', '.join(models)}",
    f"Model-runs: {len(df)}",
    f"Approaches: {', '.join(APPROACHES)}",
    f"Poisoning levels: {', '.join(str(level) for level in sorted(df.poison_pct.unique()))}%",
    "",
    "Overall metric summary:",
    df.groupby("model", observed=True)[["accuracy", "auc", "seconds"]]
    .agg(["mean", "std", "min", "max"])
    .round(4)
    .to_string(),
    "",
    "Lowest poisoning-level results:",
    df[df.poison_pct == df.poison_pct.min()]
    .sort_values(["approach", "auc"], ascending=[True, False])
    [["approach", "poison_pct", "model", "accuracy", "auc", "seconds"]]
    .round(4)
    .to_string(index=False),
]
with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(summary))

print("\n".join(summary))
print(f"\nSaved tables in: {OUT_DIR}")
