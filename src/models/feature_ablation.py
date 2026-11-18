"""Measure feature ablation at 10% poisoning with logistic regression."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_common import (
    APPROACH_NAMES,
    COUNT_COLS,
    KEYS_PER_BATCH,
    REPO,
    SEED,
    aggregate,
    features_from_counts,
)
from model_logistic_regression import make_model


POISON_PCT = 10


def load_features(path, poisoned=False):
    columns = ["batch_id", *COUNT_COLS]
    if poisoned:
        columns.append("poison_pct")
    batches = pd.read_csv(path, usecols=columns)
    if poisoned:
        batches = batches.loc[batches["poison_pct"].eq(POISON_PCT)]
    batches = batches.set_index("batch_id", verify_integrity=True).sort_index()
    features = features_from_counts(batches[COUNT_COLS].to_numpy(), KEYS_PER_BATCH)
    return batches.index, pd.DataFrame(aggregate(features.to_numpy()), columns=features.columns)


def ablation_cases(columns):
    qber_bins = [feature for feature in columns if feature.startswith("frac_qber_")]
    goodness_of_fit = ["chi2_stat", "ks_stat"]
    yield "all_features", []
    for feature in columns:
        yield f"drop_{feature}", [feature]
    yield "drop_qber_moments", ["mean_qber", "std_qber", "skew_qber", "kurtosis_qber"]
    yield "drop_qber_bins", qber_bins
    yield "drop_goodness_of_fit", goodness_of_fit
    yield "drop_qber_bins_and_goodness_of_fit", qber_bins + goodness_of_fit


def paired_samples(clean, poisoned, group_indices):
    samples = pd.concat([clean.iloc[group_indices], poisoned.iloc[group_indices]], ignore_index=True)
    labels = np.repeat([0, 1], len(group_indices))
    return samples, labels


def evaluate_approach(approach, clean, poisoned, train_groups, test_groups):
    train, y_train = paired_samples(clean, poisoned, train_groups)
    test, y_test = paired_samples(clean, poisoned, test_groups)
    rows = []
    for case, removed in ablation_cases(clean.columns):
        model = make_pipeline(StandardScaler(), make_model())
        model.fit(train.drop(columns=removed), y_train)
        scores = model.predict_proba(test.drop(columns=removed))[:, 1]
        rows.append({
            "approach": approach,
            "poison_pct": POISON_PCT,
            "case": case,
            "removed_features": ", ".join(removed),
            "n_features": clean.shape[1] - len(removed),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "test_accuracy": accuracy_score(y_test, scores > 0.5),
            "test_roc_auc": roc_auc_score(y_test, scores),
        })

    results = pd.DataFrame(rows)
    results["accuracy_drop_pp"] = 100 * (results.loc[0, "test_accuracy"] - results["test_accuracy"])
    results["roc_auc_drop"] = results.loc[0, "test_roc_auc"] - results["test_roc_auc"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(REPO) / "paper", help="Paper dataset folder")
    parser.add_argument(
        "--out", type=Path,
        default=Path(REPO) / "results" / "ablation" / "logistic_regression_10pct.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    clean_ids, clean = load_features(args.data / "clean_batches_500keys.csv")
    # Split five-batch group indices so their clean and poisoned versions stay together.
    train_groups, test_groups = train_test_split(np.arange(len(clean)), test_size=0.2, random_state=SEED)

    results = []
    for approach in APPROACH_NAMES:
        path = args.data / f"poisoned_dataset_approach_{approach}.csv"
        poisoned_ids, poisoned = load_features(path, poisoned=True)
        if not clean_ids.equals(poisoned_ids):
            raise ValueError(f"Batch IDs in {path} at {POISON_PCT}% do not match the clean dataset")

        report = evaluate_approach(approach, clean, poisoned, train_groups, test_groups)
        baseline = report.iloc[0]
        print(f"{approach} ({APPROACH_NAMES[approach]}) | full-feature test "
              f"accuracy={baseline.test_accuracy:.4f}, AUC={baseline.test_roc_auc:.4f}", flush=True)
        results.append(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(results, ignore_index=True).to_csv(args.out, index=False, float_format="%.6f")
    print(f"Ablation results saved to {args.out}")


if __name__ == "__main__":
    main()
