"""Compare QBER moments and added feature groups for all paper classifiers at 10% poisoning."""

import argparse
import csv
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
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
from model_gradient_boosting import make_model as make_gradient_boosting
from model_knn import make_model as make_knn
from model_logistic_regression import make_model as make_logistic_regression
from model_neural_network import make_model as make_neural_network
from model_random_forest import make_model as make_random_forest
from model_svm import make_model as make_svm


POISON_PCT = 10
QBER_MOMENTS = ["mean_qber", "std_qber", "skew_qber", "kurtosis_qber"]
MODEL_FACTORIES = {
    "Logistic Regression": make_logistic_regression,
    "Random Forest": make_random_forest,
    "Gradient Boosting": make_gradient_boosting,
    "SVM": make_svm,
    "KNN": make_knn,
    "Neural Network": make_neural_network,
}


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


def feature_cases(columns):
    qber_bins = [feature for feature in columns if feature.startswith("frac_qber_")]
    goodness_of_fit = ["chi2_stat", "ks_stat"]

    for feature in QBER_MOMENTS:
        yield f"single_{feature}", [feature]
    yield "qber_moments", QBER_MOMENTS
    yield "qber_moments_and_qber_bins", QBER_MOMENTS + qber_bins
    yield "qber_moments_and_goodness_of_fit", QBER_MOMENTS + goodness_of_fit
    yield "all_features", list(columns)


def paired_samples(clean, poisoned, group_indices):
    samples = pd.concat([clean.iloc[group_indices], poisoned.iloc[group_indices]], ignore_index=True)
    labels = np.repeat([0, 1], len(group_indices))
    return samples, labels


def evaluate_approach(approach, model_name, clean, poisoned, train_groups, evaluation_groups, *, cases=None):
    train, y_train = paired_samples(clean, poisoned, train_groups)
    evaluation, y_evaluation = paired_samples(clean, poisoned, evaluation_groups)
    make_model = MODEL_FACTORIES[model_name]
    if cases is None:
        cases = feature_cases(clean.columns)
    for case, features in cases:
        train_features = train[features]
        evaluation_features = evaluation[features]
        classifier = make_model()
        if model_name == "Neural Network":
            fit_sample_count = len(y_train)
            if classifier.early_stopping:
                fit_sample_count -= ceil(fit_sample_count * classifier.validation_fraction)
            classifier.set_params(batch_size=min(classifier.batch_size, fit_sample_count))
        model = make_pipeline(StandardScaler(), classifier)
        fit_started = perf_counter()
        model.fit(train_features, y_train)
        fit_seconds = perf_counter() - fit_started
        if hasattr(model, "predict_proba"):
            train_scores = model.predict_proba(train_features)[:, 1]
            evaluation_scores = model.predict_proba(evaluation_features)[:, 1]
            threshold = 0.5
        else:
            train_scores = model.decision_function(train_features)
            evaluation_scores = model.decision_function(evaluation_features)
            threshold = 0.0
        yield {
            "model": model_name,
            "approach": approach,
            "poison_pct": POISON_PCT,
            "case": case,
            "removed_features": ", ".join(clean.columns.difference(features, sort=False)),
            "kept_features": ", ".join(features),
            "n_features": len(features),
            "n_train": len(y_train),
            "n_evaluation": len(y_evaluation),
            "train_accuracy": accuracy_score(y_train, train_scores > threshold),
            "evaluation_accuracy": accuracy_score(y_evaluation, evaluation_scores > threshold),
            "evaluation_roc_auc": roc_auc_score(y_evaluation, evaluation_scores),
            "fit_seconds": fit_seconds,
        }


def study_rows(data_dir, clean_ids, clean, train_groups, test_groups, splits):
    for approach, name in APPROACH_NAMES.items():
        path = data_dir / f"poisoned_dataset_approach_{approach}.csv"
        poisoned_ids, poisoned = load_features(path, poisoned=True)
        if not clean_ids.equals(poisoned_ids):
            raise ValueError(f"Batch IDs in {path} at {POISON_PCT}% do not match the clean dataset")

        for model_name in MODEL_FACTORIES:
            for fold, (fit_groups, validation_groups) in enumerate(splits, start=1):
                print(f"Approach {approach} | {model_name} | cross-validation fold {fold}/{len(splits)}", flush=True)
                for row in evaluate_approach(approach, model_name, clean, poisoned, fit_groups, validation_groups):
                    yield {"split": "validation", "fold": fold, **row}

            print(f"Approach {approach} ({name}) | {model_name} | test holdout", flush=True)
            for row in evaluate_approach(approach, model_name, clean, poisoned, train_groups, test_groups):
                yield {"split": "test", "fold": 0, **row}


def compare_with_full_model(results, *, extra_keys=()):
    keys = ["model", "approach", "split", "fold", *extra_keys]
    scores = {"evaluation_accuracy": "reference_accuracy", "evaluation_roc_auc": "reference_auc"}
    full = results.loc[results["case"].eq("all_features"), keys + list(scores)]
    results = results.join(full.rename(columns=scores).set_index(keys), on=keys, validate="many_to_one")
    results["reference_case"] = "all_features"
    results["accuracy_drop_pp"] = 100 * (results["reference_accuracy"] - results["evaluation_accuracy"])
    results["roc_auc_drop"] = results["reference_auc"] - results["evaluation_roc_auc"]
    return results.drop(columns=["reference_accuracy", "reference_auc"])


def write_live_results(out_path, results, started):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    live_path = out_path.with_name(f"{out_path.stem}_live_{timestamp}.csv")
    rows = []
    with live_path.open("x", newline="", encoding="utf-8") as live_file:
        print(f"Live results: {live_path}", flush=True)
        writer = None
        for row in results:
            row["elapsed_seconds"] = perf_counter() - started
            if writer is None:
                writer = csv.DictWriter(live_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            live_file.flush()
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(REPO) / "paper", help="Paper dataset folder")
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds within the training split")
    parser.add_argument(
        "--out", type=Path,
        default=Path(REPO) / "results" / "ablation" / "all_models_moments_first_10pct.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    started = perf_counter()

    clean_ids, clean = load_features(args.data / "clean_batches_500keys.csv")
    # Split five-batch group indices so their clean and poisoned versions stay together.
    train_groups, test_groups = train_test_split(np.arange(len(clean)), test_size=0.2, random_state=SEED)
    if not 2 <= args.folds <= len(train_groups):
        parser.error(f"--folds must be between 2 and {len(train_groups)}")
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    splits = [(train_groups[fit], train_groups[validation]) for fit, validation in splitter.split(train_groups)]

    rows = study_rows(args.data, clean_ids, clean, train_groups, test_groups, splits)
    results = compare_with_full_model(write_live_results(args.out, rows, started))
    validation = results["split"].eq("validation")
    cv_results = results.loc[validation].drop(columns="split").rename(columns={
        "n_evaluation": "n_validation",
        "evaluation_accuracy": "validation_accuracy",
        "evaluation_roc_auc": "validation_roc_auc",
    })
    report = results.loc[~validation].drop(columns=["split", "fold"]).rename(columns={
        "n_evaluation": "n_test",
        "evaluation_accuracy": "test_accuracy",
        "evaluation_roc_auc": "test_roc_auc",
    })

    case_keys = ["model", "approach", "case"]
    cv_summary = cv_results.groupby(case_keys, sort=False).agg(
        cv_folds=("fold", "size"),
        cv_train_accuracy_mean=("train_accuracy", "mean"),
        cv_train_accuracy_std=("train_accuracy", "std"),
        cv_validation_accuracy_mean=("validation_accuracy", "mean"),
        cv_validation_accuracy_std=("validation_accuracy", "std"),
        cv_validation_roc_auc_mean=("validation_roc_auc", "mean"),
        cv_validation_roc_auc_std=("validation_roc_auc", "std"),
        cv_accuracy_drop_pp_mean=("accuracy_drop_pp", "mean"),
        cv_accuracy_drop_pp_std=("accuracy_drop_pp", "std"),
        cv_roc_auc_drop_mean=("roc_auc_drop", "mean"),
        cv_roc_auc_drop_std=("roc_auc_drop", "std"),
        cv_fit_seconds_mean=("fit_seconds", "mean"),
        cv_fit_seconds_std=("fit_seconds", "std"),
    )

    report.join(cv_summary, on=case_keys, validate="one_to_one").to_csv(
        args.out, index=False, float_format="%.6f"
    )
    folds_path = args.out.with_name(f"{args.out.stem}_cv_folds.csv")
    cv_results.to_csv(folds_path, index=False, float_format="%.6f")
    print(f"Ablation results saved to {args.out}")
    print(f"Per-fold results saved to {folds_path}")
    print(f"Total elapsed: {perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main()
