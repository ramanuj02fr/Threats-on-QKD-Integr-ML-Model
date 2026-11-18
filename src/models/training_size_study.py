"""Evaluate four-moment models trained on 500 to 4,000 samples against a fixed test set."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from feature_ablation import (
    MODEL_FACTORIES,
    POISON_PCT,
    QBER_MOMENTS,
    evaluate_approach,
    load_features,
    write_live_results,
)
from model_common import APPROACH_NAMES, REPO, SEED


TRAIN_STEP = 500
TEST_FRACTION = 0.2
SMALL_TRAIN_SIZES = range(500, 4_001, 500)
FIXED_TEST_SIZE = 8_000
MOMENTS_CASE = (("qber_moments", QBER_MOMENTS),)
STUDY_MODELS = tuple(name for name in MODEL_FACTORIES if name != "SVM")


def training_plans(train_pool, test_pool, n_folds, *, sizes=None, fixed_test=False):
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_positions = [validation for _, validation in splitter.split(train_pool)]
    if sizes is None:
        full_train_size = 2 * len(train_pool)
        sizes = [*range(TRAIN_STEP, full_train_size, TRAIN_STEP), full_train_size]

    for size in sizes:
        n_pairs = size // 2
        # Take a growing prefix of each fixed fold so samples stay in the same fold at every size.
        pairs_per_fold, extra = divmod(n_pairs, n_folds)
        selected_folds = [positions[:pairs_per_fold + (fold < extra)]
                          for fold, positions in enumerate(fold_positions)]
        selected = np.sort(np.concatenate(selected_folds))
        folds = [
            (train_pool[np.setdiff1d(selected, validation, assume_unique=True)], train_pool[validation])
            for validation in selected_folds
        ]
        if fixed_test:
            test_subset = test_pool
        else:
            n_test_pairs = max(1, round(n_pairs * len(test_pool) / len(train_pool)))
            test_subset = test_pool[:n_test_pairs]
        yield train_pool[selected], test_subset, folds


def study_rows(data_dir, clean_ids, clean, plans):
    poisoned_datasets = {}
    for approach in APPROACH_NAMES:
        path = data_dir / f"poisoned_dataset_approach_{approach}.csv"
        ids, poisoned = load_features(path, poisoned=True)
        if not clean_ids.equals(ids):
            raise ValueError(f"Batch IDs in {path} at {POISON_PCT}% do not match the clean dataset")
        poisoned_datasets[approach] = poisoned

    for train_subset, test_subset, folds in plans:
        n_train = 2 * len(train_subset)
        n_test = 2 * len(test_subset)
        splits = [("validation", fold, fit, validation)
                  for fold, (fit, validation) in enumerate(folds, start=1)]
        splits.append(("test", 0, train_subset, test_subset))

        for approach, poisoned in poisoned_datasets.items():
            for model in STUDY_MODELS:
                print(f"{n_train} train / {n_test} test | Approach {approach} | {model}", flush=True)
                for split, fold, fit, evaluation in splits:
                    for row in evaluate_approach(
                        approach, model, clean, poisoned, fit, evaluation, cases=MOMENTS_CASE
                    ):
                        yield {"split": split, "fold": fold, "training_size": n_train,
                               "test_size": n_test, **row}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(REPO) / "paper", help="Paper dataset folder")
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds within each training subset")
    parser.add_argument(
        "--out", type=Path,
        default=Path(REPO) / "results" / "training_size" / "qber_moments_10pct_no_svm_fixed_test_8000.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    started = perf_counter()

    clean_ids, clean = load_features(args.data / "clean_batches_500keys.csv")
    train_pool, test_pool = train_test_split(np.arange(len(clean)), test_size=TEST_FRACTION, random_state=SEED)
    if 2 * len(test_pool) != FIXED_TEST_SIZE:
        parser.error(f"Expected {FIXED_TEST_SIZE:,} holdout samples, got {2 * len(test_pool):,}")
    max_folds = SMALL_TRAIN_SIZES.start // 2
    if not 2 <= args.folds <= max_folds:
        parser.error(f"--folds must be between 2 and {max_folds}")

    plans = training_plans(train_pool, test_pool, args.folds, sizes=SMALL_TRAIN_SIZES, fixed_test=True)
    results = write_live_results(args.out, study_rows(args.data, clean_ids, clean, plans), started)
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

    keys = ["training_size", "model", "approach"]
    cv_summary = cv_results.groupby(keys, sort=False).agg(
        cv_folds=("fold", "size"),
        cv_train_accuracy_mean=("train_accuracy", "mean"),
        cv_train_accuracy_std=("train_accuracy", "std"),
        cv_validation_accuracy_mean=("validation_accuracy", "mean"),
        cv_validation_accuracy_std=("validation_accuracy", "std"),
        cv_validation_roc_auc_mean=("validation_roc_auc", "mean"),
        cv_validation_roc_auc_std=("validation_roc_auc", "std"),
        cv_fit_seconds_mean=("fit_seconds", "mean"),
        cv_fit_seconds_std=("fit_seconds", "std"),
    )
    report.join(cv_summary, on=keys, validate="one_to_one").to_csv(
        args.out, index=False, float_format="%.6f"
    )
    folds_path = args.out.with_name(f"{args.out.stem}_cv_folds.csv")
    cv_results.to_csv(folds_path, index=False, float_format="%.6f")
    print(f"Training-size results saved to {args.out}")
    print(f"Per-fold results saved to {folds_path}")
    print(f"Total elapsed: {perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main()
