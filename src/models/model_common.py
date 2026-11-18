"""
Shared code for the model scripts in this folder.

Pipeline (same as the final "enhanced" experiment):
  1. Read the datasets from  paper/  (clean file + poisoned_dataset_approach_<A-D>.csv).
  2. Turn the per-batch key counts into 15 features (mean, std, skew, kurtosis, 9 QBER-bin fractions,
     chi-square, KS) with features_from_counts().
  3. Aggregate: every 5 consecutive batches of the same class -> 1 row (mean of the features).
  4. Binary task: clean (label 0) vs poisoned at p % (label 1), for p = 10..50 and approaches A..D.
  5. Stratified split (random_state=42), StandardScaler fitted on the training part,
     train the model, and report accuracy, AUC, confusion matrix, and ROC points.

Results are saved after every combination to  results/models/<model>.json  and  <model>.csv,
so an interrupted run resumes where it stopped.

Run a single model, e.g.:
    python src/models/model_random_forest.py
    python src/models/model_random_forest.py --approaches B --levels 10     (quick test)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from generate_clean_dataset import features_from_counts  # noqa: E402

KEYS_PER_BATCH = 500
WINDOW = 5
SEED = 42
APPROACH_NAMES = {"A": "Random Resample", "B": "Random +/-1", "C": "+1 only (availability)", "D": "-1 only (stealth)"}
COUNT_COLS = [f"count_{k}" for k in range(9)]


def aggregate(X, window=WINDOW):
    """Mean of every `window` consecutive batches -> one row."""
    n = len(X) // window
    return X[:n * window].reshape(n, window, X.shape[1]).mean(axis=1)


def load_clean(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "clean_batches_500keys.csv"), usecols=COUNT_COLS)
    return aggregate(features_from_counts(df.to_numpy(), KEYS_PER_BATCH).to_numpy())


def load_poisoned(data_dir, approach):
    """dict: poison_pct -> aggregated feature matrix of the poisoned class."""
    df = pd.read_csv(os.path.join(data_dir, f"poisoned_dataset_approach_{approach}.csv"),
                     usecols=["poison_pct"] + COUNT_COLS)
    out = {}
    for pct, g in df.groupby("poison_pct"):
        out[int(pct)] = aggregate(features_from_counts(g[COUNT_COLS].to_numpy(), KEYS_PER_BATCH).to_numpy())
    return out


def make_split(X_clean, X_poisoned):
    X = np.vstack([X_clean, X_poisoned])
    y = np.r_[np.zeros(len(X_clean)), np.ones(len(X_poisoned))].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def run(model_name, make_model):
    ap = argparse.ArgumentParser(description=f"{model_name}: clean vs poisoned detection (all approaches/levels).")
    ap.add_argument("--approaches", nargs="+", default=["A", "B", "C", "D"], choices=list("ABCD"))
    ap.add_argument("--levels", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    ap.add_argument("--data", default=os.path.join(REPO, "paper"), help="folder with the paper CSV files")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "models"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stem = model_name.lower().replace(" ", "_")
    json_path = os.path.join(a.out, f"{stem}.json")
    results = json.load(open(json_path)) if os.path.exists(json_path) else {}

    print(f"=== {model_name} | approaches {','.join(a.approaches)} | levels {a.levels} | resume: {len(results)} done")
    t0 = time.time()
    clean = load_clean(a.data)
    for approach in a.approaches:
        todo = [
            p for p in a.levels
            if f"{approach}|{p}" not in results
            or "roc_fpr" not in results[f"{approach}|{p}"]
            or "roc_tpr" not in results[f"{approach}|{p}"]
        ]
        if not todo:
            continue
        poisoned = load_poisoned(a.data, approach)
        for pct in todo:
            Xtr, Xte, ytr, yte = make_split(clean, poisoned[pct])
            t = time.time()
            clf = make_model().fit(Xtr, ytr)
            if hasattr(clf, "predict_proba"):
                score, thr = clf.predict_proba(Xte)[:, 1], 0.5
            else:
                score, thr = clf.decision_function(Xte), 0.0
            pred = (score > thr).astype(int)
            acc, auc = accuracy_score(yte, pred), roc_auc_score(yte, score)
            fpr, tpr, _ = roc_curve(yte, score)
            results[f"{approach}|{pct}"] = {
                "model": model_name, "approach": approach, "poison_pct": pct,
                "accuracy": float(acc), "auc": float(auc), "seconds": float(time.time() - t),
                "confusion_matrix": confusion_matrix(yte, pred).tolist(),
                "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist()}
            print(f"  Approach {approach} @ {pct:>2}%  acc={acc:.4f}  auc={auc:.4f}  ({time.time() - t:.1f}s)", flush=True)
            with open(json_path, "w") as f:
                json.dump(results, f)
            pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results.values()]
                         ).sort_values(["approach", "poison_pct"]).to_csv(os.path.join(a.out, f"{stem}.csv"), index=False)
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {json_path}")
