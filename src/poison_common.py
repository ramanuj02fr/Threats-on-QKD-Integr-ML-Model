"""
Shared driver for the four poisoning-approach scripts (poison_approach_A.py ... poison_approach_D.py).

Common poisoning setup (identical for all approaches)
-----------------------------------------------------
* Start from the clean per-key data: 100,000 batches x 500 keys, 8-bit keys, Eve intercept-resend
  (the same data as paper/clean_batches_500keys.csv, regenerated from seed 42).
* "p % poisoning" means: p % of ALL key rows (uniformly at random, without replacement, across all
  batches) get their mismatch count modified according to the approach rule.
* Poisoned batch i is the modified copy of clean batch i, so batch_id matches the clean file.
* Levels: 10, 20, 30, 40, 50 %.  Random stream per (approach, level): default_rng([42, ord(approach), p]).

Output: paper/poisoned_dataset_approach_<X>.csv  (poisoned class only, all five levels stacked)
  batch_id, label (=1, poisoned), poison_pct, keys_per_batch, count_0 .. count_8
  count_k = number of keys in the batch with exactly k mismatched bits.
All features of the project are functions of count_k:
  from generate_clean_dataset import features_from_counts
  feats = features_from_counts(df[[f"count_{k}" for k in range(9)]].to_numpy(), 500)
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_clean_dataset import generate_mismatches, batch_counts, KEY_LENGTH  # noqa: E402

SEED = 42
POISON_PCTS = [10, 20, 30, 40, 50]


def make_rng(approach, pct):
    return np.random.default_rng([SEED, ord(approach), pct])


def choose_rows(rng, n_rows, pct):
    """Indices of the p % key rows to poison (uniform, without replacement)."""
    n_poison = int(round(n_rows * pct / 100))
    return rng.choice(n_rows, size=n_poison, replace=False), n_poison


def run(approach, poison_fn):
    ap = argparse.ArgumentParser(description=f"Build the poisoned dataset for Approach {approach}.")
    ap.add_argument("--batches", type=int, default=100_000)
    ap.add_argument("--keys-per-batch", type=int, default=500)
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    clean = generate_mismatches(a.batches, a.keys_per_batch, SEED)
    parts = []
    for pct in POISON_PCTS:
        t = time.time()
        poisoned = poison_fn(clean, pct)
        counts = batch_counts(poisoned, a.keys_per_batch)
        df = pd.DataFrame({"batch_id": np.arange(len(counts)), "label": 1, "poison_pct": pct,
                           "keys_per_batch": a.keys_per_batch})
        for k in range(KEY_LENGTH + 1):
            df[f"count_{k}"] = counts[:, k]
        parts.append(df)
        changed = int((poisoned != clean).sum())
        print(f"  Approach {approach} @ {pct}%: {changed:,} of {len(clean):,} keys actually changed "
              f"({time.time() - t:.0f}s)")
    out = pd.concat(parts, ignore_index=True)
    path = os.path.join(a.out, f"poisoned_dataset_approach_{approach}.csv")
    out.to_csv(path, index=False)
    print(f"Approach {approach}: {len(out):,} rows -> {path} ({os.path.getsize(path) / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s total)")
