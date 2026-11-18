"""
Clean BB84 QBER dataset generator (100,000 batches).

"Clean" = unpoisoned class: BB84, 8-bit keys, Eve intercept-resend attack on every key.
Each key's mismatch count between Alice and Bob follows Binomial(8, 0.25).

Output (one CSV per batch size, default 500 keys/batch) -> paper/clean_batches_<K>keys.csv
  batch_id, label(=0, clean), keys_per_batch,
  count_0 .. count_8   : number of keys in the batch with exactly k mismatched bits (lossless summary)
  mean_qber, std_qber, skew_qber, kurtosis_qber,
  frac_qber_0.000 .. frac_qber_1.000   (fraction of keys per QBER bin; QBER = k/8)
  chi2_stat, ks_stat   : goodness-of-fit vs theoretical Binomial(8, 0.25)

The raw per-key table would be 10 million rows (100 keys) / 50 million rows (500 keys) -- far too large
for GitHub -- so it is NOT written by default. All features of the project are functions of the
count_k columns, so nothing is lost. Use --raw to also write the per-key CSV (very large!).

The default (500 keys/batch, seed 42) reproduces exactly the clean class used in the final
100k-batch sweeps (enhanced pipeline: 500 keys/batch, 15 features, aggregation window 5 applied later).
Other batch sizes can be generated with --keys-per-batch (100 reproduces the earlier baseline sweep).

Usage:  python src/generate_clean_dataset.py
        python src/generate_clean_dataset.py --batches 2000   (quick test)
"""
import argparse, os, time
from math import comb
import numpy as np
import pandas as pd

KEY_LENGTH = 8
P_ERROR = 0.25
PMF = np.array([comb(KEY_LENGTH, k) * P_ERROR ** k * (1 - P_ERROR) ** (KEY_LENGTH - k)
                for k in range(KEY_LENGTH + 1)])
CDF = np.cumsum(PMF)


def mismatches_from_rng(rng, n_keys):
    shape = (n_keys, KEY_LENGTH)
    a_bits = rng.integers(0, 2, size=shape, dtype=np.int8)
    a_bases = rng.integers(0, 2, size=shape, dtype=np.int8)
    e_bases = rng.integers(0, 2, size=shape, dtype=np.int8)
    e_bits = np.where(e_bases == a_bases, a_bits, rng.integers(0, 2, size=shape, dtype=np.int8))
    b_bits = np.where(e_bases == a_bases, e_bits, rng.integers(0, 2, size=shape, dtype=np.int8))
    return np.sum(a_bits != b_bits, axis=1).astype(np.int8)


def generate_mismatches(num_batches, keys_per_batch, seed):
    n = num_batches * keys_per_batch
    if keys_per_batch == 100:                      # baseline sweep: one single stream
        return mismatches_from_rng(np.random.default_rng(seed), n)
    out = np.empty(n, dtype=np.int8)               # enhanced sweep: chunks of 4000 batches
    chunk = 4000
    for ci, start in enumerate(range(0, num_batches, chunk)):
        nb = min(chunk, num_batches - start)
        rng = np.random.default_rng([seed, ci])
        out[start * keys_per_batch:(start + nb) * keys_per_batch] = mismatches_from_rng(rng, nb * keys_per_batch)
    return out


def batch_counts(mb, keys_per_batch):
    """(n_batches, 9) array: number of keys per batch with exactly k mismatched bits."""
    nb = len(mb) // keys_per_batch
    mbr = mb.reshape(nb, keys_per_batch)
    return np.stack([(mbr == k).sum(axis=1) for k in range(KEY_LENGTH + 1)], axis=1).astype(np.int64)


def features_from_counts(counts, keys_per_batch):
    """15 per-batch features computed from the count_k columns (works for clean AND poisoned files)."""
    counts = np.asarray(counts)
    fracs = counts / keys_per_batch
    q = np.arange(KEY_LENGTH + 1) / KEY_LENGTH                      # QBER value of each bin
    mean = fracs @ q
    var = fracs @ (q ** 2) - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    safe = np.where(std < 1e-12, 1, std)
    c = q[None, :] - mean[:, None]
    skew = np.where(std < 1e-12, 0.0, (fracs * c ** 3).sum(axis=1) / safe ** 3)
    kurt = np.where(std < 1e-12, 0.0, (fracs * c ** 4).sum(axis=1) / safe ** 4)
    chi2 = keys_per_batch * ((fracs - PMF) ** 2 / PMF).sum(axis=1)
    ks = np.abs(np.cumsum(fracs, axis=1) - CDF).max(axis=1)
    data = {"mean_qber": mean, "std_qber": std, "skew_qber": skew, "kurtosis_qber": kurt}
    for k in range(KEY_LENGTH + 1):
        data[f"frac_qber_{k / KEY_LENGTH:.3f}"] = fracs[:, k]
    data.update({"chi2_stat": chi2, "ks_stat": ks})
    return pd.DataFrame(data)


def batch_table(mb, keys_per_batch):
    counts = batch_counts(mb, keys_per_batch)
    head = {"batch_id": np.arange(len(counts)), "label": 0, "keys_per_batch": keys_per_batch}
    for k in range(KEY_LENGTH + 1):
        head[f"count_{k}"] = counts[:, k]
    return pd.concat([pd.DataFrame(head), features_from_counts(counts, keys_per_batch)], axis=1)


def write_raw(mb, keys_per_batch, path):
    n = len(mb)
    first = True
    for s in range(0, n, 1_000_000):
        blk = mb[s:s + 1_000_000]
        idx = np.arange(s, s + len(blk))
        pd.DataFrame({"batch_id": idx // keys_per_batch, "key_id": idx % keys_per_batch,
                      "mismatched_bits": blk, "qber": blk / KEY_LENGTH, "eve_present": 1}
                     ).to_csv(path, mode="w" if first else "a", header=first, index=False, float_format="%.3f")
        first = False


def main():
    ap = argparse.ArgumentParser(description="Generate the clean (unpoisoned) BB84 QBER dataset.")
    ap.add_argument("--batches", type=int, default=100_000)
    ap.add_argument("--keys-per-batch", type=int, nargs="+", default=[500])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="paper")
    ap.add_argument("--raw", action="store_true", help="also write the per-key raw CSV (HUGE)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for K in a.keys_per_batch:
        t = time.time()
        mb = generate_mismatches(a.batches, K, a.seed)
        df = batch_table(mb, K)
        path = os.path.join(a.out, f"clean_batches_{K}keys.csv")
        df.to_csv(path, index=False, float_format="%.6f")
        print(f"[{K} keys/batch] {len(df):,} batches -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t:.0f}s) | mean QBER = {df.mean_qber.mean():.4f}")
        if a.raw:
            write_raw(mb, K, os.path.join(a.out, f"clean_raw_{K}keys.csv"))
        del mb, df


if __name__ == "__main__":
    main()
