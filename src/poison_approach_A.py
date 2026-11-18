"""
Approach A -- Random Resample
-----------------------------
Rule: pick p % of all key rows at random; for each picked key, REPLACE its mismatch count by a fresh
random integer drawn uniformly from {0, 1, ..., 8} (independent of the old value).

Effect: a poisoned key looks like a uniform draw instead of Binomial(8, 0.25). Clean data almost never
has 8 mismatches (0.25^8 ~ 0.0015 %), poisoned data has them in 1/9 of the poisoned keys, so the batch
histogram changes visibly in the tails -> easy to detect.

Run:  python src/poison_approach_A.py        -> paper/poisoned_dataset_approach_A.csv
"""
import numpy as np
from poison_common import make_rng, choose_rows, run, KEY_LENGTH

APPROACH = "A"


def poison(mb, pct):
    rng = make_rng(APPROACH, pct)
    out = mb.copy()
    idx, n_poison = choose_rows(rng, len(out), pct)
    out[idx] = rng.integers(0, KEY_LENGTH + 1, size=n_poison).astype(np.int8)
    return out


if __name__ == "__main__":
    run(APPROACH, poison)
