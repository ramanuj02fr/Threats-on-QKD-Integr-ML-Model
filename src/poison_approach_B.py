"""
Approach B -- Random +/-1 Shift (hardest to detect)
---------------------------------------------------
Rule: pick p % of all key rows at random; for each picked key, ADD +1 or -1 (each with probability 1/2)
to its mismatch count, then clip to the valid range [0, 8].

Effect: the shift is symmetric, so the batch mean QBER barely moves; only the spread (variance) of the
histogram changes slightly. This is the stealthiest approach and the hardest one to detect, especially
at low poisoning percentages.

Run:  python src/poison_approach_B.py        -> paper/poisoned_dataset_approach_B.csv
"""
import numpy as np
from poison_common import make_rng, choose_rows, run, KEY_LENGTH

APPROACH = "B"


def poison(mb, pct):
    rng = make_rng(APPROACH, pct)
    out = mb.copy()
    idx, n_poison = choose_rows(rng, len(out), pct)
    direction = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_poison)
    out[idx] = np.clip(out[idx] + direction, 0, KEY_LENGTH).astype(np.int8)
    return out


if __name__ == "__main__":
    run(APPROACH, poison)
