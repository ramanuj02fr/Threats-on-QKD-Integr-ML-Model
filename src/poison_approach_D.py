"""
Approach D -- -1 Only (Stealth attack)
--------------------------------------
Rule: pick p % of all key rows at random; SUBTRACT 1 from the mismatch count of each picked key,
clipped at 0 (keys already at 0 stay at 0, so only keys with at least one mismatch actually change).

Effect: pushes the error rate down (mean QBER falls), hiding errors so an eavesdropper looks less
noticeable -- a stealth attack on the error check.

Run:  python src/poison_approach_D.py        -> paper/poisoned_dataset_approach_D.csv
"""
import numpy as np
from poison_common import make_rng, choose_rows, run, KEY_LENGTH

APPROACH = "D"


def poison(mb, pct):
    rng = make_rng(APPROACH, pct)
    out = mb.copy()
    idx, _ = choose_rows(rng, len(out), pct)
    out[idx] = np.clip(out[idx] - 1, 0, KEY_LENGTH).astype(np.int8)
    return out


if __name__ == "__main__":
    run(APPROACH, poison)
