"""
Approach C -- +1 Only (Availability attack)
-------------------------------------------
Rule: pick p % of all key rows at random; ADD +1 to the mismatch count of each picked key,
clipped at 8 (keys already at 8 stay at 8).

Effect: pushes the error rate up (mean QBER rises), which can make a legitimate session look too noisy
and get it aborted -- an availability attack on the key exchange.

Run:  python src/poison_approach_C.py        -> paper/poisoned_dataset_approach_C.csv
"""
import numpy as np
from poison_common import make_rng, choose_rows, run, KEY_LENGTH

APPROACH = "C"


def poison(mb, pct):
    rng = make_rng(APPROACH, pct)
    out = mb.copy()
    idx, _ = choose_rows(rng, len(out), pct)
    out[idx] = np.clip(out[idx] + 1, 0, KEY_LENGTH).astype(np.int8)
    return out


if __name__ == "__main__":
    run(APPROACH, poison)
