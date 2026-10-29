"""
Noise-only BB84 QBER dataset generator (NO Eve, NO poisoning) -- amplitude damping channel.

Same data format as generate_clean_dataset.py: 8 sifted bits per key, 500 keys per batch, the same
15 per-batch features (computed from count_0..count_8 with features_from_counts). The only change is the
physics: instead of Eve's intercept-resend, the qubit goes through an amplitude damping channel
    |1> -> |0> with probability gamma          (Kraus: K0=[[1,0],[0,sqrt(1-g)]], K1=[[0,sqrt(g)],[0,0]])
and Bob measures in Alice's basis (sifted positions).

Closed form for the measurement statistics of the 4 BB84 states (verified numerically with Kraus operators
by verify_amplitude_damping() below):
    Z basis : sent 0 -> error 0 ;  sent 1 -> error gamma         => QBER_Z = gamma/2
    X basis : sent + or - -> error (1 - sqrt(1-gamma))/2          => QBER_X = (1 - sqrt(1-gamma))/2
    total   : QBER = (QBER_Z + QBER_X)/2

Besides the 15 standard features we also record (supplementary, per batch):
    qber_z, qber_x           : error rate in the Z / X basis
    qber_z0, qber_z1         : Z-basis error rate when Alice sent 0 / 1 (flip-direction asymmetry)
    dispersion               : Var(k) / (8 q(1-q)) with q = mean QBER; equals 1 for a Binomial(8,q) distribution

mode "noise" : amplitude damping channel, no Eve        (sweep over gamma)
mode "eve"   : intercept-resend Eve, ideal channel      (reference, same as the clean class)

Usage:  python src/generate_noise_dataset.py                    (gamma = 0, 0.1, ..., 1.0)
        python src/generate_noise_dataset.py --batches 4000      (quick test)
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_clean_dataset import KEY_LENGTH, batch_counts, features_from_counts

K = 500


def ad_error_probs(g):
    """(QBER_Z, QBER_X, QBER_total) of the BB84 sifted key under amplitude damping with parameter g."""
    qz = g / 2.0
    qx = (1.0 - np.sqrt(1.0 - g)) / 2.0
    return qz, qx, (qz + qx) / 2.0


def gamma_for_qber(target):
    """Invert QBER_total(gamma) = target (monotone on [0,1) ; QBER(1)=0.5)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if ad_error_probs(mid)[2] < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def verify_amplitude_damping():
    """Independent check with density matrices: apply the Kraus channel to |0>,|1>,|+>,|-> and measure."""
    s = {"0": np.array([1, 0.]), "1": np.array([0, 1.]),
         "+": np.array([1, 1.]) / np.sqrt(2), "-": np.array([1, -1.]) / np.sqrt(2)}
    out = {}
    for g in (0.1, 0.3, 0.5, 0.8):
        K0 = np.array([[1, 0], [0, np.sqrt(1 - g)]]); K1 = np.array([[0, np.sqrt(g)], [0, 0]])
        err = {}
        for name, v in s.items():
            rho = np.outer(v, v)
            rho2 = K0 @ rho @ K0.T + K1 @ rho @ K1.T
            err[name] = 1 - float(v @ rho2 @ v)      # prob. Bob measures anything but the sent state
        qz, qx, qt = ad_error_probs(g)
        out[g] = dict(err_0=err["0"], err_1=err["1"], err_plus=err["+"], err_minus=err["-"],
                      qz_numeric=(err["0"] + err["1"]) / 2, qx_numeric=(err["+"] + err["-"]) / 2,
                      qz_formula=qz, qx_formula=qx)
    return out


def simulate_chunk(rng, nb, mode, gamma):
    shape = (nb * K, KEY_LENGTH)
    a_bits = rng.integers(0, 2, size=shape, dtype=np.int8)
    basis = rng.integers(0, 2, size=shape, dtype=np.int8)          # 0 = Z, 1 = X (Alice's = Bob's, sifted)
    if mode == "noise":
        u = rng.random(shape)
        qx = (1.0 - np.sqrt(1.0 - gamma)) / 2.0
        flip_z = (a_bits == 1) & (u < gamma)                        # |1> -> |0> only
        flip_x = u < qx                                             # symmetric in X basis
        err = np.where(basis == 0, flip_z, flip_x)
    else:                                                           # intercept-resend Eve, same code as clean set
        e_bases = rng.integers(0, 2, size=shape, dtype=np.int8)
        e_bits = np.where(e_bases == basis, a_bits, rng.integers(0, 2, size=shape, dtype=np.int8))
        b_bits = np.where(e_bases == basis, e_bits, rng.integers(0, 2, size=shape, dtype=np.int8))
        err = a_bits != b_bits
    mism = err.sum(axis=1).astype(np.int8)
    r = lambda x: x.reshape(nb, K * KEY_LENGTH)
    z = basis == 0
    ex = {
        "n_z": r(z).sum(1), "err_z": r(err & z).sum(1),
        "n_x": r(~z).sum(1), "err_x": r(err & ~z).sum(1),
        "n_z0": r(z & (a_bits == 0)).sum(1), "err_z0": r(err & z & (a_bits == 0)).sum(1),
        "n_z1": r(z & (a_bits == 1)).sum(1), "err_z1": r(err & z & (a_bits == 1)).sum(1),
    }
    return mism, ex


def batch_frame(mode, gamma, nb_total, seed, gi, chunk=4000):
    cnt, extra = [], []
    for ci, start in enumerate(range(0, nb_total, chunk)):
        nb = min(chunk, nb_total - start)
        rng = np.random.default_rng([seed, gi, ci])
        mism, ex = simulate_chunk(rng, nb, mode, gamma)
        cnt.append(batch_counts(mism, K))
        extra.append(pd.DataFrame(ex))
    counts = np.vstack(cnt); ex = pd.concat(extra, ignore_index=True)
    df = features_from_counts(counts, K)
    for k in range(KEY_LENGTH + 1):
        df.insert(k, f"count_{k}", counts[:, k])
    df["qber_z"] = ex.err_z / ex.n_z; df["qber_x"] = ex.err_x / ex.n_x
    df["qber_z0"] = ex.err_z0 / ex.n_z0; df["qber_z1"] = ex.err_z1 / ex.n_z1
    q = df.mean_qber.to_numpy()
    var_k = (df.std_qber.to_numpy() * KEY_LENGTH) ** 2               # variance of the mismatch count k
    with np.errstate(divide="ignore", invalid="ignore"):
        df["dispersion"] = np.where((q > 0) & (q < 1), var_k / (KEY_LENGTH * q * (1 - q)), np.nan)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/noise")
    ap.add_argument("--gammas", type=float, nargs="+", default=[round(0.1 * i, 1) for i in range(11)])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    feats = None
    rows = []
    jobs = [("noise", g, i) for i, g in enumerate(a.gammas)] + [("eve", np.nan, 99)]
    for mode, g, gi in jobs:
        t = time.time()
        df = batch_frame(mode, g, a.batches, a.seed, gi)
        feats = [c for c in df.columns if not c.startswith("count_")]
        row = {"mode": mode, "gamma": g, "n_batches": len(df)}
        for c in feats:
            row[f"{c}_mean"] = df[c].mean(); row[f"{c}_std"] = df[c].std()
        for k in range(KEY_LENGTH + 1):
            row[f"count_{k}_mean"] = df[f"count_{k}"].mean()
        rows.append(row)
        print(f"{mode:5s} gamma={g!s:4s} mean QBER={df.mean_qber.mean():.4f}  qber_z={df.qber_z.mean():.4f} "
              f"qber_x={df.qber_x.mean():.4f} dispersion={df.dispersion.mean():.3f} ({time.time()-t:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "noise_sweep_summary.csv"), index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
