"""
Feature-level comparison (NO machine learning):
  (1) noise-only data (amplitude damping, no Eve) vs Approach D (Eve present, counts poisoned by -1) at the SAME mean QBER
  (2) noise-only data vs intercept-resend Eve at the SAME mean QBER (needs basis/bit information)
Single-feature separability is reported as AUC of that one feature used as a score (0.5 = no separation, 1.0 = perfect).
Run after generate_noise_dataset.py:   python src/noise_vs_approachD.py
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_noise_dataset import batch_frame, gamma_for_qber, ad_error_probs, K
from generate_clean_dataset import features_from_counts, KEY_LENGTH

OUT = "results/noise"
C = [f"count_{k}" for k in range(KEY_LENGTH + 1)]
NB, SEED = 100_000, 7


def add_dispersion(df):
    q = df.mean_qber.to_numpy(); var_k = (df.std_qber.to_numpy() * KEY_LENGTH) ** 2
    df["dispersion"] = var_k / (KEY_LENGTH * q * (1 - q))
    return df


def sep_auc(a, b):
    """Separability of two samples by a single feature: max(AUC, 1-AUC)."""
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]; s = np.r_[a, b]
    auc = roc_auc_score(y, s)
    return max(auc, 1 - auc)


def window_mean(x, w=5):
    n = len(x) // w * w
    return np.asarray(x[:n]).reshape(-1, w).mean(axis=1)


# ---------------- Table A: noise sweep -------------------------------------------------
S = pd.read_csv(f"{OUT}/noise_sweep_summary.csv")
N = S[S["mode"] == "noise"].copy()
tabA = pd.DataFrame({
    "gamma": N.gamma, "QBER_Z": N.qber_z_mean, "QBER_X": N.qber_x_mean, "QBER": N.mean_qber_mean,
    "QBER_theory": [ad_error_probs(g)[2] for g in N.gamma],
    "std_qber": N.std_qber_mean, "skew_qber": N.skew_qber_mean, "kurtosis_qber": N.kurtosis_qber_mean,
    "chi2": N.chi2_stat_mean, "ks": N.ks_stat_mean, "dispersion": N.dispersion_mean})
tabA.to_csv(f"{OUT}/table_noise_features.csv", index=False, float_format="%.5f")
print("=== (A) noise sweep (means over 100k batches) ===")
print(tabA.round(4).to_string(index=False))
eve = S[S["mode"] == "eve"].iloc[0]
print("\nEve reference: QBER %.4f  std %.4f  skew %.3f  kurt %.3f  chi2 %.2f  ks %.4f  disp %.3f" % (
    eve.mean_qber_mean, eve.std_qber_mean, eve.skew_qber_mean, eve.kurtosis_qber_mean,
    eve.chi2_stat_mean, eve.ks_stat_mean, eve.dispersion_mean))

# ---------------- Table B: noise vs Approach D at matched mean QBER ----------------------
D = pd.read_csv("paper/poisoned_dataset_approach_D.csv", usecols=["poison_pct"] + C)
rows, pmf_rows = [], []
for p, g in D.groupby("poison_pct"):
    f = add_dispersion(features_from_counts(g[C].to_numpy(), K))
    gm = gamma_for_qber(f.mean_qber.mean())
    nz = batch_frame("noise", gm, NB, SEED, int(p), chunk=4000)
    pmf_d = g[C].to_numpy().mean(axis=0) / K; pmf_n = nz[C].to_numpy().mean(axis=0) / K
    r = {"poison_pct": int(p), "gamma_match": gm, "QBER_D": f.mean_qber.mean(), "QBER_noise": nz.mean_qber.mean()}
    for name in ["std_qber", "skew_qber", "kurtosis_qber", "dispersion"]:
        r[f"{name}_D"] = f[name].mean(); r[f"{name}_noise"] = nz[name].mean()
        r[f"AUC_{name}_batch"] = sep_auc(nz[name].to_numpy(), f[name].to_numpy())
        r[f"AUC_{name}_win5"] = sep_auc(window_mean(nz[name].to_numpy()), window_mean(f[name].to_numpy()))
    r["max_abs_pmf_diff"] = np.abs(pmf_d - pmf_n).max()
    rows.append(r)
    for k in range(KEY_LENGTH + 1):
        pmf_rows.append({"poison_pct": int(p), "k": k, "pmf_D": pmf_d[k], "pmf_noise": pmf_n[k]})
tabB = pd.DataFrame(rows)
tabB.to_csv(f"{OUT}/table_noise_vs_approachD.csv", index=False, float_format="%.5f")
pd.DataFrame(pmf_rows).to_csv(f"{OUT}/pmf_noise_vs_approachD.csv", index=False, float_format="%.6f")
print("\n=== (B) Approach D vs noise at the same mean QBER ===")
print(tabB[["poison_pct", "gamma_match", "QBER_D", "QBER_noise", "std_qber_D", "std_qber_noise",
            "skew_qber_D", "skew_qber_noise", "kurtosis_qber_D", "kurtosis_qber_noise",
            "dispersion_D", "dispersion_noise", "max_abs_pmf_diff"]].round(4).to_string(index=False))
print("\nSingle-feature separability AUC (per batch | 5-batch window):")
print(tabB[["poison_pct"] + [f"AUC_{n}_{w}" for n in ["std_qber", "skew_qber", "kurtosis_qber", "dispersion"]
                              for w in ["batch", "win5"]]].round(3).to_string(index=False))

# ---------------- Table C: noise vs Eve at matched total QBER -----------------------------
ge = gamma_for_qber(0.25)
nz = batch_frame("noise", ge, NB, SEED, 50)
ev = batch_frame("eve", np.nan, NB, SEED, 51)
rowsC = []
for name, a, b in [("QBER_Z", nz.qber_z, ev.qber_z), ("QBER_X", nz.qber_x, ev.qber_x),
                   ("QBER_Z0 (sent 0)", nz.qber_z0, ev.qber_z0), ("QBER_Z1 (sent 1)", nz.qber_z1, ev.qber_z1),
                   ("QBER_Z - QBER_X", nz.qber_z - nz.qber_x, ev.qber_z - ev.qber_x),
                   ("mean_qber (count based)", nz.mean_qber, ev.mean_qber),
                   ("std_qber (count based)", nz.std_qber, ev.std_qber),
                   ("dispersion (count based)", nz.dispersion, ev.dispersion)]:
    rowsC.append({"feature": name, "noise": a.mean(), "eve": b.mean(), "AUC_batch": sep_auc(a.to_numpy(), b.to_numpy())})
tabC = pd.DataFrame(rowsC)
tabC.insert(0, "gamma_match", ge)
tabC.to_csv(f"{OUT}/table_noise_vs_eve.csv", index=False, float_format="%.5f")
print(f"\n=== (C) noise (gamma={ge:.4f}, QBER={nz.mean_qber.mean():.4f}) vs Eve (QBER={ev.mean_qber.mean():.4f}) ===")
print(tabC.round(4).to_string(index=False))
