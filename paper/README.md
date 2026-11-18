# paper/ - datasets used in the paper

All data is generated deterministically (seed 42): 100,000 batches per class, 500 keys per batch, 8-bit keys,
BB84 with an Eve intercept-resend attack (per-key mismatch count ~ Binomial(8, 0.25) for the clean class).

| File | Content | Rows |
|---|---|---|
| `clean_batches_500keys.csv` | Clean (unpoisoned) class, `label = 0`. Counts + 15 per-batch features. | 100,000 |
| `poisoned_dataset_approach_A.csv` | Approach A - Random Resample, poisoned class, `label = 1`, levels 10-50 % | 500,000 |
| `poisoned_dataset_approach_B.csv` | Approach B - Random +/-1 shift | 500,000 |
| `poisoned_dataset_approach_C.csv` | Approach C - +1 only (availability) | 500,000 |
| `poisoned_dataset_approach_D.csv` | Approach D - -1 only (stealth) | 500,000 |

## Columns
- Clean file: `batch_id, label, keys_per_batch, count_0..count_8, mean_qber, std_qber, skew_qber, kurtosis_qber, frac_qber_0.000..frac_qber_1.000, chi2_stat, ks_stat`
- Poisoned files: `batch_id, label, poison_pct, keys_per_batch, count_0..count_8` (the five poisoning levels are stacked; filter on `poison_pct`).
  Poisoned batch `i` is the modified copy of clean batch `i`.
- `count_k` = number of keys in the batch with exactly k mismatched bits. Every feature is a function of these counts
  (`features_from_counts` in `src/generate_clean_dataset.py` computes the 15 features for clean or poisoned counts).

## Regenerate
```
python src/generate_clean_dataset.py     # paper/clean_batches_500keys.csv
python src/poison_approach_A.py          # paper/poisoned_dataset_approach_A.csv  (likewise B, C, D)
```
Each `src/poison_approach_*.py` file documents the poisoning rule of its approach; shared setup is in `src/poison_common.py`.

## Verification
Features computed from these CSVs, aggregated over 5 batches and classified with Logistic Regression reproduce the
reported sweep numbers exactly (e.g. Approach B @ 10 %: AUC 0.9155, accuracy 0.8394).
