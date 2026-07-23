"""
compute_contiguity_baseline.py
------------------------------
Permutation null for spatial contiguity scores.

For each of N_PERMS permutations, shuffle organ labels across samples
(keeping contiguity fixed), then record per-organ mean contiguity.
This tests whether the observed per-organ contiguity distributions
are above chance level.

Output:
  spatial_patches/spatial_coherence_permuted_null.csv
    columns: perm_id, organ, mean_contiguity

Usage:
  conda run -n torch_gpu python compute_contiguity_baseline.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd

PATCH_DIR = METABOFM_ROOT / "outputs/spatial_patches"
N_PERMS   = 2000
RNG_SEED  = 42
MIN_N     = 10   # minimum samples per organ to include

def main():
    df = pd.read_csv(PATCH_DIR / "spatial_coherence_all_samples.csv")

    # Only organs with enough samples in observed data
    organ_counts = df["organ"].value_counts()
    valid_organs = organ_counts[organ_counts >= MIN_N].index.tolist()
    df = df[df["organ"].isin(valid_organs)].reset_index(drop=True)

    contiguity = df["contiguity"].values
    organs     = df["organ"].values
    rng        = np.random.default_rng(RNG_SEED)

    rows = []
    for perm_id in range(N_PERMS):
        perm_organs = rng.permutation(organs)
        for organ in valid_organs:
            mask = perm_organs == organ
            rows.append({
                "perm_id":         perm_id,
                "organ":           organ,
                "mean_contiguity": contiguity[mask].mean(),
            })

    null_df = pd.DataFrame(rows)
    out = PATCH_DIR / "spatial_coherence_permuted_null.csv"
    null_df.to_csv(out, index=False)
    print(f"Saved {len(null_df):,} rows → {out}")

    # Summary: observed vs null per organ
    obs = df.groupby("organ")["contiguity"].mean().rename("observed")
    null_mean = null_df.groupby("organ")["mean_contiguity"].mean().rename("null_mean")
    null_p95  = null_df.groupby("organ")["mean_contiguity"].quantile(0.95).rename("null_p95")
    summary = pd.concat([obs, null_mean, null_p95], axis=1).dropna()
    summary["above_null"] = summary["observed"] > summary["null_p95"]
    print(summary.to_string())


if __name__ == "__main__":
    main()
