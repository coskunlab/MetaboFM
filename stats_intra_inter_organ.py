"""
stats_intra_inter_organ.py
---------------------------
Statistical test for the intra-organ vs inter-organ cosine-distance
comparison shown in Figure 4e (draw_panel_e in plot_figure4.py).

Reuses the identical sample-pair generation procedure (seed=42, 50,000
pairs per group) so the test operates on the same distributions plotted
in the figure, then runs a two-sided Mann-Whitney U test (rank-based,
no normality assumption -- appropriate since cosine-distance
distributions are bounded and non-Gaussian) comparing intra-organ vs
inter-organ cosine distances.

Usage:
  conda run -n torch_gpu python stats_intra_inter_organ.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as _norm
from scipy.stats import mannwhitneyu

from plot_figure4 import load_umap_meta, EMB_DIR

OUT_DIR = METABOFM_ROOT / "outputs/figures/figure4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PAIRS = 50_000
SEED = 42


def main():
    sm = load_umap_meta().reset_index(drop=True)
    emb_s2 = np.load(str(EMB_DIR / "stage2_sample_cls.npy")).astype(np.float32)
    emb_normed = _norm(emb_s2, norm="l2")

    rng = np.random.default_rng(SEED)
    n = len(sm)
    organs = sm["organ"].values

    i_a = rng.integers(0, n, N_PAIRS * 4)
    i_b = rng.integers(0, n, N_PAIRS * 4)
    valid = i_a != i_b
    i_a, i_b = i_a[valid], i_b[valid]

    sims = np.einsum("ij,ij->i", emb_normed[i_a], emb_normed[i_b])
    dists = 1.0 - sims
    same = organs[i_a] == organs[i_b]

    intra = dists[same][:N_PAIRS]
    inter = dists[~same][:N_PAIRS]

    print(f"n_intra={len(intra)}  n_inter={len(inter)}")
    print(f"median intra={np.median(intra):.4f}  median inter={np.median(inter):.4f}")

    stat, p = mannwhitneyu(intra, inter, alternative="less")

    # rank-biserial correlation as an effect-size measure
    n1, n2 = len(intra), len(inter)
    r_rb = 1 - (2 * stat) / (n1 * n2)

    # normal-approximation z-score and log10(p) for cases where p underflows to 0
    from scipy.stats import norm
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (stat - mu) / sigma
    log10_p = norm.logcdf(z) / np.log(10)

    print(f"Mann-Whitney U = {stat:.6e}")
    print(f"p-value (one-sided, intra < inter) = {p:.3e}")
    print(f"z-score (normal approx.) = {z:.3f}")
    print(f"log10(p) (normal approx.) = {log10_p:.1f}")
    print(f"rank-biserial correlation r = {r_rb:.4f}")

    out = pd.DataFrame([{
        "n_intra": n1,
        "n_inter": n2,
        "median_intra": float(np.median(intra)),
        "median_inter": float(np.median(inter)),
        "mannwhitney_U": float(stat),
        "p_value_one_sided": float(p),
        "z_score": float(z),
        "log10_p_value": float(log10_p),
        "rank_biserial_r": float(r_rb),
        "seed": SEED,
    }])
    out_path = OUT_DIR / "intra_inter_organ_mannwhitney.csv"
    out.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
