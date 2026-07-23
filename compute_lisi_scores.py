"""
compute_lisi_scores.py
----------------------
Computes Local Inverse Simpson Index (LISI) for Stage 1 and Stage 2
UMAP embeddings across three technical covariates:
  - ionisation source
  - polarity
  - analyzer family

LISI = 1  → all k nearest neighbours share the same label (fully clustered)
LISI = C  → labels are uniformly mixed (C = number of unique labels)
Higher LISI → better mixing (desirable for technical covariates).

Outputs:
  sample_umap/lisi_scores.csv
    columns: covariate, stage, lisi_mean, lisi_median, lisi_std

Usage:
  conda run -n torch_gpu python compute_lisi_scores.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
UMAP_DIR = METABOFM_ROOT / "outputs/sample_umap"
K        = 30   # neighbourhood size


def _lisi(coords: np.ndarray, labels: np.ndarray, k: int = 30) -> np.ndarray:
    """Compute per-cell LISI scores."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]   # exclude self

    scores = np.empty(len(coords))
    for i, idx in enumerate(indices):
        neighbor_labels = labels[idx]
        _, counts = np.unique(neighbor_labels, return_counts=True)
        props = counts / k
        scores[i] = 1.0 / (props ** 2).sum()
    return scores


def _analyzer_family(s: str) -> str:
    s = str(s).strip()
    if "Orbitrap" in s or "Exploris" in s:        return "Orbitrap"
    if "FTICR" in s or "FT-ICR" in s or "FTMS" in s: return "FT-ICR"
    if "timsTOF" in s:                             return "timsTOF"
    if "Q-TOF" in s or "qTOF" in s:              return "Q-TOF"
    if "TOF" in s:                                 return "TOF"
    return "Other"


def main():
    # Load metadata
    ch = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                     usecols=["sample_path", "polarity", "analyzerType",
                               "ionisationSource"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    samp["analyzer_family"] = samp["analyzerType"].apply(_analyzer_family)
    samp["ionisation_clean"] = samp["ionisationSource"].str.strip().fillna("Unknown")
    samp["polarity_clean"]   = samp["polarity"].str.strip().fillna("Unknown")

    # NOTE: "Study / dataset" was considered as a 4th covariate here, but
    # every one of the 5,600 samples has a unique dataset_id in this corpus
    # (one sample = one dataset), so LISI against it is trivially maximal
    # (~k) by construction regardless of embedding quality -- not a
    # meaningful mixing check. Left out for that reason.
    covariates = {
        "Ionisation source": "ionisation_clean",
        "Polarity":          "polarity_clean",
        "Analyzer family":   "analyzer_family",
    }

    umap_s2 = np.load(str(UMAP_DIR / "umap2d_stage2.npy"))
    umap_s1 = np.load(str(UMAP_DIR / "umap2d_stage1_meanpool.npy"))
    assert len(umap_s2) == len(samp), "UMAP row count mismatch"

    rows = []
    for cov_name, col in covariates.items():
        labels = samp[col].fillna("Unknown").values
        for stage, coords in [("Stage 2", umap_s2), ("Stage 1", umap_s1)]:
            print(f"  Computing LISI: {cov_name} / {stage} …")
            scores = _lisi(coords, labels, k=K)
            rows.append({
                "covariate":   cov_name,
                "stage":       stage,
                "lisi_mean":   scores.mean(),
                "lisi_median": np.median(scores),
                "lisi_std":    scores.std(),
                "n_labels":    len(np.unique(labels)),
            })
            print(f"    mean LISI={scores.mean():.3f}  (max possible={len(np.unique(labels)):.0f})")

    out_df = pd.DataFrame(rows)
    out    = UMAP_DIR / "lisi_scores.csv"
    out_df.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
