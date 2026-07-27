"""
compute_lisi_scores_raw.py
---------------------------
Computes Local Inverse Simpson Index (LISI) for Stage 1 (mean-pool) and
Stage 2 sample embeddings directly in their native raw embedding space
(256-dim and 512-dim respectively), rather than on the 2D UMAP
projection used in compute_lisi_scores.py. This avoids the confound of
UMAP's own distortion of local neighborhood structure, and evaluates
covariate mixing in the representation space that is actually used for
downstream classification and retrieval.

Covariates: ionisation source, polarity, analyzer family.

Outputs:
  sample_umap/lisi_scores_raw.csv
    columns: covariate, stage, lisi_mean, lisi_median, lisi_std, n_labels

Usage:
  conda run -n torch_gpu python compute_lisi_scores_raw.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as _norm

from probe_sample_umap import stage1_mean_pool, S2_CLS, S2_META, CH_META

UMAP_DIR = METABOFM_ROOT / "outputs/sample_umap"
K = 30


def _lisi(coords: np.ndarray, labels: np.ndarray, k: int = 30) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="cosine").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]

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
    ch = pd.read_csv(CH_META, usecols=["sample_path", "polarity", "analyzerType",
                                        "ionisationSource"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    samp["analyzer_family"] = samp["analyzerType"].apply(_analyzer_family)
    samp["ionisation_clean"] = samp["ionisationSource"].str.strip().fillna("Unknown")
    samp["polarity_clean"]   = samp["polarity"].str.strip().fillna("Unknown")

    covariates = {
        "Ionisation source": "ionisation_clean",
        "Polarity":          "polarity_clean",
        "Analyzer family":   "analyzer_family",
    }

    s2_meta = pd.read_csv(S2_META)
    sample_paths = s2_meta["sample_path"].tolist()

    emb_s2 = np.load(str(S2_CLS)).astype(np.float32)
    emb_s1 = stage1_mean_pool(sample_paths)

    assert len(emb_s2) == len(samp), "Stage 2 row count mismatch"
    assert len(emb_s1) == len(samp), "Stage 1 mean-pool row count mismatch"

    emb_s2 = _norm(emb_s2, norm="l2")
    emb_s1 = _norm(emb_s1, norm="l2")

    rows = []
    for cov_name, col in covariates.items():
        labels = samp[col].fillna("Unknown").values
        for stage, emb in [("Stage 2", emb_s2), ("Stage 1", emb_s1)]:
            print(f"  Computing raw-space LISI: {cov_name} / {stage} …")
            scores = _lisi(emb, labels, k=K)
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
    out    = UMAP_DIR / "lisi_scores_raw.csv"
    out_df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
