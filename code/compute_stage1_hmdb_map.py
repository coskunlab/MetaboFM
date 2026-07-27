"""
compute_stage1_hmdb_map.py
--------------------------
Computes per-HMDB-super-class mAP@10 for Stage 1 (ResNet CLS) embeddings,
mirroring the existing Stage 2 computation in molecule_centroids/perclass_map10.csv.

Method:
  1. Load Stage 1 channel embeddings (resnet_cls_embeddings.npy, N×256).
  2. Round each channel's m/z to 4 d.p. → mz_r key.
  3. Compute per-mz_r centroid = mean embedding over all channels with that m/z.
  4. Join with molecule_centroids.csv to get hmdb_super_class.
  5. For each centroid, compute AP@10 in cosine-similarity space.
  6. Group by hmdb_super_class → mAP@10, n_groups.

Output:
  molecule_centroids/perclass_map10_stage1.csv
    columns: hmdb_super_class, map_at_10, n_groups

Usage:
  conda run -n torch_gpu python compute_stage1_hmdb_map.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
CENT_DIR = METABOFM_ROOT / "outputs/molecule_centroids"
MZ_ROUND = 4
TOP_K    = 10
CHUNK    = 512   # cosine sim in chunks to avoid OOM


def _ap_at_k(sim_row: np.ndarray, same_class: np.ndarray, k: int, self_idx: int) -> float:
    """Average Precision@k for one query, excluding the query's own index.

    NOTE: an earlier version excluded the constant index 0 instead of the
    per-query self_idx, so every query except index 0 kept its own (trivial,
    always-rank-1, always-same-class) self-match in the top-k — this silently
    inflated AP@10 across the board and made singleton classes (n=1) always
    score exactly 1.0. Fixed to match the self-exclusion already used
    correctly in build_molecule_centroids.py (`row[i] = -np.inf`).
    """
    order   = np.argsort(-sim_row)
    order   = order[order != self_idx][:k]   # exclude self
    hits    = same_class[order]
    n_hits  = hits.sum()
    if n_hits == 0:
        return 0.0
    precisions = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precisions * hits).sum() / n_hits)


def main():
    print("Loading Stage 1 embeddings …")
    emb  = np.load(str(EMB_DIR / "resnet_cls_embeddings.npy")).astype(np.float32)
    meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                       usecols=["mz"])
    assert len(emb) == len(meta), "embedding / meta length mismatch"

    meta["mz_r"] = meta["mz"].round(MZ_ROUND)

    print("Computing per-mz_r centroids …")
    mz_vals  = meta["mz_r"].values
    unique_mz = np.unique(mz_vals)
    centroids = np.zeros((len(unique_mz), emb.shape[1]), dtype=np.float32)
    for i, mz in enumerate(unique_mz):
        mask = mz_vals == mz
        centroids[i] = emb[mask].mean(axis=0)
    centroids = normalize(centroids, norm="l2")

    mz_df   = pd.DataFrame({"mz_r": unique_mz})
    ref     = pd.read_csv(CENT_DIR / "molecule_centroids.csv",
                          usecols=["mz_r", "hmdb_super_class", "n_obs"])
    mz_df   = mz_df.merge(ref, on="mz_r", how="left")
    valid   = mz_df["hmdb_super_class"].notna().values
    print(f"  {valid.sum()} / {len(mz_df)} centroids have HMDB annotation")

    centroids_v = centroids[valid]
    classes_v   = mz_df.loc[valid, "hmdb_super_class"].values
    n           = len(centroids_v)

    print(f"Computing pairwise AP@{TOP_K} for {n} annotated centroids …")
    aps = np.zeros(n, dtype=np.float32)
    for start in range(0, n, CHUNK):
        end  = min(start + CHUNK, n)
        sim  = centroids_v[start:end] @ centroids_v.T   # (chunk, n)
        for local_i, global_i in enumerate(range(start, end)):
            same = classes_v == classes_v[global_i]
            aps[global_i] = _ap_at_k(sim[local_i], same, TOP_K, self_idx=global_i)
        if (start // CHUNK) % 10 == 0:
            print(f"  {end}/{n}")

    result_df = pd.DataFrame({
        "mz_r":            mz_df.loc[valid, "mz_r"].values,
        "hmdb_super_class": classes_v,
        "ap_at_10":        aps,
    })
    out = result_df.groupby("hmdb_super_class").agg(
        map_at_10=("ap_at_10", "mean"),
        n_groups=("ap_at_10", "count"),
    ).reset_index()
    out = out.sort_values("map_at_10", ascending=False)

    save_path = CENT_DIR / "perclass_map10_stage1.csv"
    out.to_csv(save_path, index=False)
    print(f"\nSaved → {save_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
