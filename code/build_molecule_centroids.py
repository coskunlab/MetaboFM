"""
build_molecule_centroids.py
---------------------------
Pre-compute per-molecule (m/z group) centroids, UMAP layout, and
per-super_class MAP@10 from Stage 2 embeddings.

Outputs (OUT_DIR)
-----------------
  molecule_centroids.csv     mz_r, n_obs, hmdb_super_class, umap_x, umap_y
  centroid_embeddings.npy    (N_groups, 512) float32 — same row order as CSV
  perclass_map10.csv         hmdb_super_class, map_at_10, n_groups

Usage
-----
  conda run -n torch_gpu python build_molecule_centroids.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# ── CONFIG ─────────────────────────────────────────────────────────────────────
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/molecule_centroids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_CSV  = EMB_DIR / "stage2_channel_meta.csv"
EMB_FILE  = EMB_DIR / "stage2_channel_refined.npy"
LABEL_CSV = METABOFM_ROOT / "outputs/benchmarks_v2/_hmdb_cache/hmdb__super_class.csv"

MIN_OBS     = 10    # minimum observations per m/z group
MZ_DECIMALS = 4
TOP_K       = 10    # MAP@10
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.1
UMAP_METRIC       = "cosine"
UMAP_SEED         = 42


# ── HELPERS ────────────────────────────────────────────────────────────────────

def majority(labels):
    vals, counts = np.unique(labels.dropna(), return_counts=True)
    if len(vals) == 0:
        return "Unknown"
    return vals[counts.argmax()]


def average_precision_at_k(relevant_mask: np.ndarray, k: int) -> float:
    """AP@K for a single query given a boolean mask of sorted neighbours."""
    relevant_mask = relevant_mask[:k]
    hits = relevant_mask.cumsum()
    positions = np.arange(1, len(relevant_mask) + 1)
    prec_at_i = hits / positions
    n_rel = relevant_mask.sum()
    if n_rel == 0:
        return 0.0
    return float((prec_at_i * relevant_mask).sum() / min(n_rel, k))


def compute_map_at_k(emb_normed: np.ndarray, labels: np.ndarray, k: int = 10,
                     batch: int = 512) -> np.ndarray:
    """Return per-query AP@K using cosine similarity (vectorised in batches)."""
    n = len(emb_normed)
    aps = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        # cosine sim (emb already normalised)
        sims = emb_normed[start:end] @ emb_normed.T   # (B, n)
        for bi, i in enumerate(range(start, end)):
            row = sims[bi].copy()
            row[i] = -np.inf                           # exclude self
            nn_idx = np.argsort(-row)[:k]
            rel = (labels[nn_idx] == labels[i])
            aps[i] = average_precision_at_k(rel, k)
    return aps


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    # ── load ──────────────────────────────────────────────────────────────────
    print("[LOAD] embeddings …")
    emb  = np.load(str(EMB_FILE), mmap_mode="r").astype(np.float32)
    meta = pd.read_csv(META_CSV)
    lbl  = pd.read_csv(LABEL_CSV)     # row_id, label
    print(f"  emb: {emb.shape}  meta: {len(meta)}  labels: {len(lbl)}")

    # align label to meta index (row_id == row index in embeddings file)
    lbl = lbl.set_index("row_id")["label"].rename("super_class")
    meta = meta.join(lbl, how="left")
    meta["mz_r"] = meta["mz"].round(MZ_DECIMALS)

    # ── build centroids ────────────────────────────────────────────────────────
    print("[CENTROIDS] grouping by m/z …")
    groups = meta.groupby("mz_r")

    centroids, mz_vals, n_obs_vals, class_vals = [], [], [], []
    for mz_r, grp in groups:
        if len(grp) < MIN_OBS:
            continue
        idxs = grp.index.to_numpy()
        vecs = emb[idxs]
        if np.isnan(vecs).any():
            continue
        centroids.append(vecs.mean(axis=0))
        mz_vals.append(mz_r)
        n_obs_vals.append(len(grp))
        class_vals.append(majority(grp["super_class"]))

    centroids = np.array(centroids, dtype=np.float32)
    print(f"  {len(centroids):,} groups (≥{MIN_OBS} obs)")

    df = pd.DataFrame({
        "mz_r":          mz_vals,
        "n_obs":         n_obs_vals,
        "hmdb_super_class": class_vals,
    })

    # ── UMAP ──────────────────────────────────────────────────────────────────
    print("[UMAP] fitting …")
    import umap
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_SEED,
        verbose=True,
    )
    coords = reducer.fit_transform(centroids)   # (N, 2)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    # ── per-class MAP@10 ───────────────────────────────────────────────────────
    print("[MAP@10] computing per-class retrieval on centroids …")
    labels_arr = np.array(class_vals)
    normed = normalize(centroids, norm="l2")
    aps = compute_map_at_k(normed, labels_arr, k=TOP_K)

    df["ap_at_10"] = aps

    unique_classes = df["hmdb_super_class"].unique()
    pc_rows = []
    for cls in unique_classes:
        mask = df["hmdb_super_class"] == cls
        pc_rows.append({
            "hmdb_super_class": cls,
            "map_at_10":  float(df.loc[mask, "ap_at_10"].mean()),
            "n_groups":   int(mask.sum()),
        })
    df_pc = pd.DataFrame(pc_rows).sort_values("n_groups", ascending=False)

    # ── save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUT_DIR / "molecule_centroids.csv", index=False)
    np.save(str(OUT_DIR / "centroid_embeddings.npy"), centroids)
    df_pc.to_csv(OUT_DIR / "perclass_map10.csv", index=False)

    print("[RESULTS] per-class MAP@10:")
    print(df_pc.to_string(index=False))
    print(f"\n[DONE] outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
