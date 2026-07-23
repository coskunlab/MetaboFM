"""
build_figure7_data.py
---------------------
Compute drug-likeness landscape data for Figure 7.

Steps
-----
1. Load Stage 2 per-m/z centroids (already built).
2. Join channels_with_candidates.csv (via stage2_channel_meta.csv manifest_row)
   to get candidate molecule names and PubChem CIDs per channel.
3. Group by m/z (4dp) → per-centroid majority molecule name, CID list, drug flag.
4. PCA (2 components) on centroid embeddings.
5. Compute drug_similarity = rank-based KNN distance to drug molecules in PCA space.
6. Identify top non-drug candidate molecules.
7. Save:
     figure7_mol_df.csv        - per-centroid row with PCA coords, drug info, family
     figure7_top_nondrug.csv   - top non-drug candidates with feature heatmap cols

Usage
-----
  conda run -n torch_gpu python build_figure7_data.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# ── CONFIG ─────────────────────────────────────────────────────────────────────
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
CENT_DIR  = METABOFM_ROOT / "outputs/molecule_centroids"
OUT_DIR   = METABOFM_ROOT / "outputs/figure7_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAND_CSV  = METABOFM_ROOT / "metaspace_images_dump/channels_with_candidates.csv"
META_CSV  = EMB_DIR / "stage2_channel_meta.csv"
LABEL_CSV = METABOFM_ROOT / "outputs/benchmarks_v2/_hmdb_cache/hmdb__super_class.csv"
DRUG_TSV  = MSI_RAW_DIR / "knowledge_bases/PubChem/Drug-Names.tsv"

MZ_DECIMALS = 4
MIN_OBS     = 10
KNN_K       = 5        # neighbours for drug similarity
TOP_NONDRUG = 20       # top non-drug candidates for heatmap


def majority(vals):
    vals = [v for v in vals if v and str(v) != "nan"]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def parse_cids(s) -> list[str]:
    if not isinstance(s, str):
        return []
    return [c.strip().lstrip("CID") for c in s.split(";") if c.strip()]


def main():
    # ── load centroid embeddings & CSV ────────────────────────────────────────
    print("[LOAD] centroids …")
    df_cent = pd.read_csv(CENT_DIR / "molecule_centroids.csv")
    emb     = np.load(str(CENT_DIR / "centroid_embeddings.npy")).astype(np.float32)
    print(f"  {len(df_cent):,} centroids, emb {emb.shape}")

    # ── load meta & candidate annotations ────────────────────────────────────
    print("[LOAD] stage2 meta & candidate annotations …")
    meta = pd.read_csv(META_CSV, usecols=["manifest_row", "mz"])
    meta["mz_r"] = meta["mz"].round(MZ_DECIMALS)

    cand = pd.read_csv(CAND_CSV, low_memory=False,
                       usecols=["dataset_id", "channel_idx", "cand_names", "cand_pubchem_cids"])
    cand["channel_idx"] = pd.to_numeric(cand["channel_idx"], errors="coerce")

    # ── load drug CIDs ────────────────────────────────────────────────────────
    print("[LOAD] drug CIDs …")
    drug_df = pd.read_csv(DRUG_TSV, sep="\t", usecols=["PubChem_CID", "Drug_name"])
    drug_df["cid_int"] = pd.to_numeric(drug_df["PubChem_CID"], errors="coerce").dropna()
    drug_df = drug_df.dropna(subset=["cid_int"])
    drug_df["cid_str"] = drug_df["cid_int"].astype(int).astype(str)
    drug_cids = set(drug_df["cid_str"])
    drug_names = dict(zip(drug_df["cid_str"], drug_df["Drug_name"]))
    print(f"  {len(drug_cids):,} drug CIDs")

    # ── load HMDB super_class ─────────────────────────────────────────────────
    lbl = pd.read_csv(LABEL_CSV).set_index("row_id")["label"]

    # ── per-m/z group annotation ──────────────────────────────────────────────
    print("[ANNOTATE] grouping by m/z …")
    # meta index == stage2 embedding row index
    meta_full = pd.read_csv(META_CSV)
    meta_full["mz_r"] = meta_full["mz"].round(MZ_DECIMALS)
    merged_full = meta_full.merge(cand, on=["dataset_id", "channel_idx"], how="left")

    mol_rows = []
    for mz_r, grp in merged_full.groupby("mz_r"):
        if len(grp) < MIN_OBS:
            continue

        # molecule name (most common first candidate)
        names_flat = []
        for s in grp["cand_names"].dropna():
            first = str(s).split(";")[0].strip()
            if first:
                names_flat.append(first)
        mol_name = Counter(names_flat).most_common(1)[0][0] if names_flat else None

        # drug flag: any CID in this group matches a known drug
        cid_flat = []
        for s in grp["cand_pubchem_cids"].dropna():
            cid_flat.extend(parse_cids(str(s)))
        matched_drugs = [drug_names[c] for c in cid_flat if c in drug_cids]
        has_drug = len(matched_drugs) > 0
        drug_label = Counter(matched_drugs).most_common(1)[0][0] if matched_drugs else None

        mol_rows.append({
            "mz_r":      mz_r,
            "mol_name":  mol_name,
            "has_drug":  has_drug,
            "drug_name": drug_label,
            "n_cids":    len(set(cid_flat)),
        })

    df_mol = pd.DataFrame(mol_rows)
    # merge into centroid df
    df_cent = df_cent.merge(df_mol, on="mz_r", how="left")
    df_cent["has_drug"] = df_cent["has_drug"].fillna(False)

    print(f"  drug-matched centroids: {df_cent['has_drug'].sum()} / {len(df_cent)}")

    # ── PCA ───────────────────────────────────────────────────────────────────
    print("[PCA] fitting 2 components …")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(normalize(emb))
    df_cent["PC1"] = coords[:, 0]
    df_cent["PC2"] = coords[:, 1]
    print(f"  explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}  "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    # ── drug similarity ───────────────────────────────────────────────────────
    print("[DRUG SIM] KNN distance to drug molecules …")
    drug_mask = df_cent["has_drug"].to_numpy()
    X = coords.copy()
    X_drug = X[drug_mask]

    k = min(KNN_K, int(drug_mask.sum()))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_drug)
    dists, _ = nbrs.kneighbors(X)
    mean_dist = dists.mean(axis=1)

    # rank-percentile score (higher = more similar to drugs)
    rank_pct = pd.Series(mean_dist).rank(method="average", pct=True, ascending=True).values
    df_cent["drug_similarity"] = 1.0 - rank_pct
    df_cent["mean_drug_dist"]  = mean_dist

    # ── top non-drug candidates ───────────────────────────────────────────────
    non_drug = df_cent[~df_cent["has_drug"]].copy()
    # score = drug_similarity (rank-based)
    non_drug = non_drug.nlargest(TOP_NONDRUG, "drug_similarity")

    # feature columns for heatmap (standardised)
    # use: drug_similarity, n_obs, umap_x, umap_y as proxy features
    feat_cols = ["drug_similarity", "mean_drug_dist", "n_obs"]
    for col in feat_cols:
        mu, sd = non_drug[col].mean(), non_drug[col].std()
        non_drug[f"{col}_z"] = (non_drug[col] - mu) / (sd if sd > 0 else 1)

    # ── save ──────────────────────────────────────────────────────────────────
    df_cent.to_csv(OUT_DIR / "figure7_mol_df.csv", index=False)
    non_drug.to_csv(OUT_DIR / "figure7_top_nondrug.csv", index=False)

    print(f"\n[DONE] outputs → {OUT_DIR}")
    print(f"  mol_df: {len(df_cent)} rows, drug={df_cent['has_drug'].sum()}, "
          f"non-drug={len(non_drug)}")
    print("\nTop non-drug candidates:")
    print(non_drug[["mol_name", "drug_similarity", "hmdb_super_class"]].to_string(index=False))


if __name__ == "__main__":
    main()
