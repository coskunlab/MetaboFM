"""
probe_crossdataset_retrieval.py
-------------------------------
Leave-one-acquisition-out organ retrieval benchmark for sample embeddings.

For each sample, retrieve k nearest neighbours from all samples NOT from the
same MSI acquisition (dataset_id), and measure what fraction come from the
same organ. Compares Stage 2 vs Stage 1 sample embeddings.

Note: "dataset_id" here is one MSI acquisition/upload (1:1 with sample_path
in this corpus), not a publication or submitting researcher -- this
benchmark therefore excludes only the query's own file from its gallery, not
other acquisitions by the same lab. A separate, stricter benchmark that
excludes the query's entire submitting researcher (using METASPACE-derived
study metadata) is implemented in probe_leave_study_out.py and reported in
Supplementary Fig. S10d, for direct comparison.

This tests whether embeddings capture organ biology rather than
trivially retrieving from the identical source file.

Outputs (OUT_DIR)
-----------------
  crossdataset_retrieval_per_organ.csv   per-organ recall@k, both stages
  crossdataset_retrieval_overall.csv     macro + weighted mean recall@k
  crossdataset_retrieval_rk_curve.csv    R@1,5,10,20 for each stage

Usage
-----
  conda run -n torch_gpu python probe_crossdataset_retrieval.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/crossdataset_retrieval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS           = [1, 5, 10, 20]   # recall@k values to evaluate
MIN_ORGAN_N  = 10                # min samples per organ to include in per-organ stats
MIN_DATASETS = 2                 # organ must span at least this many datasets


def load_metadata():
    """Load sample embeddings and per-sample metadata."""
    emb_s2 = np.load(str(EMB_DIR / "stage2_sample_cls.npy")).astype(np.float32)

    # Stage 1: mean-pool across channels per sample
    # resnet_cls_embeddings.npy is channel-level; aggregate to sample level
    ch_meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                          usecols=["sample_path", "Organism_Part", "organism",
                                   "analyzerType", "dataset_id"])
    sm = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv")

    # stage2_sample_meta.csv already carries dataset_id (merged in from the
    # METASPACE study-metadata lookup); drop it from ch_meta before merging
    # to avoid a dataset_id_x/dataset_id_y column-name collision.
    samp = ch_meta.drop_duplicates("sample_path").reset_index(drop=True)
    if "dataset_id" in sm.columns:
        samp = samp.drop(columns=["dataset_id"])
    sm   = sm.merge(samp, on="sample_path", how="left")
    assert len(sm) == len(emb_s2), f"Shape mismatch: meta={len(sm)}, emb={len(emb_s2)}"

    # normalise organ labels (typos)
    organ_fix = {"Kideny": "Kidney", "colon": "Colon"}
    sm["organ"] = sm["Organism_Part"].replace(organ_fix)

    # Exclusion key for this benchmark is the raw dataset_id (one MSI
    # acquisition). See module docstring: a stricter, researcher-level
    # exclusion is implemented separately in probe_leave_study_out.py.
    sm["study_key"] = sm["dataset_id"]

    return sm, emb_s2


def build_mz_embeddings(sm):
    """Build bag-of-m/z binary vectors per sample (4dp rounded m/z vocabulary)."""
    print("  [m/z] building bag-of-m/z embeddings …")
    ch_meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                          usecols=["sample_path", "mz"])
    ch_meta["mz_r"] = ch_meta["mz"].round(4)

    # build vocabulary
    vocab = sorted(ch_meta["mz_r"].unique())
    mz_to_idx = {mz: i for i, mz in enumerate(vocab)}
    print(f"    vocabulary size: {len(vocab):,} unique m/z values")

    sample_to_si = {sp: i for i, sp in enumerate(sm["sample_path"])}
    n_samples = len(sm)
    emb_mz = np.zeros((n_samples, len(vocab)), dtype=np.float32)

    for sp, grp in ch_meta.groupby("sample_path"):
        si = sample_to_si.get(sp)
        if si is None:
            continue
        for mz in grp["mz_r"]:
            emb_mz[si, mz_to_idx[mz]] = 1.0

    print(f"    built {n_samples} sample m/z vectors")
    return emb_mz


def load_stage1_embeddings(sm):
    """Build stage1 sample embeddings by mean-pooling channel-level ResNet CLS."""
    print("  [Stage 1] mean-pooling ResNet CLS per sample …")
    ch_emb  = np.load(str(EMB_DIR / "resnet_cls_embeddings.npy"),
                      mmap_mode="r").astype(np.float32)
    ch_meta = pd.read_csv(EMB_DIR / "resnet_cls_meta.csv",
                          usecols=["sample_path"])

    # align to sample order in sm
    sample_to_idx = {sp: i for i, sp in enumerate(sm["sample_path"])}
    n_samples = len(sm)
    emb_dim   = ch_emb.shape[1]
    emb_s1    = np.zeros((n_samples, emb_dim), dtype=np.float32)
    counts    = np.zeros(n_samples, dtype=np.int32)

    for row_idx, sp in enumerate(ch_meta["sample_path"]):
        si = sample_to_idx.get(sp)
        if si is None:
            continue
        emb_s1[si] += ch_emb[row_idx]
        counts[si]  += 1

    valid = counts > 0
    emb_s1[valid] /= counts[valid, None]
    print(f"    {valid.sum()} / {n_samples} samples have Stage 1 embeddings")
    return emb_s1


def cross_dataset_recall(emb_normed, sm, k_max, batch_size=256):
    """
    For each sample i, retrieve k_max nearest neighbours from all samples
    NOT belonging to sample i's study (see study_key in load_metadata).
    Return recall@k for each k in KS.

    Returns
    -------
    df_results : DataFrame with columns [sample_idx, organ, dataset_id,
                                         study_key, recall@1, recall@5, ...]
    """
    n       = len(sm)
    organs  = sm["organ"].values
    ds_ids  = sm["dataset_id"].values
    study_ids = sm["study_key"].values
    unique_study = np.unique(study_ids)

    # pre-build study masks
    study_mask = {s: (study_ids == s) for s in unique_study}

    records = []
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = emb_normed[start:end]          # (B, D)

        # cosine similarity to all samples
        sims  = batch @ emb_normed.T           # (B, n)

        for bi, gi in enumerate(range(start, end)):
            q_organ = organs[gi]
            q_ds    = ds_ids[gi]
            q_study = study_ids[gi]

            # mask out same-study samples (set sim to -inf)
            same_study_mask = study_mask[q_study]
            row = sims[bi].copy()
            row[same_study_mask] = -np.inf
            row[gi] = -np.inf              # exclude self (already same study)

            nn_idx = np.argsort(-row)[:k_max]
            nn_organs = organs[nn_idx]

            rec = {}
            for k in KS:
                if k <= k_max:
                    rec[f"recall@{k}"] = float((nn_organs[:k] == q_organ).mean())

            rec["sample_idx"] = gi
            rec["organ"]      = q_organ
            rec["dataset_id"] = q_ds
            rec["study_key"]  = q_study
            records.append(rec)

        if start % 1024 == 0:
            print(f"    {start}/{n} …", end="\r")

    print(f"    {n}/{n} done    ")
    return pd.DataFrame(records)


def random_baseline_recall(sm, k):
    """Expected recall@k for a random retriever (class-frequency baseline)."""
    n = len(sm)
    organ_counts = sm["organ"].value_counts()
    # For organ with n_i samples, random R@k ≈ min(k, n_i-1) / min(k, n-1)
    # Simplified: (n_i - 1) / (n - 1) * k, capped at 1
    per_organ = {}
    for organ, ni in organ_counts.items():
        per_organ[organ] = min(k * (ni - 1) / (n - 1), 1.0)
    return per_organ


def summarise(df_res, sm, label):
    """Compute per-organ and overall recall stats."""
    organ_counts    = sm["organ"].value_counts()
    organ_datasets  = sm.groupby("organ")["dataset_id"].nunique()
    organ_studies   = sm.groupby("organ")["study_key"].nunique()

    keep = organ_counts[
        (organ_counts >= MIN_ORGAN_N) &
        (organ_studies >= MIN_DATASETS)
    ].index

    sub = df_res[df_res["organ"].isin(keep)].copy()

    per_organ = (sub.groupby("organ")[[f"recall@{k}" for k in KS]]
                    .mean()
                    .reset_index())
    per_organ["n_samples"]  = per_organ["organ"].map(organ_counts)
    per_organ["n_datasets"] = per_organ["organ"].map(organ_datasets)
    per_organ["n_studies"]  = per_organ["organ"].map(organ_studies)
    per_organ["variant"]    = label

    # add random baseline per organ
    for k in KS:
        rnd = random_baseline_recall(sm, k)
        per_organ[f"random@{k}"] = per_organ["organ"].map(rnd)

    overall = {}
    for k in KS:
        col = f"recall@{k}"
        rnd_col = f"random@{k}"
        overall[f"macro_recall@{k}"]    = per_organ[col].mean()
        overall[f"weighted_recall@{k}"] = np.average(
            per_organ[col], weights=per_organ["n_samples"])
        overall[f"macro_random@{k}"]    = per_organ[rnd_col].mean()
        overall[f"weighted_random@{k}"] = np.average(
            per_organ[rnd_col], weights=per_organ["n_samples"])
    overall["variant"] = label
    overall["n_organs"] = len(per_organ)

    return per_organ, overall


def main():
    print("[LOAD] metadata & embeddings …")
    sm, emb_s2 = load_metadata()
    print(f"  {len(sm)} samples, {sm['dataset_id'].nunique()} acquisitions, "
          f"{sm['study_key'].nunique()} studies, {sm['organ'].nunique()} organs")
    print(f"  Stage 2 emb: {emb_s2.shape}")

    emb_s1 = load_stage1_embeddings(sm)
    emb_mz = build_mz_embeddings(sm)

    # normalise
    s2_normed = normalize(emb_s2, norm="l2")
    s1_normed = normalize(emb_s1, norm="l2")
    mz_normed = normalize(emb_mz, norm="l2")

    k_max = max(KS)

    print("\n[RETRIEVAL] Stage 2 …")
    df_s2 = cross_dataset_recall(s2_normed, sm, k_max)

    print("\n[RETRIEVAL] Stage 1 …")
    df_s1 = cross_dataset_recall(s1_normed, sm, k_max)

    print("\n[RETRIEVAL] m/z bag-of-words …")
    df_mz = cross_dataset_recall(mz_normed, sm, k_max)

    # ── summarise ─────────────────────────────────────────────────────────────
    per_organ_s2, overall_s2 = summarise(df_s2, sm, "Stage 2")
    per_organ_s1, overall_s1 = summarise(df_s1, sm, "Stage 1")
    per_organ_mz, overall_mz = summarise(df_mz, sm, "m/z")

    df_per_organ = pd.concat([per_organ_s2, per_organ_s1, per_organ_mz], ignore_index=True)
    df_overall   = pd.DataFrame([overall_s2, overall_s1, overall_mz])

    # ── print results ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("OVERALL (macro / weighted) Recall@k  [random baseline in brackets]")
    print("="*60)
    for row in [overall_s2, overall_s1, overall_mz]:
        print(f"\n{row['variant']}:")
        for k in KS:
            print(f"  R@{k:2d}  macro={row[f'macro_recall@{k}']:.3f} "
                  f"[rnd={row[f'macro_random@{k}']:.3f}]  "
                  f"weighted={row[f'weighted_recall@{k}']:.3f} "
                  f"[rnd={row[f'weighted_random@{k}']:.3f}]")

    print("\n" + "="*60)
    print("PER-ORGAN Recall@10")
    print("="*60)
    pivot = df_per_organ.pivot(index="organ", columns="variant",
                               values="recall@10").reset_index()
    pivot["n_samples"]  = pivot["organ"].map(sm["organ"].value_counts())
    pivot["n_datasets"] = pivot["organ"].map(sm.groupby("organ")["dataset_id"].nunique())
    pivot["n_studies"]  = pivot["organ"].map(sm.groupby("organ")["study_key"].nunique())
    pivot = pivot.sort_values("Stage 2", ascending=False)
    # also save R@1 pivot for plotting
    pivot_r1 = df_per_organ.pivot(index="organ", columns="variant",
                                  values="recall@1").reset_index()
    pivot_r1["n_samples"]  = pivot_r1["organ"].map(sm["organ"].value_counts())
    pivot_r1["n_datasets"] = pivot_r1["organ"].map(sm.groupby("organ")["dataset_id"].nunique())
    pivot_r1["n_studies"]  = pivot_r1["organ"].map(sm.groupby("organ")["study_key"].nunique())
    # add random baseline
    n_total = len(sm)
    pivot_r1["random@1"] = pivot_r1["organ"].map(
        sm["organ"].value_counts().apply(lambda ni: (ni - 1) / (n_total - 1)))
    pivot_r1 = pivot_r1.sort_values("Stage 2", ascending=False)
    print(pivot.to_string(index=False))

    # ── save ──────────────────────────────────────────────────────────────────
    df_per_organ.to_csv(OUT_DIR / "crossdataset_retrieval_per_organ.csv", index=False)
    df_overall.to_csv(OUT_DIR / "crossdataset_retrieval_overall.csv", index=False)
    pivot.to_csv(OUT_DIR / "crossdataset_retrieval_pivot.csv", index=False)
    pivot_r1.to_csv(OUT_DIR / "crossdataset_retrieval_pivot_r1.csv", index=False)
    print(f"\n[DONE] outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
