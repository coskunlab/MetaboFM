"""
probe_leave_study_out.py
-------------------------
Strict leave-one-study-out organ retrieval benchmark for sample embeddings,
directly responding to the reviewer request for "strict leave-study-out,
leave-laboratory-out, or leave-platform-out validation."

For each sample, retrieve k nearest neighbours from all samples NOT
submitted by the same real-world researcher/lab, and measure what fraction
come from the same organ. Compares Stage 2, Stage 1, and an m/z-only
baseline.

Study identity is the METASPACE submitting researcher (metaspace_submitter),
obtained by querying the public METASPACE GraphQL API for every dataset_id
in the corpus (see outputs/crossdataset_retrieval/metaspace_study_metadata_
matched.csv) and merged into stage2_sample_meta.csv. Submitter is used
rather than the coarser submitting institution (metaspace_group) because
institution-level grouping conflates unrelated independent researchers --
e.g. the "NIH KPMP" consortium alone spans 2 institutions and 10 different
submitters. Samples with no resolvable METASPACE submitter (missing/errored
API lookup, ~1.7% of the corpus) fall back to excluding only their own
dataset_id, since no stronger grouping is available for them.

This is a stricter, complementary benchmark to probe_crossdataset_retrieval.py
(leave-one-acquisition-out), which excludes only the query's own MSI file and
therefore does not test whether embeddings encode lab/researcher-specific
acquisition-batch signatures rather than organ biology.

Outputs (OUT_DIR)
-----------------
  leavestudyout_per_organ.csv   per-organ recall@k, all variants
  leavestudyout_overall.csv     macro + weighted mean recall@k
  leavestudyout_pivot_r1.csv    per-organ Recall@1 pivot (for plotting)

Usage
-----
  conda run -n torch_gpu python probe_leave_study_out.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/leave_study_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS           = [1, 5, 10, 20]   # recall@k values to evaluate
MIN_ORGAN_N  = 10                # min samples per organ to include in per-organ stats
MIN_STUDIES  = 2                 # organ must span at least this many studies


def load_metadata():
    """Load sample embeddings and per-sample metadata, including METASPACE
    submitter identity for strict study-level exclusion."""
    emb_s2 = np.load(str(EMB_DIR / "stage2_sample_cls.npy")).astype(np.float32)

    ch_meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                          usecols=["sample_path", "Organism_Part", "organism",
                                   "analyzerType", "dataset_id"])
    sm = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv")

    samp = ch_meta.drop_duplicates("sample_path").reset_index(drop=True)
    if "dataset_id" in sm.columns:
        samp = samp.drop(columns=["dataset_id"])
    sm = sm.merge(samp, on="sample_path", how="left")
    assert len(sm) == len(emb_s2), f"Shape mismatch: meta={len(sm)}, emb={len(emb_s2)}"

    organ_fix = {"Kideny": "Kidney", "colon": "Colon"}
    sm["organ"] = sm["Organism_Part"].replace(organ_fix)

    if "metaspace_submitter" not in sm.columns:
        raise RuntimeError(
            "metaspace_submitter column not found in stage2_sample_meta.csv -- "
            "run the METASPACE study-metadata merge before this script."
        )
    n_missing = sm["metaspace_submitter"].isna().sum()
    sm["study_key"] = sm["metaspace_submitter"].fillna(sm["dataset_id"])
    print(f"  study_key: {sm['study_key'].nunique()} unique studies "
          f"({n_missing} samples fell back to per-acquisition dataset_id)")

    return sm, emb_s2


def build_mz_embeddings(sm):
    print("  [m/z] building bag-of-m/z embeddings ...")
    ch_meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                          usecols=["sample_path", "mz"])
    ch_meta["mz_r"] = ch_meta["mz"].round(4)
    vocab = sorted(ch_meta["mz_r"].unique())
    mz_to_idx = {mz: i for i, mz in enumerate(vocab)}

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
    print("  [Stage 1] mean-pooling ResNet CLS per sample ...")
    ch_emb  = np.load(str(EMB_DIR / "resnet_cls_embeddings.npy"),
                      mmap_mode="r").astype(np.float32)
    ch_meta = pd.read_csv(EMB_DIR / "resnet_cls_meta.csv",
                          usecols=["sample_path"])
    return _mean_pool_channels_to_samples(sm, ch_emb, ch_meta, "Stage 1")


def load_imagenet_embeddings(sm):
    print("  [ImageNet] mean-pooling ResNet-18 CLS per sample ...")
    ch_emb  = np.load(str(EMB_DIR / "imagenet_cls_embeddings.npy"),
                      mmap_mode="r").astype(np.float32)
    ch_meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                          usecols=["sample_path"])
    return _mean_pool_channels_to_samples(sm, ch_emb, ch_meta, "ImageNet")


def _mean_pool_channels_to_samples(sm, ch_emb, ch_meta, label):
    sample_to_idx = {sp: i for i, sp in enumerate(sm["sample_path"])}
    n_samples = len(sm)
    emb_dim   = ch_emb.shape[1]
    emb_out   = np.zeros((n_samples, emb_dim), dtype=np.float32)
    counts    = np.zeros(n_samples, dtype=np.int32)

    for row_idx, sp in enumerate(ch_meta["sample_path"]):
        si = sample_to_idx.get(sp)
        if si is None:
            continue
        emb_out[si] += ch_emb[row_idx]
        counts[si]  += 1

    valid = counts > 0
    emb_out[valid] /= counts[valid, None]
    print(f"    {valid.sum()} / {n_samples} samples have {label} embeddings")
    return emb_out


def leave_study_out_recall(emb_normed, sm, k_max, batch_size=256):
    """For each sample i, retrieve k_max nearest neighbours from all samples
    NOT belonging to sample i's study (metaspace_submitter). Return recall@k
    for each k in KS."""
    n         = len(sm)
    organs    = sm["organ"].values
    study_ids = sm["study_key"].values
    unique_study = np.unique(study_ids)
    study_mask = {s: (study_ids == s) for s in unique_study}

    records = []
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = emb_normed[start:end]
        sims  = batch @ emb_normed.T

        for bi, gi in enumerate(range(start, end)):
            q_organ = organs[gi]
            q_study = study_ids[gi]

            same_study_mask = study_mask[q_study]
            row = sims[bi].copy()
            row[same_study_mask] = -np.inf
            row[gi] = -np.inf

            nn_idx = np.argsort(-row)[:k_max]
            nn_organs = organs[nn_idx]

            rec = {}
            for k in KS:
                if k <= k_max:
                    rec[f"recall@{k}"] = float((nn_organs[:k] == q_organ).mean())
            rec["sample_idx"] = gi
            rec["organ"]      = q_organ
            rec["study_key"]  = q_study
            records.append(rec)

        if start % 1024 == 0:
            print(f"    {start}/{n} ...", end="\r")

    print(f"    {n}/{n} done    ")
    return pd.DataFrame(records)


def random_baseline_recall(sm, k):
    n = len(sm)
    organ_counts = sm["organ"].value_counts()
    per_organ = {}
    for organ, ni in organ_counts.items():
        per_organ[organ] = min(k * (ni - 1) / (n - 1), 1.0)
    return per_organ


def summarise(df_res, sm, label):
    organ_counts  = sm["organ"].value_counts()
    organ_studies = sm.groupby("organ")["study_key"].nunique()

    keep = organ_counts[
        (organ_counts >= MIN_ORGAN_N) &
        (organ_studies >= MIN_STUDIES)
    ].index

    sub = df_res[df_res["organ"].isin(keep)].copy()
    per_organ = (sub.groupby("organ")[[f"recall@{k}" for k in KS]]
                    .mean()
                    .reset_index())
    per_organ["n_samples"] = per_organ["organ"].map(organ_counts)
    per_organ["n_studies"] = per_organ["organ"].map(organ_studies)
    per_organ["variant"]   = label

    for k in KS:
        rnd = random_baseline_recall(sm, k)
        per_organ[f"random@{k}"] = per_organ["organ"].map(rnd)

    overall = {}
    for k in KS:
        col, rnd_col = f"recall@{k}", f"random@{k}"
        overall[f"macro_recall@{k}"]    = per_organ[col].mean()
        overall[f"weighted_recall@{k}"] = np.average(per_organ[col], weights=per_organ["n_samples"])
        overall[f"macro_random@{k}"]    = per_organ[rnd_col].mean()
        overall[f"weighted_random@{k}"] = np.average(per_organ[rnd_col], weights=per_organ["n_samples"])
    overall["variant"]  = label
    overall["n_organs"] = len(per_organ)

    return per_organ, overall


def main():
    print("[LOAD] metadata & embeddings ...")
    sm, emb_s2 = load_metadata()
    print(f"  {len(sm)} samples, {sm['study_key'].nunique()} studies, "
          f"{sm['organ'].nunique()} organs")

    emb_s1 = load_stage1_embeddings(sm)
    emb_in = load_imagenet_embeddings(sm)
    emb_mz = build_mz_embeddings(sm)

    s2_normed = normalize(emb_s2, norm="l2")
    s1_normed = normalize(emb_s1, norm="l2")
    in_normed = normalize(emb_in, norm="l2")
    mz_normed = normalize(emb_mz, norm="l2")

    k_max = max(KS)

    print("\n[RETRIEVAL] Stage 2 ...")
    df_s2 = leave_study_out_recall(s2_normed, sm, k_max)
    print("\n[RETRIEVAL] Stage 1 ...")
    df_s1 = leave_study_out_recall(s1_normed, sm, k_max)
    print("\n[RETRIEVAL] ImageNet ResNet ...")
    df_in = leave_study_out_recall(in_normed, sm, k_max)
    print("\n[RETRIEVAL] m/z bag-of-words ...")
    df_mz = leave_study_out_recall(mz_normed, sm, k_max)

    per_organ_s2, overall_s2 = summarise(df_s2, sm, "Stage 2")
    per_organ_s1, overall_s1 = summarise(df_s1, sm, "Stage 1")
    per_organ_in, overall_in = summarise(df_in, sm, "ImageNet")
    per_organ_mz, overall_mz = summarise(df_mz, sm, "m/z")

    df_per_organ = pd.concat([per_organ_s2, per_organ_s1, per_organ_in, per_organ_mz], ignore_index=True)
    df_overall   = pd.DataFrame([overall_s2, overall_s1, overall_in, overall_mz])

    print("\n" + "="*60)
    print("LEAVE-ONE-STUDY-OUT: OVERALL (macro / weighted) Recall@k")
    print("="*60)
    for row in [overall_s2, overall_s1, overall_in, overall_mz]:
        print(f"\n{row['variant']}:")
        for k in KS:
            print(f"  R@{k:2d}  macro={row[f'macro_recall@{k}']:.3f} "
                  f"[rnd={row[f'macro_random@{k}']:.3f}]  "
                  f"weighted={row[f'weighted_recall@{k}']:.3f} "
                  f"[rnd={row[f'weighted_random@{k}']:.3f}]")

    pivot_r1 = df_per_organ.pivot(index="organ", columns="variant", values="recall@1").reset_index()
    pivot_r1["n_samples"] = pivot_r1["organ"].map(sm["organ"].value_counts())
    pivot_r1["n_studies"] = pivot_r1["organ"].map(sm.groupby("organ")["study_key"].nunique())
    n_total = len(sm)
    pivot_r1["random@1"] = pivot_r1["organ"].map(
        sm["organ"].value_counts().apply(lambda ni: (ni - 1) / (n_total - 1)))
    pivot_r1 = pivot_r1.sort_values("Stage 2", ascending=False)

    df_per_organ.to_csv(OUT_DIR / "leavestudyout_per_organ.csv", index=False)
    df_overall.to_csv(OUT_DIR / "leavestudyout_overall.csv", index=False)
    pivot_r1.to_csv(OUT_DIR / "leavestudyout_pivot_r1.csv", index=False)
    print(f"\n[DONE] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
