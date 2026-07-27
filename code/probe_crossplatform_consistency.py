"""
probe_crossplatform_consistency.py
------------------------------------
Test whether sample embeddings encode stable biological signal across platforms.

A foundation model should embed the same tissue type similarly regardless of
which instrument acquired the data. This benchmark tests that directly using
the platform diversity already in the dataset — no external labels needed.

Three similarity groups (per tissue type with coverage on ≥2 analyzers):
  A. same_tissue / same_platform   — biological + technical similarity
  B. same_tissue / diff_platform   — biological similarity only  ← key claim
  C. diff_tissue / diff_platform   — neither (random cross-tissue baseline)

Prediction: A > B > C. If B > C, cross-platform biological signal is preserved.
If stage2 B > stage1 B, the Transformer adds cross-platform generalisation.

Variants:
  - stage2_sample_cls (512-dim)
  - stage1_mean_pool: mean of resnet_only CLS over channels per sample

Outputs
-------
  outputs/crossplatform_consistency/similarity_by_group.csv
  outputs/crossplatform_consistency/summary.csv
  outputs/crossplatform_consistency/crossplatform_bar.png

Usage
-----
  python probe_crossplatform_consistency.py
"""

from __future__ import annotations
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ── CONFIG ─────────────────────────────────────────────────────────────────────
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/crossplatform_consistency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES_PER_GROUP = 5    # minimum samples per (tissue, platform) cell
N_PAIRS_PER_TYPE      = 500  # pairs sampled per group per tissue type
MIN_PLATFORMS         = 2    # tissue must appear on at least this many analyzers
SEED                  = 42
# ── ────────────────────────────────────────────────────────────────────────────


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(a @ b)


def batch_cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A * B).sum(axis=1)


def sample_pairs(idx_a: np.ndarray, idx_b: np.ndarray,
                 n: int, rng: np.random.Generator,
                 same_set: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Sample n random pairs from idx_a × idx_b (without self-pairs if same_set)."""
    pairs_i, pairs_j = [], []
    attempts = 0
    while len(pairs_i) < n and attempts < n * 10:
        attempts += 1
        i = rng.choice(idx_a)
        j = rng.choice(idx_b)
        if same_set and i == j:
            continue
        pairs_i.append(i)
        pairs_j.append(j)
    return np.array(pairs_i), np.array(pairs_j)


def build_stage1_meanpool(ch_meta: pd.DataFrame, resnet_emb: np.ndarray,
                           sample_list: list[str]) -> dict[str, np.ndarray]:
    """Mean-pool Stage 1 CLS tokens over channels per sample."""
    sp2emb = {}
    for sp in tqdm(sample_list, desc="stage1 mean-pool", leave=False):
        rows = ch_meta[ch_meta["sample_path"] == sp].index.tolist()
        if not rows:
            continue
        sp2emb[sp] = resnet_emb[rows].mean(axis=0)
    return sp2emb


def main():
    print("[LOAD] Sample metadata ...")
    ch_meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv")
    samp_meta = ch_meta.drop_duplicates("sample_path").copy().reset_index(drop=True)
    print(f"  {len(samp_meta):,} samples")

    print("[LOAD] Stage 2 sample_cls ...")
    s2_cls = np.load(str(EMB_DIR / "stage2_sample_cls.npy"))     # (N_samp, 512)
    s2_meta_raw = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv")  # sample_path, n_channels

    # Build sample_path → row_index map for stage2
    sp_to_s2idx = {sp: i for i, sp in enumerate(s2_meta_raw["sample_path"].values)}

    # Attach Organism_Part and analyzerType from channel_meta (one row per sample)
    sp_info = samp_meta.set_index("sample_path")[["Organism_Part", "analyzerType",
                                                   "organism", "ionisationSource"]]
    s2_meta = s2_meta_raw.join(sp_info, on="sample_path", how="left")

    print("[LOAD] Stage 1 ResNet CLS ...")
    resnet_emb = np.load(str(EMB_DIR / "resnet_only.npy"), mmap_mode="r")  # (N_ch, 256)
    ch_meta_indexed = ch_meta.copy()
    ch_meta_indexed.index = ch_meta_indexed.index   # original row = CSV row

    print("[BUILD] Stage 1 mean-pool per sample ...")
    all_sample_paths = s2_meta["sample_path"].tolist()
    sp_to_s1emb = build_stage1_meanpool(ch_meta_indexed, resnet_emb, all_sample_paths)
    print(f"  Built {len(sp_to_s1emb):,} Stage 1 mean-pool embeddings")

    # Tissue types with coverage on ≥2 different analyzer types
    tissue_col    = "Organism_Part"
    platform_col  = "analyzerType"

    s2_meta_clean = s2_meta.dropna(subset=[tissue_col, platform_col]).copy()
    coverage = (s2_meta_clean.groupby([tissue_col, platform_col])
                .size().reset_index(name="n_samples"))
    coverage = coverage[coverage["n_samples"] >= MIN_SAMPLES_PER_GROUP]
    tissue_platform_counts = coverage.groupby(tissue_col)[platform_col].nunique()
    valid_tissues = tissue_platform_counts[tissue_platform_counts >= MIN_PLATFORMS].index.tolist()
    print(f"\n[EVAL] Tissues with ≥{MIN_PLATFORMS} platforms: {valid_tissues}")

    rng = np.random.default_rng(SEED)
    records = []

    variants = {"stage2": s2_cls, "stage1_meanpool": None}

    for tissue in tqdm(valid_tissues, desc="tissues"):
        tissue_df = s2_meta_clean[s2_meta_clean[tissue_col] == tissue]
        platforms = tissue_df[platform_col].unique().tolist()

        # Indices by platform
        plat_idx = {p: tissue_df[tissue_df[platform_col] == p].index.tolist()
                    for p in platforms}

        # Diff-tissue samples (from completely different tissues on any platform)
        diff_tissue_df = s2_meta_clean[s2_meta_clean[tissue_col] != tissue]

        for vname in variants:
            def get_emb(row_idx):
                sp = s2_meta_clean.loc[row_idx, "sample_path"]
                s2i = sp_to_s2idx.get(sp)
                if s2i is None:
                    return None
                if vname == "stage2":
                    return s2_cls[s2i]
                else:
                    return sp_to_s1emb.get(sp)

            # Group A: same tissue, same platform
            for p in platforms:
                idx = plat_idx[p]
                if len(idx) < 2:
                    continue
                ii, jj = sample_pairs(np.array(idx), np.array(idx),
                                      N_PAIRS_PER_TYPE, rng, same_set=True)
                for i, j in zip(ii, jj):
                    ei, ej = get_emb(i), get_emb(j)
                    if ei is None or ej is None:
                        continue
                    records.append({"tissue": tissue, "variant": vname,
                                    "group": "A_same_tissue_same_platform",
                                    "sim": cosine_sim(ei, ej)})

            # Group B: same tissue, different platform
            for pi, p1 in enumerate(platforms):
                for p2 in platforms[pi+1:]:
                    idx1, idx2 = plat_idx[p1], plat_idx[p2]
                    if not idx1 or not idx2:
                        continue
                    ii, jj = sample_pairs(np.array(idx1), np.array(idx2),
                                          N_PAIRS_PER_TYPE, rng)
                    for i, j in zip(ii, jj):
                        ei, ej = get_emb(i), get_emb(j)
                        if ei is None or ej is None:
                            continue
                        records.append({"tissue": tissue, "variant": vname,
                                        "group": "B_same_tissue_diff_platform",
                                        "sim": cosine_sim(ei, ej)})

            # Group C: different tissue, different platform
            tissue_platforms = set(tissue_df[platform_col].unique())
            diff_df = diff_tissue_df[~diff_tissue_df[platform_col].isin(tissue_platforms)]
            if len(diff_df) < MIN_SAMPLES_PER_GROUP:
                diff_df = diff_tissue_df   # fall back to any diff-tissue
            if len(diff_df) < 2:
                continue
            same_idx = tissue_df.index.tolist()
            diff_idx = diff_df.index.tolist()
            ii, jj = sample_pairs(np.array(same_idx), np.array(diff_idx),
                                  N_PAIRS_PER_TYPE, rng)
            for i, j in zip(ii, jj):
                ei = get_emb(i)
                # For diff-tissue sample, need to look up in full s2_meta_clean
                sp_j = s2_meta_clean.loc[j, "sample_path"]
                s2j  = sp_to_s2idx.get(sp_j)
                if ei is None or s2j is None:
                    continue
                ej = s2_cls[s2j] if vname == "stage2" else sp_to_s1emb.get(sp_j)
                if ej is None:
                    continue
                records.append({"tissue": tissue, "variant": vname,
                                 "group": "C_diff_tissue_diff_platform",
                                 "sim": cosine_sim(ei, ej)})

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUT_DIR / "similarity_by_group.csv", index=False)

    # Summary
    summ = (df_res.groupby(["variant", "group"])["sim"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
            .sort_values(["variant", "group"]))
    summ.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n=== Cross-Platform Consistency (cosine similarity) ===")
    print(summ.to_string(index=False))

    # Check key claim: B > C for stage2
    for vname in ["stage2", "stage1_meanpool"]:
        sub = summ[summ["variant"] == vname].set_index("group")
        if "B_same_tissue_diff_platform" in sub.index and "C_diff_tissue_diff_platform" in sub.index:
            b = sub.loc["B_same_tissue_diff_platform", "mean"]
            c = sub.loc["C_diff_tissue_diff_platform", "mean"]
            print(f"\n  [{vname}] same-tissue cross-platform B={b:.4f}  vs  diff-tissue C={c:.4f}  "
                  f"delta={b-c:+.4f}  {'✓ HOLDS' if b > c else '✗ FAILS'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        group_labels = {
            "A_same_tissue_same_platform": "Same tissue\nSame platform",
            "B_same_tissue_diff_platform": "Same tissue\nDiff platform",
            "C_diff_tissue_diff_platform": "Diff tissue\nDiff platform",
        }
        group_order = list(group_labels.keys())
        vnames = ["stage2", "stage1_meanpool"]
        colors = {"stage2": "#2196F3", "stage1_meanpool": "#FF9800"}
        x = np.arange(len(group_order))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        for vi, vname in enumerate(vnames):
            sub = summ[summ["variant"] == vname].set_index("group")
            means = [sub.loc[g, "mean"] if g in sub.index else 0.0 for g in group_order]
            stds  = [sub.loc[g, "std"]  if g in sub.index else 0.0 for g in group_order]
            offset = (vi - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=vname,
                   color=colors[vname], alpha=0.8, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels([group_labels[g] for g in group_order])
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title("Cross-platform consistency of sample embeddings")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(OUT_DIR / "crossplatform_bar.png"), dpi=150)
        plt.close(fig)
        print(f"\n[DONE] Outputs: {OUT_DIR}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")
        print(f"[DONE] Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
