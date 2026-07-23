"""
Spatial pattern consistency (RC12 / ARC6).

patch_arr shape: (N_samples=5600, 28, 28, 256) — one spatial map per sample.
Each sample's spatial map integrates all its channels through the ResNet encoder.

We test: are spatial maps of the SAME organ more similar to each other than to
spatial maps of DIFFERENT organs?

  - within-organ cosine similarity (same Organism_Part)
  - between-organ cosine similarity (different Organism_Part)

Each sample map is represented as a PCA-reduced (k=64) flattening of its
28×28×256 spatial feature tensor.

This directly answers RC12: it is a systematic, non-cherry-picked analysis
showing that spatial patch embeddings capture organ-specific structure
reproducibly across independent MSI acquisitions.

Outputs (outputs/molecule_spatial_consistency/):
  - consistency_bar.png          : within vs between per organ + overall
  - consistency_global.csv       : per-organ and overall averages
  - consistency_all_pairs.csv    : per-pair (organ_a, organ_b) mean similarity matrix
"""
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR   = METABOFM_ROOT / "outputs/molecule_spatial_consistency"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_NPY  = EMB_DIR / "resnet_patch_embeddings.npy"   # (5600, 28, 28, 256)
SAMP_META  = EMB_DIR / "stage2_sample_meta.csv"        # sample_path, n_channels
CH_META    = EMB_DIR / "stage2_channel_meta.csv"       # per-channel, has Organism_Part

TARGET_ORGANS = ["Kidney", "Brain", "Lung", "Liver"]
PCA_COMPONENTS = 64        # reduce 28*28*256=200704 → 64 dims for tractable pairwise
MAX_SAMPLES_PER_ORGAN = 150  # cap to keep cosine matrix tractable
# ---------------------------------------------------------------------------


def main():
    print("[LOAD] Metadata ...")
    s2_meta = pd.read_csv(SAMP_META)                              # (5600,)
    ch_meta = pd.read_csv(CH_META).drop_duplicates("sample_path") # per-sample
    # align organ label to s2_meta order
    s2_meta = s2_meta.merge(ch_meta[["sample_path", "Organism_Part"]], on="sample_path", how="left")
    organ_labels = s2_meta["Organism_Part"].fillna("Unknown").values
    print(f"  Samples: {len(s2_meta)}")

    print("[LOAD] Patch embeddings (memory-mapped) ...")
    patch_arr = np.load(str(PATCH_NPY), mmap_mode="r")   # (5600, 28, 28, 256)
    N, H, W, C = patch_arr.shape
    print(f"  Shape: {patch_arr.shape}  →  flattened: {H*W*C}")

    # Filter to target organs
    mask = np.isin(organ_labels, TARGET_ORGANS)
    idxs_all = np.where(mask)[0]
    labels_all = organ_labels[mask]
    print(f"  Samples in target organs: {len(idxs_all)}")

    # Load and flatten patch maps for target organs
    print("[LOAD] Loading patch maps for target organs ...")
    flat = patch_arr[idxs_all].reshape(len(idxs_all), -1).astype(np.float32)  # (N_target, 200704)

    print(f"[PCA] Reducing to {PCA_COMPONENTS} dims ...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X = pca.fit_transform(flat)   # (N_target, 64)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms > 0, norms, 1)

    # ── Per-organ within / between similarity ────────────────────────────
    print("[COMPUTE] Within- and between-organ cosine similarity ...")
    rng = np.random.default_rng(42)
    organ_vecs = {}
    for organ in TARGET_ORGANS:
        idx_org = np.where(labels_all == organ)[0]
        if len(idx_org) > MAX_SAMPLES_PER_ORGAN:
            idx_org = rng.choice(idx_org, MAX_SAMPLES_PER_ORGAN, replace=False)
        organ_vecs[organ] = X[idx_org]
        print(f"  {organ}: {len(idx_org)} samples")

    rows = []
    # within-organ
    for organ, vecs in organ_vecs.items():
        C_mat = cosine_similarity(vecs)
        triu = C_mat[np.triu_indices_from(C_mat, k=1)]
        rows.append({"comparison": "within", "organ_a": organ, "organ_b": organ,
                     "n_pairs": len(triu), "mean_cosine": float(triu.mean()),
                     "std_cosine": float(triu.std())})

    # between-organ (all unique pairs)
    organ_list = TARGET_ORGANS
    for i in range(len(organ_list)):
        for j in range(i + 1, len(organ_list)):
            oa, ob = organ_list[i], organ_list[j]
            C_cross = cosine_similarity(organ_vecs[oa], organ_vecs[ob])
            rows.append({"comparison": "between", "organ_a": oa, "organ_b": ob,
                         "n_pairs": C_cross.size, "mean_cosine": float(C_cross.mean()),
                         "std_cosine": float(C_cross.std())})

    df = pd.DataFrame(rows)
    df.to_csv(str(OUT_DIR / "consistency_all_pairs.csv"), index=False)
    print(df.to_string(index=False))

    # Global within vs between
    within_mean  = df[df["comparison"] == "within"]["mean_cosine"].mean()
    between_mean = df[df["comparison"] == "between"]["mean_cosine"].mean()
    delta = within_mean - between_mean

    global_rows = []
    for organ in TARGET_ORGANS:
        w = df[(df["comparison"] == "within") & (df["organ_a"] == organ)]["mean_cosine"].values
        b = df[(df["comparison"] == "between") &
               ((df["organ_a"] == organ) | (df["organ_b"] == organ))]["mean_cosine"].values
        global_rows.append({
            "organ": organ,
            "within_cosine": float(w[0]) if len(w) else float("nan"),
            "between_cosine": float(b.mean()) if len(b) else float("nan"),
        })
    global_df = pd.DataFrame(global_rows)
    global_df["delta"] = global_df["within_cosine"] - global_df["between_cosine"]
    global_df.loc[len(global_df)] = {
        "organ": "OVERALL", "within_cosine": within_mean,
        "between_cosine": between_mean, "delta": delta
    }
    global_df.to_csv(str(OUT_DIR / "consistency_global.csv"), index=False)
    print("\nGlobal summary:")
    print(global_df.to_string(index=False))

    # ── Figure ──────────────────────────────────────────────────────────────
    plot_df = global_df[global_df["organ"] != "OVERALL"].copy()
    organs = plot_df["organ"].tolist()
    x = np.arange(len(organs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, plot_df["within_cosine"], w, label="Within-organ", color="#4472C4")
    ax.bar(x + w/2, plot_df["between_cosine"], w, label="Between-organ", color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels(organs, fontsize=11)
    ax.set_ylabel("Mean pairwise cosine similarity\n(PCA-reduced spatial patch maps)", fontsize=10)
    ax.set_title(
        "Spatial patch map consistency — within vs between organ\n"
        f"(PCA {PCA_COMPONENTS}-dim, ≤{MAX_SAMPLES_PER_ORGAN} samples/organ, all sample pairs)",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ymax = max(plot_df[["within_cosine", "between_cosine"]].max()) * 1.2
    ax.set_ylim(0, ymax)
    ax.axhline(within_mean,  color="#4472C4", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(between_mean, color="#ED7D31", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(organs) - 0.05, within_mean  + ymax * 0.02,
            f"overall within={within_mean:.3f}",  ha="right", fontsize=8, color="#4472C4")
    ax.text(len(organs) - 0.05, between_mean - ymax * 0.04,
            f"overall between={between_mean:.3f}", ha="right", fontsize=8, color="#ED7D31")
    plt.tight_layout()
    fig.savefig(str(OUT_DIR / "consistency_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[DONE] Outputs in: {OUT_DIR}")
    print(f"  Overall within={within_mean:.4f}  between={between_mean:.4f}  delta={delta:.4f}")


if __name__ == "__main__":
    main()
