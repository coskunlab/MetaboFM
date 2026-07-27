"""
Sample-level UMAP of Stage 2 sample_cls embeddings (5600 samples × 512 dim).

Outputs (to outputs/sample_umap/):
  - sample_umap_organ.png        : UMAP colored by Organism_Part
  - sample_umap_organism.png     : UMAP colored by organism (human vs mouse)
  - sample_umap_platform.png     : UMAP colored by analyzerType
  - silhouette_scores.csv        : per-organ silhouette score + macro average
  - sample_umap_comparison.png   : Stage1 mean-pool vs Stage2 side-by-side
"""
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/sample_umap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_CLS   = EMB_DIR / "stage2_sample_cls.npy"           # (N_samples, 512)
S2_META  = EMB_DIR / "stage2_sample_meta.csv"           # sample_path
CH_META  = EMB_DIR / "stage2_channel_meta.csv"          # per-channel with Organism_Part
S1_EMB   = EMB_DIR / "resnet_cls_embeddings.npy"        # (N_channels, 256)
S1_META  = EMB_DIR / "resnet_cls_meta.csv"

# Minimum samples per organ to include in UMAP plot legend individually
MIN_SAMPLES = 30
# ---------------------------------------------------------------------------


def load_sample_meta():
    """Build per-sample metadata by taking first channel row per sample_path."""
    ch = pd.read_csv(CH_META)
    # One row per sample_path: take first occurrence (all metadata cols are identical within a sample)
    samp = ch.drop_duplicates(subset="sample_path", keep="first").reset_index(drop=True)
    return samp


def stage1_mean_pool(s2_sample_paths):
    """Compute Stage 1 mean-pooled embedding per sample (average ResNet CLS over channels)."""
    s1 = np.load(str(S1_EMB), mmap_mode="r")          # (N_ch, 256)
    s1_meta = pd.read_csv(S1_META)
    results = []
    for sp in s2_sample_paths:
        idx = s1_meta.index[s1_meta["sample_path"] == sp].tolist()
        if len(idx) == 0:
            results.append(np.zeros(s1.shape[1], dtype=np.float32))
        else:
            results.append(s1[idx].mean(axis=0))
    return np.stack(results, axis=0)


def run_umap(X):
    """Return 2-D UMAP projection of X."""
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric="cosine", random_state=42, verbose=True)
    return reducer.fit_transform(X)


def nice_palette(n):
    """Return n distinct colors cycling through tab20 + Set3."""
    import matplotlib.colors as mcolors
    base = list(plt.cm.tab20.colors) + list(plt.cm.Set3.colors)
    return [base[i % len(base)] for i in range(n)]


def plot_umap(umap2d, labels, title, out_path, min_samples=MIN_SAMPLES):
    """Scatter plot with per-category color; rare categories merged into 'Other'."""
    labels = np.asarray(labels, dtype=str)
    counts = pd.Series(labels).value_counts()
    rare = counts[counts < min_samples].index.tolist()
    labels_plot = np.where(np.isin(labels, rare), "Other", labels)

    cats = sorted(set(labels_plot))
    if "Other" in cats:
        cats = [c for c in cats if c != "Other"] + ["Other"]
    colors = nice_palette(len(cats))
    cat2col = {c: colors[i] for i, c in enumerate(cats)}
    cat2col["Other"] = "#aaaaaa"

    fig, ax = plt.subplots(figsize=(9, 7))
    for cat in cats:
        mask = labels_plot == cat
        ax.scatter(umap2d[mask, 0], umap2d[mask, 1],
                   c=[cat2col[cat]], label=f"{cat} ({mask.sum()})",
                   s=10, alpha=0.7, linewidths=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              markerscale=2, frameon=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def compute_silhouette(umap2d, labels):
    """Silhouette score per category (macro average); requires ≥2 unique labels."""
    labels = np.asarray(labels, dtype=str)
    counts = pd.Series(labels).value_counts()
    valid = counts[counts >= 2].index
    mask = np.isin(labels, valid)
    if mask.sum() < 2 or len(valid) < 2:
        return pd.DataFrame()
    le = LabelEncoder()
    y = le.fit_transform(labels[mask])
    try:
        macro = silhouette_score(umap2d[mask], y)
    except Exception:
        macro = float("nan")
    rows = []
    for lab in valid:
        m2 = np.isin(labels[mask], [lab])
        # per-class mean of sample silhouette values (sklearn computes per-sample scores)
        from sklearn.metrics import silhouette_samples
        sil_vals = silhouette_samples(umap2d[mask], y)
        cls_idx = le.transform([lab])[0]
        rows.append({"organ": lab, "n_samples": int(counts[lab]),
                     "silhouette_mean": float(sil_vals[y == cls_idx].mean())})
    df = pd.DataFrame(rows).sort_values("silhouette_mean", ascending=False)
    df.loc[len(df)] = {"organ": "MACRO_AVG", "n_samples": int(mask.sum()), "silhouette_mean": macro}
    return df


def side_by_side_comparison(u_s1, u_s2, labels, out_path):
    """Two-panel comparison: Stage1 mean-pool vs Stage2 sample_cls."""
    labels = np.asarray(labels, dtype=str)
    cats = sorted(set(labels))
    colors = nice_palette(len(cats))
    cat2col = {c: colors[i] for i, c in enumerate(cats)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, u, title in zip(axes, [u_s1, u_s2],
                             ["Stage 1 (mean-pool, 256-dim)", "Stage 2 sample_cls (512-dim)"]):
        for cat in cats:
            mask = labels == cat
            ax.scatter(u[mask, 0], u[mask, 1], c=[cat2col[cat]],
                       label=cat, s=10, alpha=0.7, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.set_ylabel("UMAP-2", fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, bbox_to_anchor=(0.5, -0.02), loc="upper center",
               ncol=min(6, len(cats)), fontsize=7, markerscale=2, frameon=False)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("[LOAD] Stage 2 sample_cls embeddings ...")
    X_s2 = np.load(str(S2_CLS))                        # (N, 512)
    s2_meta_raw = pd.read_csv(S2_META)                  # sample_path, n_channels
    print(f"  X_s2 shape: {X_s2.shape}")

    print("[LOAD] Per-sample metadata (from channel CSV) ...")
    samp_meta = load_sample_meta()
    # Align samp_meta to s2_meta_raw ordering
    samp_meta = s2_meta_raw[["sample_path"]].merge(
        samp_meta, on="sample_path", how="left"
    )
    print(f"  Samples with organ label: {samp_meta['Organism_Part'].notna().sum()}/{len(samp_meta)}")

    organ_labels  = samp_meta["Organism_Part"].fillna("Unknown").values
    org_labels    = samp_meta["organism"].fillna("Unknown").values
    platform_lbls = samp_meta["analyzerType"].fillna("Unknown").values

    # -----------------------------------------------------------------------
    print("\n[UMAP] Running UMAP on Stage 2 embeddings ...")
    u_s2 = run_umap(X_s2)
    np.save(str(OUT_DIR / "umap2d_stage2.npy"), u_s2)

    plot_umap(u_s2, organ_labels,  "Stage 2 UMAP — Organ (Organism_Part)",
              OUT_DIR / "sample_umap_organ.png")
    plot_umap(u_s2, org_labels,    "Stage 2 UMAP — Organism",
              OUT_DIR / "sample_umap_organism.png", min_samples=0)
    plot_umap(u_s2, platform_lbls, "Stage 2 UMAP — Analyzer Type (Platform)",
              OUT_DIR / "sample_umap_platform.png", min_samples=0)

    # -----------------------------------------------------------------------
    print("\n[Silhouette] Computing per-organ silhouette scores ...")
    df_sil = compute_silhouette(u_s2, organ_labels)
    if len(df_sil):
        df_sil.to_csv(str(OUT_DIR / "silhouette_scores.csv"), index=False)
        print(df_sil.to_string(index=False))

    # -----------------------------------------------------------------------
    print("\n[Stage1 comparison] Mean-pooling Stage 1 embeddings ...")
    X_s1 = stage1_mean_pool(s2_meta_raw["sample_path"].tolist())
    print(f"  X_s1 mean-pool shape: {X_s1.shape}")
    print("[UMAP] Running UMAP on Stage 1 mean-pool ...")
    u_s1 = run_umap(X_s1)
    np.save(str(OUT_DIR / "umap2d_stage1_meanpool.npy"), u_s1)

    plot_umap(u_s1, organ_labels, "Stage 1 mean-pool UMAP — Organ",
              OUT_DIR / "sample_umap_stage1_organ.png")

    side_by_side_comparison(u_s1, u_s2, organ_labels,
                            OUT_DIR / "sample_umap_comparison.png")

    # Silhouette for Stage1
    df_sil_s1 = compute_silhouette(u_s1, organ_labels)
    if len(df_sil_s1):
        df_sil_s1.to_csv(str(OUT_DIR / "silhouette_scores_stage1.csv"), index=False)
        print("\nStage 1 silhouette:")
        print(df_sil_s1.to_string(index=False))

    print(f"\n[DONE] All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
