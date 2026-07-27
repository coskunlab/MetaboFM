"""
probe_spatial_patches.py
-------------------------
Spatial patch-level analysis of Stage 1 ResNet embeddings.

Addresses Reviewer Comment 2: shows that MetaboFM produces spatially
resolved embeddings that capture within-tissue metabolic heterogeneity,
not just sample-level summaries.

Three analyses:

Figure A — Within-tissue spatial UMAP
  For each selected tissue sample: UMAP of 784 patches colored by (row, col)
  grid position. Shows spatial coherence: nearby patches cluster together.
  The UMAP embedding is overlaid back onto the tissue grid as a color image.

Figure B — Within-tissue metabolic clustering
  UMAP of patches from one sample, K-means clustered → each cluster overlaid
  on the original tissue image. Shows unsupervised discovery of spatial
  metabolic microregions (e.g., tissue substructures).

Figure C — Cross-tissue organ specificity at pixel level
  UMAP of patches pooled from multiple samples across organs, colored by
  Organism_Part. Shows that the model encodes organ identity even at the
  patch level — a much stronger claim than sample-level classification.

Figure D — Leave-platform-out patch retrieval
  For each query patch, retrieve nearest neighbours from a held-out platform.
  Reports recall@10 for correct organ matching — tests platform generalization
  at the spatial level (addresses Comment 8).

Figure E — Leave-ionisation-source-out organ probe
  Same LogReg setup as Figure C but held-out group is ionisationSource
  (MALDI / DESI / AP-SMALDI / IR-MALDESI). Stricter than analyzerType because
  ionization physics differ fundamentally across sources.

Figure F — Leave-organism-out organ probe
  Train on human (Homo sapiens) samples, test on mouse (Mus musculus) and
  vice versa. Tests biological generalization independent of technical platform.

Usage
-----
  python probe_spatial_patches.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
PATCH_NPY = EMB_DIR / "resnet_patch_embeddings.npy"
PATCH_CSV = EMB_DIR / "resnet_patch_meta.csv"

OUT_DIR  = METABOFM_ROOT / "outputs/spatial_patches"

PATCH_GRID    = 28
N_CLUSTERS    = 6      # K-means clusters for Figure B
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.1

# Tissues to show in within-tissue figures (A, B)
SAMPLE_ORGANS = ["Brain", "Kidney", "Liver", "Lung"]

# Organs for cross-tissue figure (C) — pick well-represented ones
CROSS_ORGANS   = ["Kidney", "Brain", "Lung", "Liver", "Breast", "Skin"]
N_PER_ORGAN    = 30    # samples per organ
N_PATCH_SUBSAMPLE = 100  # patches per sample to keep for cross-tissue UMAP

SEED = 42


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_umap():
    try:
        import umap
        return umap.UMAP
    except ImportError:
        raise RuntimeError("pip install umap-learn")


def l2_norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def get_sample_patches(patch_arr: np.ndarray, si: int) -> np.ndarray:
    """Return (784, 256) patch matrix for sample index si."""
    return patch_arr[si].reshape(PATCH_GRID * PATCH_GRID, -1)


def organ_color_map(organs: list[str]) -> dict[str, str]:
    cmap = plt.cm.get_cmap("tab20", len(organs))
    return {o: mcolors.to_hex(cmap(i)) for i, o in enumerate(sorted(set(organs)))}


# ── SPATIAL COHERENCE HELPERS ─────────────────────────────────────────────────

def most_representative_sample(patch_arr: np.ndarray, indices: list[int]) -> int:
    """Return the index closest to the centroid of the group (in mean-pooled patch space)."""
    vecs = np.array([patch_arr[i].reshape(-1) for i in indices], dtype=np.float32)
    centroid = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - centroid, axis=1)
    return indices[int(dists.argmin())]


def spatial_contiguity(labels: np.ndarray, grid: int = PATCH_GRID) -> float:
    """
    Fraction of 4-connected patch neighbours sharing the same cluster label.
    Random assignment baseline ≈ 1/n_clusters. Good clustering → >0.6.
    """
    g = labels.reshape(grid, grid)
    same = 0; total = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        r0 = max(0, -dr);  r1 = min(grid, grid - dr)
        c0 = max(0, -dc);  c1 = min(grid, grid - dc)
        same  += (g[r0:r1, c0:c1] == g[r0+dr:r1+dr, c0+dc:c1+dc]).sum()
        total += (r1 - r0) * (c1 - c0)
    return float(same / total) if total > 0 else 0.0


def pca_spatial_correlation(patches: np.ndarray, grid: int = PATCH_GRID) -> tuple[float, float]:
    """
    PCA on (784, D) patch matrix. Returns Spearman correlation of
    PC1 with patch row-index and PC2 with patch col-index.
    Uses absolute value — sign of PC is arbitrary.
    """
    from sklearn.decomposition import PCA
    from scipy.stats import spearmanr
    pca = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(patches)          # (784, 2)
    row_ids = np.arange(grid * grid) // grid
    col_ids = np.arange(grid * grid) % grid
    r_row = abs(spearmanr(proj[:, 0], row_ids).statistic)
    r_col = abs(spearmanr(proj[:, 1], col_ids).statistic)
    # also check swapped assignment (PC1~col, PC2~row)
    r_row2 = abs(spearmanr(proj[:, 0], col_ids).statistic)
    r_col2 = abs(spearmanr(proj[:, 1], row_ids).statistic)
    # take best assignment
    if r_row + r_col >= r_row2 + r_col2:
        return r_row, r_col
    return r_col2, r_row2


# ── SPATIAL COHERENCE: all samples ────────────────────────────────────────────

def compute_spatial_coherence_all_samples(
    patch_arr: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    n_clusters: int = N_CLUSTERS,
    organs_to_plot: list[str] | None = None,
):
    """
    Run K-means + spatial contiguity and PCA spatial correlation on every
    sample. Reports mean ± std across all samples and per organ.
    Saves a summary CSV and a violin/box plot.
    """
    from sklearn.cluster import MiniBatchKMeans
    from scipy.stats import spearmanr

    print(f"[Coherence] Processing {len(meta):,} samples ...")
    rows_out = []
    for si in tqdm(range(len(meta)), desc="coherence"):
        patches = get_sample_patches(patch_arr, si)
        if patches.max() == 0:
            continue
        patches = l2_norm(patches)

        # K-means contiguity
        km     = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED,
                                 n_init=3, batch_size=256)
        labels = km.fit_predict(patches)
        contiguity = spatial_contiguity(labels)

        # PCA spatial correlation
        r_row, r_col = pca_spatial_correlation(patches)

        rows_out.append({
            "sample_idx":  si,
            "organ":       meta.loc[si, "Organism_Part"] if "Organism_Part" in meta.columns else "unknown",
            "contiguity":  round(contiguity, 4),
            "pca_r_row":   round(r_row, 4),
            "pca_r_col":   round(r_col, 4),
            "pca_r_mean":  round((r_row + r_col) / 2, 4),
        })

    df = pd.DataFrame(rows_out)
    df.to_csv(out_dir / "spatial_coherence_all_samples.csv", index=False)

    # Random baseline for contiguity
    random_baseline = 1.0 / n_clusters

    print(f"\n=== Spatial coherence across {len(df):,} samples ===")
    print(f"  Contiguity  : {df['contiguity'].mean():.3f} ± {df['contiguity'].std():.3f}"
          f"  (random baseline = {random_baseline:.3f})")
    print(f"  PCA r_row   : {df['pca_r_row'].mean():.3f} ± {df['pca_r_row'].std():.3f}  (0=random)")
    print(f"  PCA r_col   : {df['pca_r_col'].mean():.3f} ± {df['pca_r_col'].std():.3f}  (0=random)")
    print(f"  PCA r_mean  : {df['pca_r_mean'].mean():.3f} ± {df['pca_r_mean'].std():.3f}")

    # Per-organ summary
    organs = [o for o in CROSS_ORGANS if o in df["organ"].values]
    if organs:
        print("\n  Per-organ mean contiguity / PCA r_mean:")
        for org in organs:
            sub = df[df["organ"].str.contains(org, case=False, na=False)]
            print(f"    {org:<12s}  n={len(sub):4d}  "
                  f"contiguity={sub['contiguity'].mean():.3f}  "
                  f"pca_r_mean={sub['pca_r_mean'].mean():.3f}")

    # ── Plot: violin of contiguity and PCA r_mean per organ ──────────────────
    plot_organs = [o for o in (organs_to_plot or CROSS_ORGANS)
                   if o in df["organ"].values]
    if not plot_organs:
        return df

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(plot_organs) * 1.5), 5))

    organ_data_cg = [df[df["organ"].str.contains(o, na=False)]["contiguity"].values
                     for o in plot_organs]
    organ_data_pc = [df[df["organ"].str.contains(o, na=False)]["pca_r_mean"].values
                     for o in plot_organs]

    vp0 = axes[0].violinplot(organ_data_cg, positions=range(len(plot_organs)),
                              showmedians=True, showextrema=False)
    axes[0].axhline(random_baseline, color="red", linestyle="--", linewidth=1.2,
                    label=f"Random ({random_baseline:.2f})")
    axes[0].set_xticks(range(len(plot_organs)))
    axes[0].set_xticklabels(plot_organs, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Spatial contiguity")
    axes[0].set_title("K-means cluster spatial contiguity\n(fraction of neighbours sharing same metabolic region)")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    vp1 = axes[1].violinplot(organ_data_pc, positions=range(len(plot_organs)),
                              showmedians=True, showextrema=False)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.2, label="Random (0)")
    axes[1].set_xticks(range(len(plot_organs)))
    axes[1].set_xticklabels(plot_organs, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Spearman |r| (PCA vs grid position)")
    axes[1].set_title("PCA spatial rank correlation\n(|Spearman r| between PC1/2 and patch row/col)")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1)

    n_total = len(df)
    fig.suptitle(
        f"Spatial coherence of patch embeddings — {n_total:,} samples\n"
        f"(selected as most representative per organ from {n_total:,} total; "
        f"contiguity random baseline = {random_baseline:.2f})",
        fontsize=9,
    )
    fig.tight_layout()
    p = out_dir / "spatial_coherence_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p.name}")
    return df


# ── Spatial UMAP: within-tissue spatial coherence ─────────────────────────────

def within_tissue_spatial_umap(patch_arr: np.ndarray, meta: pd.DataFrame,
                                organ: str, out_dir: Path, UMAP,
                                coherence_df: pd.DataFrame | None = None):
    rows = meta[meta["Organism_Part"].str.contains(organ, case=False, na=False)]
    if rows.empty:
        print(f"[SKIP] spatial UMAP: no samples for {organ}")
        return

    n_total   = len(rows)
    si        = most_representative_sample(patch_arr, rows.index.tolist())
    name      = Path(meta.loc[si, "sample_path"]).stem

    patches   = get_sample_patches(patch_arr, si)
    patches   = l2_norm(patches)
    row_ids   = np.arange(PATCH_GRID * PATCH_GRID) // PATCH_GRID
    col_ids   = np.arange(PATCH_GRID * PATCH_GRID) % PATCH_GRID

    reducer  = UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                    metric="cosine", random_state=SEED)
    umap_xy  = reducer.fit_transform(patches)

    # Per-sample coherence stats for subtitle
    if coherence_df is not None:
        organ_coh = coherence_df[coherence_df["organ"].str.contains(organ, na=False)]
        coh_str   = (f"median contiguity={organ_coh['contiguity'].median():.2f}  "
                     f"PCA |r|={organ_coh['pca_r_mean'].median():.2f}  "
                     f"across {len(organ_coh)} samples")
    else:
        coh_str = ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    sc0 = axes[0].scatter(umap_xy[:, 0], umap_xy[:, 1],
                          c=row_ids, cmap="plasma", s=10, alpha=0.85)
    plt.colorbar(sc0, ax=axes[0], label="Patch row (top→bottom)")
    axes[0].set_title(f"{organ} — embedding coloured by row position", fontsize=9)
    axes[0].set_axis_off()

    sc1 = axes[1].scatter(umap_xy[:, 0], umap_xy[:, 1],
                          c=col_ids, cmap="viridis", s=10, alpha=0.85)
    plt.colorbar(sc1, ax=axes[1], label="Patch col (left→right)")
    axes[1].set_title(f"{organ} — embedding coloured by column position", fontsize=9)
    axes[1].set_axis_off()

    umap_grid = umap_xy[:, 0].reshape(PATCH_GRID, PATCH_GRID)
    im = axes[2].imshow(umap_grid, cmap="plasma", aspect="auto")
    plt.colorbar(im, ax=axes[2], label="UMAP dim-1 value")
    axes[2].set_title("UMAP dim-1 projected back to tissue grid", fontsize=9)
    axes[2].set_axis_off()

    fig.suptitle(
        f"Within-tissue spatial coherence of patch embeddings — {organ}\n"
        f"Most representative sample (selected by centroid distance, "
        f"N={n_total} total) · {coh_str}",
        fontsize=8.5,
    )
    fig.tight_layout()
    p = out_dir / f"within_tissue_spatial_coherence_{organ}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p.name}")


# ── Metabolic microregions: within-tissue clustering ──────────────────────────

def metabolic_microregions(patch_arr: np.ndarray, meta: pd.DataFrame,
                            organ: str, out_dir: Path, UMAP,
                            coherence_df: pd.DataFrame | None = None):
    rows = meta[meta["Organism_Part"].str.contains(organ, case=False, na=False)]
    if rows.empty:
        return

    n_total  = len(rows)
    si       = most_representative_sample(patch_arr, rows.index.tolist())
    name     = Path(meta.loc[si, "sample_path"]).stem

    patches  = get_sample_patches(patch_arr, si)
    patches  = l2_norm(patches)

    km       = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels   = km.fit_predict(patches)
    contiguity = spatial_contiguity(labels)

    reducer  = UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                    metric="cosine", random_state=SEED)
    umap_xy  = reducer.fit_transform(patches)

    # Population-level contiguity for subtitle
    if coherence_df is not None:
        organ_coh = coherence_df[coherence_df["organ"].str.contains(organ, na=False)]
        pop_coh   = f"population median={organ_coh['contiguity'].median():.2f}"
    else:
        pop_coh = ""

    cmap_k   = plt.cm.get_cmap("tab10", N_CLUSTERS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for k in range(N_CLUSTERS):
        m = labels == k
        axes[0].scatter(umap_xy[m, 0], umap_xy[m, 1],
                        color=cmap_k(k), s=12, alpha=0.85, label=f"Region {k+1}")
    axes[0].legend(markerscale=2, fontsize=8, framealpha=0.8)
    axes[0].set_title(f"{organ} — metabolic regions in embedding space (UMAP)", fontsize=9)
    axes[0].set_axis_off()

    cluster_grid = labels.reshape(PATCH_GRID, PATCH_GRID)
    axes[1].imshow(cluster_grid, cmap="tab10", vmin=0, vmax=N_CLUSTERS - 1,
                   aspect="auto", interpolation="nearest")
    axes[1].set_title(f"Unsupervised metabolic microregion map ({N_CLUSTERS} regions)", fontsize=9)
    axes[1].set_axis_off()

    fig.suptitle(
        f"Metabolic microregion discovery — {organ}\n"
        f"Most representative sample (N={n_total} total) · "
        f"spatial contiguity={contiguity:.2f}  {pop_coh}  "
        f"(random baseline={1/N_CLUSTERS:.2f})",
        fontsize=8.5,
    )
    fig.tight_layout()
    p = out_dir / f"metabolic_microregions_{organ}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p.name}")


# ── FIGURE C: sample-level organ UMAP (Stage 2 sample_cls, balanced) ──────────

SAMPLE_CLS_NPY = EMB_DIR / "stage2_sample_cls.npy"
SAMPLE_CLS_CSV = EMB_DIR / "stage2_sample_meta.csv"


def _build_mz_fingerprint(meta_df: pd.DataFrame, n_bins: int = 500,
                           mz_min: float = 50.0, mz_max: float = 1200.0) -> np.ndarray:
    """Binary m/z fingerprint: one row per sample, 1 if any channel falls in that bin."""
    bins = np.linspace(mz_min, mz_max, n_bins + 1)
    sp_groups = meta_df.groupby("sample_path", sort=False)
    result = []
    for sp, grp in sp_groups:
        vec = np.zeros(n_bins, dtype=np.float32)
        idxs = np.digitize(grp["mz"].values, bins) - 1
        idxs = np.clip(idxs, 0, n_bins - 1)
        vec[idxs] = 1.0
        result.append(vec)
    return np.array(result, dtype=np.float32)


def _run_leave_platform_probe(X: np.ndarray, y_org: np.ndarray,
                               y_plat: np.ndarray, plat_classes,
                               n_cls: int, min_test: int = 5) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    rows = []
    for plat_code, plat_name in enumerate(plat_classes):
        m_test  = y_plat == plat_code
        m_train = ~m_test
        if m_test.sum() < min_test or m_train.sum() < n_cls:
            continue
        train_cls = set(y_org[m_train])
        if len(train_cls) < 2:
            continue
        m_te_ok = np.isin(y_org[m_test], list(train_cls))
        if m_te_ok.sum() < 2:
            continue
        Xtr, Xte = X[m_train], X[m_test][m_te_ok]
        ytr, yte = y_org[m_train], y_org[m_test][m_te_ok]
        clf = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced",
                                 solver="lbfgs", random_state=SEED, n_jobs=1)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        rows.append({
            "platform":      plat_name,
            "n_test":        int(m_te_ok.sum()),
            "n_organs_test": int(len(set(yte))),
            "accuracy":      round(float(accuracy_score(yte, yhat)), 3),
            "f1_macro":      round(float(f1_score(yte, yhat, average="macro")), 3),
        })
    return rows


def figure_c_leave_platform_organ_probe(out_dir: Path):
    """
    Leave-platform-out organ classification with baselines.
    Compares Stage 2, Stage 1 mean-pooled, m/z fingerprint, and random embeddings.
    """
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings("ignore")

    if not SAMPLE_CLS_NPY.exists():
        print("[SKIP] Fig C: stage2_sample_cls.npy not found")
        return

    print("[LOAD] embeddings and metadata ...")
    S2     = np.load(str(SAMPLE_CLS_NPY))           # (N_samples, 512) Stage 2
    S1_ch  = np.load(str(EMB_DIR / "resnet_cls_embeddings.npy"), mmap_mode="r")
    s2_meta   = pd.read_csv(SAMPLE_CLS_CSV)
    ch_meta_cls = pd.read_csv(EMB_DIR / "resnet_cls_meta.csv")  # channel-level (158k rows, has mz)
    pm      = (ch_meta_cls.drop_duplicates("sample_path").set_index("sample_path"))

    s2_meta["Organism_Part"] = s2_meta["sample_path"].map(pm["Organism_Part"])
    s2_meta["analyzerType"]  = s2_meta["sample_path"].map(pm["analyzerType"])

    # Filter to valid samples
    m_valid = (s2_meta["Organism_Part"].isin(CROSS_ORGANS) &
               s2_meta["analyzerType"].notna())
    s_sub   = s2_meta[m_valid].reset_index(drop=True)
    valid_idx = s2_meta[m_valid].index.tolist()

    le_org  = LabelEncoder().fit(s_sub["Organism_Part"])
    le_plat = LabelEncoder().fit(s_sub["analyzerType"])
    y_org   = le_org.transform(s_sub["Organism_Part"])
    y_plat  = le_plat.transform(s_sub["analyzerType"])
    n_cls   = len(le_org.classes_)
    print(f"  {len(s_sub):,} samples | {n_cls} organs | {len(le_plat.classes_)} platforms")

    # ── Build embedding variants ──────────────────────────────────────────────

    # Stage 2 sample_cls
    X_stage2 = l2_norm(S2[valid_idx])

    # Stage 1: mean-pool CLS tokens per sample
    print("  Building Stage 1 mean-pooled embeddings ...")
    ch_meta_full = ch_meta_cls
    sp_to_s2idx  = {sp: i for i, sp in enumerate(s2_meta["sample_path"])}
    valid_idx_set = {v: i for i, v in enumerate(valid_idx)}  # O(1) lookup
    s1_vecs = np.zeros((len(s_sub), S1_ch.shape[1]), dtype=np.float32)
    for sp, grp in ch_meta_full.groupby("sample_path", sort=False):
        if sp not in sp_to_s2idx:
            continue
        s2i = sp_to_s2idx[sp]
        if s2i not in valid_idx_set:
            continue
        row = valid_idx_set[s2i]
        s1_vecs[row] = np.asarray(S1_ch[grp.index.tolist()], dtype=np.float32).mean(0)
    X_stage1 = l2_norm(s1_vecs)

    # m/z fingerprint (binary, 500 bins)
    print("  Building m/z fingerprints ...")
    valid_sps = s_sub["sample_path"].tolist()
    ch_valid  = ch_meta_full[ch_meta_full["sample_path"].isin(valid_sps)]
    sp_order  = {sp: i for i, sp in enumerate(valid_sps)}
    mz_fp     = np.zeros((len(s_sub), 500), dtype=np.float32)
    bins      = np.linspace(50.0, 1200.0, 501)
    for sp, grp in ch_valid.groupby("sample_path", sort=False):
        if sp not in sp_order:
            continue
        idxs = np.clip(np.digitize(grp["mz"].values, bins) - 1, 0, 499)
        mz_fp[sp_order[sp]][idxs] = 1.0
    X_mz = mz_fp  # already unit-scale

    # Random baseline (same dim as Stage 2)
    rng = np.random.default_rng(SEED)
    X_rand = l2_norm(rng.standard_normal((len(s_sub), 512)).astype(np.float32))

    variants = [
        ("Stage 2 (ours)",      X_stage2, "steelblue"),
        ("Stage 1 mean-pool",   X_stage1, "seagreen"),
        ("m/z fingerprint",     X_mz,     "darkorange"),
        ("Random",              X_rand,   "lightgray"),
    ]

    # ── Run probes ────────────────────────────────────────────────────────────
    all_rows = []
    for vname, X, _ in variants:
        print(f"  Running: {vname} ...")
        rows = _run_leave_platform_probe(X, y_org, y_plat,
                                         le_plat.classes_, n_cls)
        for r in rows:
            r["variant"] = vname
        all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out_dir / "figC_leave_platform_organ_probe.csv", index=False)

    # ── Plot: F1 macro per platform, grouped by variant ───────────────────────
    platforms = (df_all[df_all["variant"] == "Stage 2 (ours)"]
                 .sort_values("f1_macro", ascending=False)["platform"].tolist())
    n_plat = len(platforms)
    n_var  = len(variants)
    bar_w  = 0.8 / n_var
    x      = np.arange(n_plat)

    fig, ax = plt.subplots(figsize=(max(10, n_plat * 1.5), 5))
    for vi, (vname, _, color) in enumerate(variants):
        sub = df_all[df_all["variant"] == vname].set_index("platform")
        f1s = [sub.loc[p, "f1_macro"] if p in sub.index else 0.0 for p in platforms]
        ns  = [int(sub.loc[p, "n_test"]) if p in sub.index else 0 for p in platforms]
        offset = (vi - n_var / 2 + 0.5) * bar_w
        ax.bar(x + offset, f1s, bar_w, label=vname, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n(n={df_all[(df_all.variant=='Stage 2 (ours)')&(df_all.platform==p)]['n_test'].values[0]})"
         if p in df_all[df_all.variant=="Stage 2 (ours)"]["platform"].values else p
         for p in platforms],
        rotation=25, ha="right", fontsize=9)
    ax.axhline(1 / n_cls, color="gray", linestyle="--", linewidth=1,
               label=f"Chance (1/{n_cls}={1/n_cls:.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 macro")
    ax.set_title(f"Leave-platform-out organ classification — F1 macro\n"
                 f"({n_cls} organs, LogReg, trained on all other platforms)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out = out_dir / "figC_leave_platform_organ_probe.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")

    # Summary table
    summ = (df_all.groupby("variant")[["f1_macro", "accuracy"]]
                  .mean().round(3)
                  .sort_values("f1_macro", ascending=False))
    print("\n=== Mean across platforms ===")
    print(summ.to_string())


# ── FIGURE D: leave-platform-out patch retrieval ──────────────────────────────

def figure_d_leave_platform_out(patch_arr: np.ndarray, meta: pd.DataFrame,
                                 out_dir: Path):
    try:
        import faiss
    except ImportError:
        print("[SKIP] Fig D: faiss not installed")
        return

    le_organ    = LabelEncoder()
    le_platform = LabelEncoder()

    # Filter to samples with valid organ and platform
    m = (meta["Organism_Part"].notna() &
         meta["analyzerType"].notna() &
         meta["Organism_Part"].isin(CROSS_ORGANS))
    sub = meta[m].copy().reset_index()

    if sub.empty:
        print("[SKIP] Fig D: no valid samples")
        return

    y_organ    = le_organ.fit_transform(sub["Organism_Part"].astype(str))
    y_platform = le_platform.fit_transform(sub["analyzerType"].astype(str))
    platforms  = le_platform.classes_

    # Sample-level mean-pooled patches → (N_samples, 256)
    patch_vecs = np.array([
        l2_norm(get_sample_patches(patch_arr, int(sub.loc[i, "index"]))).mean(0)
        for i in range(len(sub))
    ], dtype=np.float32)
    patch_vecs = l2_norm(patch_vecs)

    results = []
    K = 10
    for hold_out_plat in platforms:
        m_test  = y_platform == le_platform.transform([hold_out_plat])[0]
        m_train = ~m_test
        if m_train.sum() < K + 1 or m_test.sum() < 1:
            continue

        Xtr, Xte = patch_vecs[m_train], patch_vecs[m_test]
        ytr, yte = y_organ[m_train], y_organ[m_test]

        idx = faiss.IndexFlatIP(Xtr.shape[1])
        idx.add(Xtr)
        _, I = idx.search(Xte, min(K, len(Xtr)))

        recall = float((ytr[I] == yte[:, None]).any(axis=1).mean())
        purity = float((ytr[I] == yte[:, None]).mean())
        results.append({"platform": hold_out_plat,
                        "n_test": int(m_test.sum()),
                        "n_train": int(m_train.sum()),
                        "recall@10": round(recall, 3),
                        "purity@10": round(purity, 3)})

    if not results:
        return

    df_r = pd.DataFrame(results)
    df_r.to_csv(out_dir / "figD_leave_platform_out.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(6, len(df_r)), 4))
    x = np.arange(len(df_r))
    ax.bar(x - 0.2, df_r["recall@10"], 0.35, label="Recall@10", color="steelblue")
    ax.bar(x + 0.2, df_r["purity@10"], 0.35, label="Purity@10", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(df_r["platform"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Leave-platform-out organ retrieval at patch level\n(patch mean-pooled per sample)")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "figD_leave_platform_out.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {p.name}")
    print(df_r.to_string(index=False))


# ── FIGURE E: leave-ionisation-source-out organ probe ────────────────────────

def _build_leaveout_probe(
    group_col: str,
    label: str,
    s2_meta: pd.DataFrame,
    ch_meta_cls: pd.DataFrame,
    S2: np.ndarray,
    S1_ch: np.ndarray,
    out_dir: Path,
    fig_tag: str,
    min_group_test: int = 5,
):
    """
    Generic leave-one-group-out organ classification probe.
    group_col: column in s2_meta / ch_meta_cls used to define held-out folds.
    label: human-readable name for axis labels.
    """
    from sklearn.preprocessing import LabelEncoder
    import warnings
    warnings.filterwarnings("ignore")

    pm = ch_meta_cls.drop_duplicates("sample_path").set_index("sample_path")

    for col in (group_col, "Organism_Part", "ionisationSource", "analyzerType", "organism"):
        if col not in s2_meta.columns:
            s2_meta[col] = s2_meta["sample_path"].map(pm.get(col, pd.Series(dtype=str)))

    # Filter: need valid organ AND valid group column
    m_valid = (s2_meta["Organism_Part"].isin(CROSS_ORGANS) &
               s2_meta[group_col].notna())
    s_sub     = s2_meta[m_valid].reset_index(drop=True)
    valid_idx = s2_meta[m_valid].index.tolist()

    if len(s_sub) < 20:
        print(f"[SKIP] {fig_tag}: not enough samples ({len(s_sub)})")
        return

    le_org   = LabelEncoder().fit(s_sub["Organism_Part"])
    le_group = LabelEncoder().fit(s_sub[group_col])
    y_org    = le_org.transform(s_sub["Organism_Part"])
    y_group  = le_group.transform(s_sub[group_col])
    n_cls    = len(le_org.classes_)
    print(f"  {len(s_sub):,} samples | {n_cls} organs | {len(le_group.classes_)} {label} groups")

    # Stage 2
    X_stage2 = l2_norm(S2[valid_idx])

    # Stage 1 mean-pool
    sp_to_s2idx   = {sp: i for i, sp in enumerate(s2_meta["sample_path"])}
    valid_idx_set = {v: i for i, v in enumerate(valid_idx)}
    s1_vecs = np.zeros((len(s_sub), S1_ch.shape[1]), dtype=np.float32)
    for sp, grp in ch_meta_cls.groupby("sample_path", sort=False):
        if sp not in sp_to_s2idx:
            continue
        s2i = sp_to_s2idx[sp]
        if s2i not in valid_idx_set:
            continue
        row = valid_idx_set[s2i]
        s1_vecs[row] = np.asarray(S1_ch[grp.index.tolist()], dtype=np.float32).mean(0)
    X_stage1 = l2_norm(s1_vecs)

    # m/z fingerprint
    valid_sps = s_sub["sample_path"].tolist()
    ch_valid  = ch_meta_cls[ch_meta_cls["sample_path"].isin(valid_sps)]
    sp_order  = {sp: i for i, sp in enumerate(valid_sps)}
    mz_fp     = np.zeros((len(s_sub), 500), dtype=np.float32)
    bins      = np.linspace(50.0, 1200.0, 501)
    for sp, grp in ch_valid.groupby("sample_path", sort=False):
        if sp not in sp_order:
            continue
        idxs = np.clip(np.digitize(grp["mz"].values, bins) - 1, 0, 499)
        mz_fp[sp_order[sp]][idxs] = 1.0

    rng    = np.random.default_rng(SEED)
    X_rand = l2_norm(rng.standard_normal((len(s_sub), 512)).astype(np.float32))

    variants = [
        ("Stage 2 (ours)",    X_stage2, "steelblue"),
        ("Stage 1 mean-pool", X_stage1, "seagreen"),
        ("m/z fingerprint",   mz_fp,    "darkorange"),
        ("Random",            X_rand,   "lightgray"),
    ]

    all_rows = []
    for vname, X, _ in variants:
        print(f"    Running: {vname} ...")
        rows = _run_leave_platform_probe(X, y_org, y_group,
                                         le_group.classes_, n_cls,
                                         min_test=min_group_test)
        for r in rows:
            r["variant"] = vname
        all_rows.extend(rows)

    if not all_rows:
        print(f"[SKIP] {fig_tag}: no valid folds")
        return

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out_dir / f"{fig_tag}_probe.csv", index=False)

    groups = (df_all[df_all["variant"] == "Stage 2 (ours)"]
              .sort_values("f1_macro", ascending=False)["platform"].tolist())
    n_grp = len(groups)
    n_var = len(variants)
    bar_w = 0.8 / n_var
    x     = np.arange(n_grp)

    fig, ax = plt.subplots(figsize=(max(8, n_grp * 1.8), 5))
    for vi, (vname, _, color) in enumerate(variants):
        sub = df_all[df_all["variant"] == vname].set_index("platform")
        f1s = [sub.loc[g, "f1_macro"] if g in sub.index else 0.0 for g in groups]
        offset = (vi - n_var / 2 + 0.5) * bar_w
        ax.bar(x + offset, f1s, bar_w, label=vname, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{g}\n(n={df_all[(df_all.variant=='Stage 2 (ours)')&(df_all.platform==g)]['n_test'].values[0]})"
         if g in df_all[df_all.variant=="Stage 2 (ours)"]["platform"].values else g
         for g in groups],
        rotation=25, ha="right", fontsize=9)
    ax.axhline(1 / n_cls, color="gray", linestyle="--", linewidth=1,
               label=f"Chance (1/{n_cls}={1/n_cls:.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 macro")
    ax.set_title(f"Leave-{label}-out organ classification — F1 macro\n"
                 f"({n_cls} organs, LogReg, trained on all other {label} groups)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out_fig = out_dir / f"{fig_tag}_probe.png"
    fig.savefig(str(out_fig), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_fig.name}")

    summ = (df_all.groupby("variant")[["f1_macro", "accuracy"]]
                  .mean().round(3)
                  .sort_values("f1_macro", ascending=False))
    print(f"\n=== {fig_tag} — mean across {label} groups ===")
    print(summ.to_string())


def figure_e_leave_source_out(s2_meta, ch_meta_cls, S2, S1_ch, out_dir):
    """Leave-ionisation-source-out organ probe (MALDI / DESI / AP-SMALDI / IR-MALDESI)."""
    print("\n[Figure E] Leave-ionisation-source-out organ probe ...")
    _build_leaveout_probe(
        group_col="ionisationSource",
        label="ionisation-source",
        s2_meta=s2_meta.copy(),
        ch_meta_cls=ch_meta_cls,
        S2=S2, S1_ch=S1_ch,
        out_dir=out_dir,
        fig_tag="figE_leave_source_out",
        min_group_test=5,
    )


def figure_f_leave_organism_out(s2_meta, ch_meta_cls, S2, S1_ch, out_dir):
    """Leave-organism-out organ probe (Homo sapiens / Mus musculus)."""
    print("\n[Figure F] Leave-organism-out organ probe ...")

    # Restrict to two main organisms for a clean binary leave-one-out
    s2_meta = s2_meta.copy()
    pm = ch_meta_cls.drop_duplicates("sample_path").set_index("sample_path")
    if "organism" not in s2_meta.columns:
        s2_meta["organism"] = s2_meta["sample_path"].map(pm["organism"])

    main_orgs = ["Homo sapiens", "Mus musculus"]
    s2_meta = s2_meta[s2_meta["organism"].isin(main_orgs)].copy()

    _build_leaveout_probe(
        group_col="organism",
        label="organism",
        s2_meta=s2_meta,
        ch_meta_cls=ch_meta_cls,
        S2=S2, S1_ch=S1_ch,
        out_dir=out_dir,
        fig_tag="figF_leave_organism_out",
        min_group_test=5,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    for p in (PATCH_NPY, PATCH_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun extract_resnet_patch_embeddings.py first."
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Patch embeddings ...")
    patch_arr = np.load(str(PATCH_NPY), mmap_mode="r")
    meta = pd.read_csv(PATCH_CSV)
    print(f"  shape={patch_arr.shape}  samples={len(meta):,}")

    UMAP = load_umap()

    print("\n[Spatial coherence] Computing across all samples (fast K-means + PCA) ...")
    coherence_df = compute_spatial_coherence_all_samples(patch_arr, meta, OUT_DIR)

    print("\n[Within-tissue spatial coherence] UMAP figures ...")
    for organ in SAMPLE_ORGANS:
        within_tissue_spatial_umap(patch_arr, meta, organ, OUT_DIR, UMAP, coherence_df)

    print("\n[Metabolic microregions] Clustering figures ...")
    for organ in SAMPLE_ORGANS:
        metabolic_microregions(patch_arr, meta, organ, OUT_DIR, UMAP, coherence_df)

    print("\n[Figure C] Leave-platform-out organ probe (Stage 2 sample_cls) ...")
    figure_c_leave_platform_organ_probe(OUT_DIR)

    print("\n[Figure D] Leave-platform-out patch retrieval ...")
    figure_d_leave_platform_out(patch_arr, meta, OUT_DIR)

    # Load shared data for figures E and F (avoid re-loading in each function)
    if SAMPLE_CLS_NPY.exists():
        print("\n[LOAD] Stage 2 + channel embeddings for leave-out probes E & F ...")
        S2       = np.load(str(SAMPLE_CLS_NPY))
        S1_ch    = np.load(str(EMB_DIR / "resnet_cls_embeddings.npy"), mmap_mode="r")
        s2_meta  = pd.read_csv(SAMPLE_CLS_CSV)
        ch_meta_cls = pd.read_csv(EMB_DIR / "resnet_cls_meta.csv")

        figure_e_leave_source_out(s2_meta, ch_meta_cls, S2, S1_ch, OUT_DIR)
        figure_f_leave_organism_out(s2_meta, ch_meta_cls, S2, S1_ch, OUT_DIR)
    else:
        print("[SKIP] Figures E & F: stage2_sample_cls.npy not found")

    print(f"\n[DONE] All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
