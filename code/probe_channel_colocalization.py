"""
probe_channel_colocalization.py
---------------------------------
Test whether Stage 2 channel embeddings encode spatial co-localization.

Hypothesis: two ion channels that co-localize in tissue (appear in the same
spatial regions) should have more similar channel_refined embeddings than
channels that are spatially segregated. This validates that Stage 2 learned
biologically meaningful co-occurrence structure — without any external labels.

Method (per sampled MSI sample):
  1. Load raw 2D ion images for all channels in the sample.
  2. Compute pairwise SPATIAL OVERLAP: Pearson correlation of flattened,
     max-normalised ion images (standard co-localisation metric in MSI).
  3. Load channel embeddings for those channels.
  4. Compute pairwise EMBEDDING SIMILARITY: cosine similarity.
  5. Spearman rank correlation between the two N×N matrices (upper triangle).

Variants compared:
  - stage2_ch_refined (512-dim)  ← primary
  - resnet_only (256-dim)        ← Stage 1 baseline
  - mz_only (1-dim)              ← trivial baseline

Aggregated across N_SAMPLES randomly chosen samples.

Outputs
-------
  outputs/channel_colocalization/spearman_per_sample.csv
  outputs/channel_colocalization/summary.csv
  outputs/channel_colocalization/colocalization_violin.png

Usage
-----
  python probe_channel_colocalization.py
"""

from __future__ import annotations
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH  = METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv"
DATA_ROOT = MSI_RAW_DIR
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR   = METABOFM_ROOT / "outputs/channel_colocalization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES   = 400    # number of MSI samples to evaluate
MIN_CH      = 5      # minimum channels per sample (need ≥5 for meaningful pairwise)
MAX_CH      = 32     # cap to keep pairwise matrices tractable
SEED        = 42
# ── ────────────────────────────────────────────────────────────────────────────


def resolve_path(p: str) -> Path:
    fp = Path(p)
    if fp.is_absolute() and fp.exists():
        return fp
    c = DATA_ROOT / fp
    if c.exists():
        return c
    return fp


def load_sample_images(npz_path: str, channel_idxs: list[int]) -> np.ndarray | None:
    """Returns (len(channel_idxs), H*W) float32 array, or None on failure."""
    try:
        npz  = np.load(str(resolve_path(npz_path)))
        key  = "patch" if "patch" in npz else "data"
        imgs = npz[key]                      # (C, H, W)
        if imgs.ndim == 3 and imgs.shape[2] > imgs.shape[0]:
            imgs = imgs.transpose(2, 0, 1)
    except Exception:
        return None

    out = []
    for ci in channel_idxs:
        if ci >= imgs.shape[0]:
            return None
        im = imgs[ci].astype(np.float32).ravel()
        mx = im.max()
        if mx > 0:
            im = im / mx
        out.append(im)
    return np.stack(out, axis=0)   # (N_ch, H*W)


def pairwise_pearson(X: np.ndarray) -> np.ndarray:
    """X: (N, D) → (N, N) Pearson correlation matrix."""
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    X = X / norms
    return X @ X.T


def pairwise_cosine(X: np.ndarray) -> np.ndarray:
    """X: (N, D) → (N, N) cosine similarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    X = X / norms
    return X @ X.T


def upper_tri(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    idx = np.triu_indices(n, k=1)
    return M[idx]


def main():
    print("[LOAD] Channel CSV ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} channels, {df['sample_path'].nunique():,} samples")

    # Load embeddings
    variants = {}
    for name, fname, dim in [
        ("stage2_ch_refined", "stage2_channel_refined.npy", 512),
        ("resnet_only",        "resnet_only.npy",            256),
    ]:
        p = EMB_DIR / fname
        if p.exists():
            variants[name] = np.load(str(p), mmap_mode="r")
            print(f"  {name}: {variants[name].shape}")
        else:
            print(f"  [SKIP] {name}: not found")

    # mz_only: single feature from CSV
    mz_arr = df["mz"].fillna(0.0).values.astype(np.float32).reshape(-1, 1)
    mz_arr = (mz_arr - mz_arr.mean()) / (mz_arr.std() + 1e-8)
    variants["mz_only"] = mz_arr

    # Select samples with enough channels
    sample_groups = df.groupby("sample_path")
    eligible = [sp for sp, grp in sample_groups
                if MIN_CH <= len(grp) <= MAX_CH]
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(eligible, size=min(N_SAMPLES, len(eligible)), replace=False)
    print(f"\n[EVAL] {len(chosen)} samples (MIN_CH={MIN_CH}, MAX_CH={MAX_CH})")

    records = []
    for sp in tqdm(chosen, desc="samples"):
        grp = sample_groups.get_group(sp)
        row_ids    = grp.index.to_numpy(dtype=np.int64)   # original CSV integer index
        ch_idxs    = grp["channel_idx"].values.tolist()

        # Load raw images → spatial co-localisation matrix
        imgs = load_sample_images(sp, ch_idxs)
        if imgs is None:
            continue
        spatial_mat = pairwise_pearson(imgs)
        spatial_vec = upper_tri(spatial_mat)
        if spatial_vec.std() < 1e-6:
            continue

        for vname, emb in variants.items():
            sub = emb[row_ids]
            emb_mat = pairwise_cosine(np.asarray(sub, dtype=np.float32))
            emb_vec = upper_tri(emb_mat)
            rho, pval = spearmanr(spatial_vec, emb_vec)
            records.append({
                "sample_path": sp,
                "n_channels":  len(ch_idxs),
                "variant":     vname,
                "spearman_rho": float(rho),
                "pval":         float(pval),
            })

    df_res = pd.DataFrame(records)
    df_res.to_csv(OUT_DIR / "spearman_per_sample.csv", index=False)
    if df_res.empty:
        print("[ERROR] No records produced — check NPZ paths and embedding files.")
        return

    # Summary
    summ = (df_res.groupby("variant")["spearman_rho"]
            .agg(mean="mean", std="std", median="median",
                 pct25=lambda x: x.quantile(0.25),
                 pct75=lambda x: x.quantile(0.75),
                 n="count")
            .reset_index()
            .sort_values("mean", ascending=False))
    summ.to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n=== Channel Co-localisation → Embedding Similarity (Spearman rho) ===")
    print(summ.to_string(index=False))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        order = summ["variant"].tolist()
        data  = [df_res[df_res["variant"] == v]["spearman_rho"].values for v in order]
        vp = ax.violinplot(data, positions=range(len(order)), showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_alpha(0.6)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=15, ha="right")
        ax.set_ylabel("Spearman ρ  (co-localisation vs embedding similarity)")
        ax.set_title(f"Channel co-localisation correlation  (n={len(chosen)} samples)")
        fig.tight_layout()
        fig.savefig(str(OUT_DIR / "colocalization_violin.png"), dpi=150)
        plt.close(fig)
        print(f"\n[DONE] Outputs: {OUT_DIR}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")
        print(f"[DONE] Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
