"""
plot_figS9.py
--------------
Supplementary Figure S9: Stage 1's organ-centroid similarity maps reflect
genuine spatial structure, not colour-scale artefacts.

Motivation: Supplementary Fig. 8 establishes that organ identity is not
trivially recoverable from raw pixel content at the whole-sample level. This
figure asks a narrower, complementary question about Stage 1's spatially-
resolved patch embeddings specifically: is the visible spatial coherence in
an organ-centroid similarity heatmap real (i.e., does neighbouring spatial
position carry information about similarity), or would the same set of
values look just as coherent if the patch positions were randomly permuted?
This mirrors the logic already established in Fig. 4c / Supplementary Fig. 6b
(real patch clusters are far more spatially contiguous than a random
assignment, 0.539 vs. 0.167) but applied directly to the organ-centroid
similarity maps rather than to unsupervised k-means clusters.

Panels:
  a  Pipeline schematic (separate .mmd file, not drawn here)
  b  Real Stage 1 organ-centroid similarity maps, one representative sample
     per organ (same construction as the retired Supplementary Fig. 8 panel
     f: top-1 PC removed, mean-centred, cosine similarity to a position-
     agnostic organ centroid estimated from a held-out pool of samples)
  c  The same similarity values as panel b, with patch positions randomly
     permuted within each sample (null model -- identical value distribution,
     no spatial structure)
  d  Quantitative comparison: spatial contiguity (foreground-patches only,
     same metric as Fig. 4c / Supplementary Fig. 6b) of the real maps in
     panel b versus the shuffled maps in panel c, across all 6 organs

Usage:
  conda run -n torch_gpu python plot_figS9.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from plot_utils import set_nature_style, MSI_DATA
set_nature_style()

# ── CONFIG ──────────────────────────────────────────────────────────────
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR = OUT_DIR / "figS9_similarity_map_null_model"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
PATCH_GRID   = 28
PATCH_PX     = 8
CENTROID_N   = 200
CENTROID_SEED = 456
SHUFFLE_SEED  = 789
FOREGROUND_THRESH = 0.05
SIM_CMAP = "RdBu_r"

ORGANS = ["Kidney", "Brain", "Liver", "Lung", "Skin", "Breast"]
VARIANT_COLORS = {"Shuffled (null)": "#aaaaaa", "Real": "#2166ac"}

CAPTION = """\
Supplementary Figure 9 | Stage 1's organ-centroid similarity maps reflect genuine spatial structure, not colour-scale artefacts.

a, Schematic: for a held-out representative sample, each of the 784 patch positions is coloured by cosine similarity to its organ's centroid (estimated from a separate pool of samples, as in Supplementary Fig. 8). The same similarity values are then compared at their real spatial positions versus randomly permuted positions (null model), testing whether the visible spatial coherence reflects genuine structure or would appear regardless of spatial arrangement.

b, Real organ-centroid similarity maps, one representative sample per organ.

c, The same similarity values as panel b, with patch positions randomly shuffled within each sample -- identical value distribution, no spatial structure.

d, Spatial contiguity (foreground patches only, fraction of 4-connected foreground-foreground neighbour pairs agreeing on above- versus below-median similarity; same base metric as Fig. 4c / Supplementary Fig. 6b) of the real maps in panel b versus the shuffled maps in panel c, across all 6 organs.
"""


def write_caption():
    (PANEL_DIR / "captions.txt").write_text(CAPTION, encoding="utf-8")
    print("  saved captions.txt")


def save_panel(fig, stem):
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    fig.savefig(str(PANEL_DIR / f"{stem}.svg"), bbox_inches="tight", pad_inches=0)
    print(f"  saved panel {stem}.svg")


# ── DATA HELPERS (mirrors the retired Supp Fig 13 panel f machinery) ────────

def _load_resnet_meta():
    resnet_meta = pd.read_csv(EMB_DIR / "resnet_patch_meta.csv")
    fix = {"Kideny": "Kidney", "colon": "Colon"}
    resnet_meta["organ"] = resnet_meta["Organism_Part"].apply(lambda s: fix.get(str(s), str(s)))
    return resnet_meta


def _resized_mean_projection(sample_path: str, size: int = 224) -> np.ndarray | None:
    p = MSI_DATA / Path(sample_path).name
    if not p.exists():
        return None
    patch = np.load(str(p))["patch"].astype(np.float32)
    proj = patch.mean(axis=0)
    nz = proj[proj > 0]
    lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0.0, 1.0))
    proj = np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1)
    img = np.array(
        Image.fromarray((proj * 255).astype(np.uint8)).resize((size, size), Image.NEAREST),
        dtype=np.float32,
    ) / 255.0
    return img


def _raw_patch_grid(img224: np.ndarray) -> np.ndarray:
    g = img224.reshape(PATCH_GRID, PATCH_PX, PATCH_GRID, PATCH_PX)
    return g.transpose(0, 2, 1, 3).reshape(PATCH_GRID * PATCH_GRID, PATCH_PX * PATCH_PX)


def _fit_stage1_whitener(resnet_meta, rng):
    from sklearn.decomposition import PCA
    emb = np.load(str(EMB_DIR / "resnet_patch_embeddings.npy"), mmap_mode="r")
    idx = rng.choice(len(resnet_meta), size=min(500, len(resnet_meta)), replace=False)
    fit_vecs = emb[idx].reshape(-1, emb.shape[-1]).astype(np.float32)
    pca = PCA(n_components=1, random_state=0)
    pca.fit(fit_vecs)
    print(f"  [whiten] Stage 1: top-1 PC explained variance ratio "
          f"{pca.explained_variance_ratio_[0]:.5f}")
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def _apply_whitener(vecs, mean_vec, comps):
    c = vecs - mean_vec
    if comps.shape[0] == 0:
        return c
    return c - (c @ comps.T) @ comps


def _organ_centroids_s1(resnet_meta, emb, rng, mean_s1, comps_s1):
    centroids = {}
    for organ in ORGANS:
        idx_all = resnet_meta.index[resnet_meta["organ"] == organ].to_numpy()
        idx_sub = rng.choice(idx_all, size=min(CENTROID_N, len(idx_all)), replace=False)
        vecs = emb[idx_sub].reshape(-1, emb.shape[-1]).astype(np.float32)
        vecs = _apply_whitener(vecs, mean_s1, comps_s1)
        centroids[organ] = vecs.mean(axis=0)
        print(f"  [centroid] {organ}: {len(idx_sub)} samples")
    return centroids


def _cosine_sim_map(vecs, centroid):
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    c = centroid / (np.linalg.norm(centroid) + 1e-8)
    return (v @ c).reshape(PATCH_GRID, PATCH_GRID)


def _spatial_contiguity(labels, grid=PATCH_GRID, fg_mask=None):
    g = labels.reshape(grid, grid)
    fg = fg_mask.reshape(grid, grid) if fg_mask is not None else np.ones_like(g, dtype=bool)
    same = 0; total = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r0 = max(0, -dr); r1 = min(grid, grid - dr)
        c0 = max(0, -dc); c1 = min(grid, grid - dc)
        pair_fg = fg[r0:r1, c0:c1] & fg[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        same += ((g[r0:r1, c0:c1] == g[r0 + dr:r1 + dr, c0 + dc:c1 + dc]) & pair_fg).sum()
        total += pair_fg.sum()
    return float(same / total) if total > 0 else 0.0


def draw_panels_bcd(resnet_meta, emb, centroids_s1, mean_s1, comps_s1):
    from plot_utils import pick_best_sample, add_scale_bar_stretched
    rng_shuffle = np.random.default_rng(SHUFFLE_SEED)
    records = []

    for organ in ORGANS:
        candidates = resnet_meta.index[resnet_meta["organ"] == organ]
        sp = pick_best_sample(resnet_meta.loc[candidates, "sample_path"].unique())
        if sp is None:
            print(f"  [SKIP] organ={organ}: no representative sample")
            continue
        row_idx = resnet_meta.index[resnet_meta["sample_path"] == sp]
        if len(row_idx) == 0:
            continue
        i = int(row_idx[0])
        slug = organ.lower()

        s1_vecs = emb[i].reshape(-1, emb.shape[-1]).astype(np.float32)
        s1_vecs = _apply_whitener(s1_vecs, mean_s1, comps_s1)
        sim_real = _cosine_sim_map(s1_vecs, centroids_s1[organ])

        img = _resized_mean_projection(sp)
        raw_grid = _raw_patch_grid(img)
        fg_mask = raw_grid.mean(axis=1) > FOREGROUND_THRESH

        # panel b: real map
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(sim_real, cmap=SIM_CMAP, vmin=-1, vmax=1, aspect="equal",
                  interpolation="antialiased")
        ax.axis("off")
        add_scale_bar_stretched(ax, sp, PATCH_GRID, PATCH_GRID, color="black", fontsize=6)
        save_panel(fig, f"figS9_panelB_{slug}_real")
        plt.close(fig)

        # panel c: same values, positions shuffled
        flat = sim_real.flatten().copy()
        rng_shuffle.shuffle(flat)
        sim_shuffled = flat.reshape(PATCH_GRID, PATCH_GRID)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(sim_shuffled, cmap=SIM_CMAP, vmin=-1, vmax=1, aspect="equal",
                  interpolation="antialiased")
        ax.axis("off")
        save_panel(fig, f"figS9_panelC_{slug}_shuffled")
        plt.close(fig)

        # reference ion image
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, cmap="viridis", aspect="equal", interpolation="antialiased")
        ax.axis("off")
        add_scale_bar_stretched(ax, sp, 224, 224, fontsize=6)
        save_panel(fig, f"figS9_panelB_{slug}_reference")
        plt.close(fig)

        # panel d data: foreground-restricted contiguity, real vs shuffled
        sim_real_flat = sim_real.flatten()
        med_real = np.median(sim_real_flat[fg_mask]) if fg_mask.any() else np.median(sim_real_flat)
        labels_real = (sim_real_flat >= med_real).astype(int)

        # shuffle the foreground mask consistently with the value shuffle:
        # re-derive shuffled labels the same way we shuffled the values
        fg_flat = fg_mask.copy()
        combined = np.stack([sim_real.flatten(), fg_flat.astype(np.float32)], axis=1)
        rng_shuffle2 = np.random.default_rng(SHUFFLE_SEED)  # fresh, same seed as value shuffle
        perm = rng_shuffle2.permutation(len(combined))
        shuffled_vals = combined[perm, 0]
        shuffled_fg = combined[perm, 1].astype(bool)
        med_shuf = np.median(shuffled_vals[shuffled_fg]) if shuffled_fg.any() else np.median(shuffled_vals)
        labels_shuf = (shuffled_vals >= med_shuf).astype(int)

        contig_real = _spatial_contiguity(labels_real, fg_mask=fg_mask)
        contig_shuf = _spatial_contiguity(labels_shuf, fg_mask=shuffled_fg)
        records.append({"organ": organ, "contiguity_real": contig_real,
                        "contiguity_shuffled": contig_shuf})
        print(f"  [contiguity] {organ}: real={contig_real:.3f}  shuffled={contig_shuf:.3f}")

    return pd.DataFrame(records)


def draw_panel_d(ax, df):
    organs = df["organ"].tolist()
    y = np.arange(len(organs))
    h = 0.35
    ax.barh(y + h / 2, df["contiguity_real"], height=h, color=VARIANT_COLORS["Real"],
            alpha=0.9, edgecolor="white", linewidth=0.5, label="Real")
    ax.barh(y - h / 2, df["contiguity_shuffled"], height=h, color=VARIANT_COLORS["Shuffled (null)"],
            alpha=0.9, edgecolor="white", linewidth=0.5, label="Shuffled (null)")
    for i, row in df.iterrows():
        ax.text(row["contiguity_real"] + 0.01, i + h / 2, f"{row['contiguity_real']:.3f}",
                va="center", fontsize=8, color="#222")
        ax.text(row["contiguity_shuffled"] + 0.01, i - h / 2, f"{row['contiguity_shuffled']:.3f}",
                va="center", fontsize=8, color="#222")
    ax.axvline(0.5, color="#888", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(organs, fontsize=10)
    ax.set_xlabel("Spatial contiguity of similarity map", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")


def main():
    resnet_meta = _load_resnet_meta()
    rng = np.random.default_rng(CENTROID_SEED)
    mean_s1, comps_s1 = _fit_stage1_whitener(resnet_meta, rng)

    emb = np.load(str(EMB_DIR / "resnet_patch_embeddings.npy"), mmap_mode="r")
    centroids_s1 = _organ_centroids_s1(resnet_meta, emb, rng, mean_s1, comps_s1)

    df = draw_panels_bcd(resnet_meta, emb, centroids_s1, mean_s1, comps_s1)

    fig_d, ax_d = plt.subplots(figsize=(6.5, 3.5))
    draw_panel_d(ax_d, df)
    save_panel(fig_d, "figS9_panelD_contiguity")
    plt.close(fig_d)

    print(df.to_string(index=False))
    write_caption()
    print("[DONE] outputs ->", PANEL_DIR)


if __name__ == "__main__":
    main()
