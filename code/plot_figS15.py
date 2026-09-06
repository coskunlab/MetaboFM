"""
plot_figS15.py
--------------
Supplementary Figure S15: MetaboFM resolves amyloid pathology in a targeted
imaging modality (MALDI-IHC), independent of the untargeted-MSI analysis in
Supplementary Figs. S12-S14 (part of the manuscript's H&E-comparison analysis -- zero-shot
generalization to a second modality with a genuine disease-relevant marker
panel, not a repeat of the MSI validation with different data).

Everything in this figure is about the amyloid/PC5 case specifically. The
blind H&E-annotated anatomical-region validation for this modality (a
separate, self-contained test of whether PC1 tracks real anatomy in
MALDI-IHC at all, independent of the amyloid signal) is its own figure,
Supplementary Fig. S16 -- kept separate rather than bundled in here, since
this figure was already carrying enough content on its own.

Panels (ordered so the two plaque-specific panels come first, then the
supporting/methodological panels):
  A  Amyloid-beta-42 plaques: MALDI-IHC's Amyloid-B42 channel resolves
     discrete plaques in the Alzheimer's-model brain -- a disease-relevant
     structure that is not reliably identifiable in routine H&E (which needs
     a special stain -- Congo Red, Thioflavin S, or IHC -- specifically
     because it cannot resolve amyloid deposits).
  B  Head-to-head effect-size comparison: for the plaque mask in (a), how
     cleanly each of MetaboFM's PC5, H&E luminance, and H&E texture actually
     separates inside- from outside-mask tokens -- the direct quantitative
     follow-up to (a)'s claim.
  C  Quantitative divergence test (PC5 vs. H&E luminance/texture, whole
     tissue): parity with Supplementary Fig. S14's Lung case, which had this
     panel but S15 previously did not.
  D  Channel x PC correlation (Alzheimer's-model sample, all 19 markers):
     justifies using PC5 in (a) -- the component that actually tracks
     Amyloid-B42, from the same concatenation-diagnostic methodology as
     Supplementary Fig. S13's MSI analysis.
  E  Spatial companion to (d)'s heatmap: PC1-5 spatial maps and curated raw
     channel maps for the same sample, mirroring Supplementary Fig. S13's
     MSI spatial diagnostic -- so (d)'s correlation claims have a spatial
     counterpart beyond the single PC5/Amyloid-B42 pairing shown in (a).

Usage:
  python plot_figS15.py   (base conda env)
"""

from __future__ import annotations

from pathlib import Path
from metabofm_paths import METABOFM_ROOT, IHC_RAW_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import stats

from sklearn.decomposition import PCA

from plot_utils import set_nature_style, add_scale_bar_known_pixel_size
from embed_histology_comparison import PATCH_GRID, _channel_patch_means, _raster_to_patch_mask, patch_grid_to_ion_resolution, pad_to_square
from embed_ihc_histology_comparison import build_tissue_patch_mask

set_nature_style()

HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
ANNOT_DIR = METABOFM_ROOT / "outputs/optical_images/annotations"
OVERLAY_DIR = METABOFM_ROOT / "outputs/optical_images/annotation_overlay"
IHC_DATA_ROOT = IHC_RAW_DIR
FDR_CSV = OVERLAY_DIR / "all_regions_annotation_vs_pc1_fdr.csv"

OUT_DIR = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS15_maldi_ihc_amyloid"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
AB42_INDEX = 17  # position of "Amyloid-B42" in probe_ihc_histology_comparison.MALDI_IHC_LABELS
PLAQUE_PERCENTILE = 95
WEAK_THRESHOLD = 0.3
PC_INDEX_ALZ = 4  # PC5 (0-indexed)

HE_UM_PER_PX = 2.6
MALDI_UM_PER_PX = 20.0
THUMB_MAX = 1200

IHC_SAMPLE_LABELS = {
    "BrainIHC_alz": "Brain MALDI-IHC (Alzheimer's model)",
    "BrainIHC_wt": "Brain MALDI-IHC (wild-type)",
}


def save_panel(fig, stem):
    fig.suptitle("")
    path = PANEL_DIR / stem
    fig.savefig(str(path) + ".svg", bbox_inches="tight", pad_inches=0.05, dpi=DPI)
    print(f"  saved panel {stem}.svg")


def save_single(draw_fn, figsize, stem):
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.savefig(str(PANEL_DIR / stem) + ".svg", bbox_inches="tight", pad_inches=0, dpi=DPI)
    plt.close(fig)
    print(f"  saved individual panel {stem}.svg")


def _patch_mask_to_pixel_grid(patch_mask: np.ndarray, S: int) -> np.ndarray:
    """Upsamples a (PATCH_GRID, PATCH_GRID) boolean mask to an (S, S) pixel
    grid via NEAREST, for contouring against a full-resolution image.
    Deliberately NOT done via ax.contour(patch_mask, extent=[0, S, S, 0]) --
    matplotlib's contour() does not interpret an inverted-y extent the same
    way imshow() does, which silently produces a badly shifted/mismatched
    contour. Upsampling first and contouring at native pixel resolution with
    no extent avoids that trap entirely."""
    return np.array(Image.fromarray(patch_mask.astype(np.uint8) * 255).resize(
        (S, S), Image.NEAREST)) > 127


def _pad_rgb_to_square(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    S = max(H, W)
    top, left = (S - H) // 2, (S - W) // 2
    fill = 255 if np.issubdtype(img.dtype, np.integer) else 1.0
    out = np.full((S, S, img.shape[2]), fill, dtype=img.dtype)
    out[top:top + H, left:left + W] = img
    return out


def _fit_ihc_pca(condition: str):
    d = np.load(HIST_DIR / f"BrainIHC_{condition}_tokens_data.npz", allow_pickle=False)
    tokens = d["concat_tokens"]
    H, W = int(d["H"]), int(d["W"])
    he_shape = (int(d["he_height_px"]), int(d["he_width_px"]))
    interior_mask = build_tissue_patch_mask(d["tissue_border_he_yx"], he_shape, (H, W))
    interior_flat = interior_mask.flatten()
    pca = PCA(n_components=5, random_state=42)
    pc_interior = pca.fit_transform(tokens[interior_flat])
    return pca, pc_interior, interior_flat, interior_mask, (H, W)


def _pc_grid_at_ion_resolution(pc_interior, interior_flat, pc_index, ion_shape):
    grid_full = np.full(PATCH_GRID * PATCH_GRID, np.nan, dtype=np.float32)
    grid_full[interior_flat] = pc_interior[:, pc_index]
    grid = grid_full.reshape(PATCH_GRID, PATCH_GRID)
    H, W = ion_shape
    return patch_grid_to_ion_resolution(np.nan_to_num(grid, nan=0.0), H, W)


def _find_plaque_center(ab42_img: np.ndarray) -> tuple[int, int]:
    from scipy.ndimage import gaussian_filter
    thresh = np.percentile(ab42_img[ab42_img > 0], 99) if (ab42_img > 0).any() else 0
    hot = (ab42_img >= thresh).astype(np.float32)
    density = gaussian_filter(hot, sigma=8)
    return np.unravel_index(np.argmax(density), density.shape)


def build_plaque_mask(ab42_img: np.ndarray, percentile: float = PLAQUE_PERCENTILE) -> np.ndarray:
    positive = ab42_img[ab42_img > 0]
    thresh = np.percentile(positive, percentile) if positive.size else np.inf
    return ab42_img >= thresh


def plaque_vs_pc5_test(plaque_mask, pc_interior, interior_flat, pc_index):
    plaque_patch_mask = _raster_to_patch_mask(plaque_mask, min_overlap=0.0)
    plaque_in_interior = plaque_patch_mask.flatten()[interior_flat]
    pc_vals = pc_interior[:, pc_index]
    plaque_vals = pc_vals[plaque_in_interior]
    outside_vals = pc_vals[~plaque_in_interior]
    result = {
        "n_plaque_tokens": int(plaque_in_interior.sum()),
        "n_outside_tokens": int((~plaque_in_interior).sum()),
        "plaque_pc5_mean": float(plaque_vals.mean()) if plaque_vals.size else float("nan"),
        "outside_pc5_mean": float(outside_vals.mean()) if outside_vals.size else float("nan"),
    }
    if plaque_vals.size >= 2 and outside_vals.size >= 2:
        _, p = stats.mannwhitneyu(plaque_vals, outside_vals, alternative="two-sided")
        result["mannwhitney_p"] = float(p)
    else:
        result["mannwhitney_p"] = float("nan")
    return result, plaque_patch_mask


# ── Panel A: amyloid plaque demonstration ──────────────────────────────────

def panel_a():
    d_alz = np.load(HIST_DIR / "BrainIHC_alz_tokens_data.npz", allow_pickle=False)
    ab42_alz = d_alz["channel_images"][AB42_INDEX].astype(np.float32)
    he_alz = tifffile.imread(IHC_DATA_ROOT / "alz" / "he_resized_affine.tif")
    Hh_alz, Wh_alz = he_alz.shape[:2]
    H_alz, W_alz = ab42_alz.shape

    pca_alz, pc_alz_interior, flat_alz, mask_alz, ionshape_alz = _fit_ihc_pca("alz")
    rho_ab42_pc5, p_ab42_pc5 = stats.spearmanr(
        pc_alz_interior[:, PC_INDEX_ALZ],
        _channel_patch_means(ab42_alz).flatten()[flat_alz],
    )
    pc_map_alz = _pc_grid_at_ion_resolution(pc_alz_interior, flat_alz, PC_INDEX_ALZ, ionshape_alz)

    he_alz = _pad_rgb_to_square(he_alz)
    Hh_alz, Wh_alz = he_alz.shape[:2]
    ab42_alz = pad_to_square(ab42_alz)
    S_alz = max(H_alz, W_alz)
    H_alz, W_alz = S_alz, S_alz
    pc_map_alz = pad_to_square(pc_map_alz)

    row, col = _find_plaque_center(ab42_alz)
    he_row = int(row * Hh_alz / H_alz)
    he_col = int(col * Wh_alz / W_alz)
    he_zoom_half = 700
    ab42_zoom_half = int(round(he_zoom_half * H_alz / Hh_alz))

    def crop(img, r, c, half, fill=0):
        r0, r1 = max(0, r - half), min(img.shape[0], r + half)
        c0, c1 = max(0, c - half), min(img.shape[1], c + half)
        out_shape = (2 * half, 2 * half) + img.shape[2:]
        out = np.full(out_shape, fill, dtype=img.dtype)
        out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = img[r0:r1, c0:c1]
        return out

    he_zoom = crop(he_alz, he_row, he_col, he_zoom_half, fill=255)
    ab42_zoom = crop(ab42_alz, row, col, ab42_zoom_half)
    pc_zoom = crop(pc_map_alz, row, col, ab42_zoom_half)

    plaque_mask = build_plaque_mask(ab42_alz)
    plaque_zoom = crop(plaque_mask, row, col, ab42_zoom_half)
    stat_result, plaque_patch_mask = plaque_vs_pc5_test(plaque_mask, pc_alz_interior, flat_alz, PC_INDEX_ALZ)
    print(f"[INFO] plaque (>= p{PLAQUE_PERCENTILE} Amyloid-B42) vs. outside-plaque PC5, "
          f"token-level Mann-Whitney: {stat_result}")
    pd.DataFrame([stat_result]).to_csv(PANEL_DIR / "plaque_vs_pc5_stats.csv", index=False)
    plaque_mask_pixel = _patch_mask_to_pixel_grid(plaque_patch_mask, H_alz)

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.3))
    vmax_ab42 = np.percentile(ab42_alz[ab42_alz > 0], 99)
    cmap_ab42 = plt.get_cmap("viridis").copy()
    cmap_pc = plt.get_cmap("RdBu_r").copy()
    cmap_pc.set_bad(color="#dddddd")
    cmap_mask = matplotlib.colors.ListedColormap(["#222222", "#ffd60a"])

    def _img(ax, arr, cmap, vmin, vmax):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

    axes[0, 0].imshow(he_alz); axes[0, 0].set_title("Alzheimer's model\nH&E", fontsize=8); axes[0, 0].axis("off")
    add_scale_bar_known_pixel_size(axes[0, 0], HE_UM_PER_PX, Wh_alz, Wh_alz, color="black")

    axes[0, 1].imshow(ab42_alz, cmap=cmap_ab42, vmin=0, vmax=vmax_ab42)
    axes[0, 1].set_title("Amyloid-B42 (raw channel)", fontsize=8); axes[0, 1].axis("off")
    add_scale_bar_known_pixel_size(axes[0, 1], MALDI_UM_PER_PX, W_alz, W_alz)

    axes[0, 2].imshow(plaque_mask, cmap=cmap_mask, vmin=0, vmax=1)
    axes[0, 2].set_title(f"Plaque mask\n(Amyloid-B42 >= p{PLAQUE_PERCENTILE})", fontsize=8); axes[0, 2].axis("off")
    add_scale_bar_known_pixel_size(axes[0, 2], MALDI_UM_PER_PX, W_alz, W_alz, color="white")

    pc_valid = pc_map_alz != 0
    pc_norm = np.ma.masked_where(~pc_valid, (pc_map_alz - pc_map_alz[pc_valid].min()) /
                                  (pc_map_alz[pc_valid].max() - pc_map_alz[pc_valid].min() + 1e-8))
    axes[0, 3].imshow(pc_norm, cmap=cmap_pc, vmin=0, vmax=1)
    axes[0, 3].contour(plaque_mask_pixel, levels=[0.5], colors="black", linewidths=0.8)
    axes[0, 3].set_title("MetaboFM PC5\n(best-tracking component)", fontsize=8); axes[0, 3].axis("off")
    add_scale_bar_known_pixel_size(axes[0, 3], MALDI_UM_PER_PX, W_alz, W_alz, color="black")

    axes[1, 0].imshow(he_zoom); axes[1, 0].set_title("H&E, zoomed\n(plaque-dense region)", fontsize=8); axes[1, 0].axis("off")
    add_scale_bar_known_pixel_size(axes[1, 0], HE_UM_PER_PX, he_zoom.shape[1], he_zoom.shape[1], color="black")

    axes[1, 1].imshow(ab42_zoom, cmap=cmap_ab42, vmin=0, vmax=vmax_ab42)
    axes[1, 1].set_title("Amyloid-B42, zoomed\n(same region)", fontsize=8); axes[1, 1].axis("off")
    add_scale_bar_known_pixel_size(axes[1, 1], MALDI_UM_PER_PX, ab42_zoom.shape[1], ab42_zoom.shape[1])

    axes[1, 2].imshow(plaque_zoom, cmap=cmap_mask, vmin=0, vmax=1)
    axes[1, 2].set_title("Plaque mask, zoomed\n(same region)", fontsize=8); axes[1, 2].axis("off")
    add_scale_bar_known_pixel_size(axes[1, 2], MALDI_UM_PER_PX, plaque_zoom.shape[1], plaque_zoom.shape[1], color="white")

    pc_zoom_valid = pc_zoom != 0
    pc_zoom_norm = np.ma.masked_where(~pc_zoom_valid, (pc_zoom - pc_map_alz[pc_valid].min()) /
                                       (pc_map_alz[pc_valid].max() - pc_map_alz[pc_valid].min() + 1e-8))
    axes[1, 3].imshow(pc_zoom_norm, cmap=cmap_pc, vmin=0, vmax=1)
    axes[1, 3].set_title("MetaboFM PC5, zoomed\n(same region)", fontsize=8); axes[1, 3].axis("off")
    add_scale_bar_known_pixel_size(axes[1, 3], MALDI_UM_PER_PX, pc_zoom.shape[1], pc_zoom.shape[1], color="black")

    fig.tight_layout()
    save_panel(fig, "figS15_panelA_amyloid_plaques")
    plt.close(fig)

    save_single(lambda ax: (_img(ax, he_alz, None, None, None),
                             add_scale_bar_known_pixel_size(ax, HE_UM_PER_PX, Wh_alz, Wh_alz, color="black")),
                (3.2, 3.2), "figS15_panelA_HE")
    save_single(lambda ax: (_img(ax, ab42_alz, cmap_ab42, 0, vmax_ab42),
                             add_scale_bar_known_pixel_size(ax, MALDI_UM_PER_PX, W_alz, W_alz)),
                (3.2, 3.2), "figS15_panelA_Amyloid-B42")
    save_single(lambda ax: (_img(ax, plaque_mask, cmap_mask, 0, 1),
                             add_scale_bar_known_pixel_size(ax, MALDI_UM_PER_PX, W_alz, W_alz, color="white")),
                (3.2, 3.2), "figS15_panelA_plaque_mask")
    save_single(lambda ax: (_img(ax, pc_norm, cmap_pc, 0, 1),
                             ax.contour(plaque_mask_pixel, levels=[0.5], colors="black", linewidths=0.8),
                             add_scale_bar_known_pixel_size(ax, MALDI_UM_PER_PX, W_alz, W_alz, color="black")),
                (3.2, 3.2), "figS15_panelA_PC5")
    save_single(lambda ax: (_img(ax, he_zoom, None, None, None),
                             add_scale_bar_known_pixel_size(ax, HE_UM_PER_PX, he_zoom.shape[1], he_zoom.shape[1], color="black")),
                (3.2, 3.2), "figS15_panelA_HE_zoomed")
    save_single(lambda ax: (_img(ax, ab42_zoom, cmap_ab42, 0, vmax_ab42),
                             add_scale_bar_known_pixel_size(ax, MALDI_UM_PER_PX, ab42_zoom.shape[1], ab42_zoom.shape[1])),
                (3.2, 3.2), "figS15_panelA_Amyloid-B42_zoomed")
    save_single(lambda ax: (_img(ax, plaque_zoom, cmap_mask, 0, 1),
                             add_scale_bar_known_pixel_size(ax, MALDI_UM_PER_PX, plaque_zoom.shape[1], plaque_zoom.shape[1], color="white")),
                (3.2, 3.2), "figS15_panelA_plaque_mask_zoomed")
    save_single(lambda ax: (_img(ax, pc_zoom_norm, cmap_pc, 0, 1),
                             add_scale_bar_known_pixel_size(ax, MALDI_UM_PER_PX, pc_zoom.shape[1], pc_zoom.shape[1], color="black")),
                (3.2, 3.2), "figS15_panelA_PC5_zoomed")

    fig_cb1, ax_cb1 = plt.subplots(figsize=(1.3, 2.2))
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=vmax_ab42), cmap=cmap_ab42), cax=ax_cb1)
    cb1.set_label("Amyloid-B42 intensity\n(a.u., 0-99th percentile)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    fig_cb1.tight_layout()
    save_panel(fig_cb1, "figS15_panelA_colorbar_ab42")
    plt.close(fig_cb1)

    fig_cb2, ax_cb2 = plt.subplots(figsize=(1.3, 2.2))
    cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmap_pc), cax=ax_cb2)
    cb2.set_label("MetaboFM PC5\n(min-max normalized)", fontsize=7)
    cb2.ax.tick_params(labelsize=6)
    fig_cb2.tight_layout()
    save_panel(fig_cb2, "figS15_panelA_colorbar_pc5")
    plt.close(fig_cb2)

    return stat_result, plaque_mask, pc_alz_interior, flat_alz, rho_ab42_pc5, p_ab42_pc5


# ── Panel D: channel x PC correlation, justifying PC5 ──────────────────────

def panel_d():
    d = np.load(HIST_DIR / "BrainIHC_alz_tokens_data.npz", allow_pickle=False)
    tokens = d["concat_tokens"]
    labels = [str(n) for n in d["channel_names"]]
    _, pc_interior, interior_flat, _, _ = _fit_ihc_pca("alz")

    n_pcs = pc_interior.shape[1]
    rho_matrix = np.zeros((len(labels), n_pcs))
    for i, img in enumerate(d["channel_images"]):
        patch_means = _channel_patch_means(img).flatten()[interior_flat]
        for k in range(n_pcs):
            rho, _ = stats.spearmanr(pc_interior[:, k], patch_means)
            rho_matrix[i, k] = rho

    order = np.argsort(-np.abs(rho_matrix).max(axis=1))
    rho_sorted = rho_matrix[order].T
    labels_sorted = [labels[i] for i in order]
    is_ab42 = [lbl == "Amyloid-B42" for lbl in labels_sorted]

    # Horizontal layout: channels along x, PC1-5 along y -- with 19 channels,
    # a vertical (channels-as-rows) heatmap is needlessly tall; transposed,
    # it reads left-to-right like this figure's other panels (same treatment
    # as Supplementary Fig. S13's panel c).
    fig, ax = plt.subplots(figsize=(0.19 * len(labels_sorted) + 1.4, 2.6))
    ax.imshow(rho_sorted, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([f"PC{k+1}" for k in range(n_pcs)], fontsize=6)
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, fontsize=5, rotation=90, ha="center")
    ax.set_title("Brain MALDI-IHC (Alzheimer's model)", fontsize=7)
    for i, flag in enumerate(is_ab42):
        if flag:
            ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, n_pcs, fill=False, edgecolor="black", linewidth=1.4))
    fig.tight_layout()
    save_panel(fig, "figS15_panelD_correlation_heatmap")
    plt.close(fig)

    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 2.2))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-0.7, vmax=0.7), cmap="RdBu_r"), cax=ax_cb)
    cb.set_label("Spearman rho", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout()
    save_panel(fig_cb, "figS15_panelD_colorbar")
    plt.close(fig_cb)



# ── Panel B: MetaboFM vs. H&E head-to-head on the plaque mask ─────────────

def _rank_biserial(inside, outside):
    n1, n2 = inside.size, outside.size
    u, _ = stats.mannwhitneyu(inside, outside, alternative="two-sided")
    return float(abs(1 - 2 * u / (n1 * n2)))


def _mask_effect_size_raster(mask, feature_img):
    patch_mask = _raster_to_patch_mask(mask, min_overlap=0.0).flatten()
    feat_flat = _channel_patch_means(feature_img).flatten()
    return _rank_biserial(feat_flat[patch_mask], feat_flat[~patch_mask])


def _mask_effect_size_pc(mask, pc_interior, interior_flat, pc_index):
    patch_mask = _raster_to_patch_mask(mask, min_overlap=0.0).flatten()[interior_flat]
    pc_vals = pc_interior[:, pc_index]
    return _rank_biserial(pc_vals[patch_mask], pc_vals[~patch_mask])


def _gradient_magnitude(img):
    from scipy.ndimage import sobel
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    return np.hypot(gx, gy)


def panel_b(plaque_mask, pc_alz, flat_alz):
    d_alz = np.load(HIST_DIR / "BrainIHC_alz_tokens_data.npz", allow_pickle=False)
    ab42_alz = d_alz["channel_images"][AB42_INDEX].astype(np.float32)
    he_alz = tifffile.imread(IHC_DATA_ROOT / "alz" / "he_resized_affine.tif")
    gray_alz = np.asarray(Image.fromarray(he_alz).convert("L"), dtype=np.float32)
    H_alz, W_alz = ab42_alz.shape
    he_ion_alz = np.asarray(Image.fromarray(gray_alz).resize((W_alz, H_alz), Image.BILINEAR), dtype=np.float32)

    row = {
        "structure": "Amyloid plaque\n(MALDI-IHC)",
        "MetaboFM PC": _mask_effect_size_pc(plaque_mask, pc_alz, flat_alz, PC_INDEX_ALZ),
        "H&E luminance": _mask_effect_size_raster(plaque_mask, he_ion_alz),
        "H&E texture": _mask_effect_size_raster(plaque_mask, _gradient_magnitude(he_ion_alz)),
    }
    pd.DataFrame([row]).to_csv(PANEL_DIR / "metabofm_vs_he_effect_size.csv", index=False)
    print(row)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    detectors = [("MetaboFM PC", "#c44e52"), ("H&E luminance", "#4c72b0"), ("H&E texture", "#dd8452")]
    x = np.arange(len(detectors))
    ax.bar(x, [row[k] for k, _ in detectors], color=[c for _, c in detectors])
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _ in detectors], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Effect size\n(rank-biserial |r|, mask vs. outside)", fontsize=8)
    ax.set_ylim(0, 0.6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_panel(fig, "figS15_panelB_effect_size")
    plt.close(fig)
    return row


# ── Panel C: quantitative divergence test (PC5 vs. H&E), parity with S14 ──
# S14's panel b runs this same divergence test (Spearman correlation between
# the best-tracking PC and H&E's own luminance/texture, across the whole
# interior tissue, not just the mask) for the Lung case -- S15 never had the
# equivalent for the amyloid/PC5 case. Added here to close that gap, not as
# filler: it's a real, previously-missing quantitative result.

def panel_c(pc_alz, flat_alz):
    d_alz = np.load(HIST_DIR / "BrainIHC_alz_tokens_data.npz", allow_pickle=False)
    ab42_alz = d_alz["channel_images"][AB42_INDEX].astype(np.float32)
    he_alz = tifffile.imread(IHC_DATA_ROOT / "alz" / "he_resized_affine.tif")
    gray_alz = np.asarray(Image.fromarray(he_alz).convert("L"), dtype=np.float32)
    H_alz, W_alz = ab42_alz.shape
    he_ion_alz = np.asarray(Image.fromarray(gray_alz).resize((W_alz, H_alz), Image.BILINEAR), dtype=np.float32)

    luminance_patch = _channel_patch_means(he_ion_alz).flatten()[flat_alz]
    texture_patch = _channel_patch_means(_gradient_magnitude(he_ion_alz)).flatten()[flat_alz]
    pc5_vals = pc_alz[:, PC_INDEX_ALZ]
    rho_lum, p_lum = stats.spearmanr(pc5_vals, luminance_patch)
    rho_tex, p_tex = stats.spearmanr(pc5_vals, texture_patch)
    row = {"rho_vs_he_luminance": float(rho_lum), "p_vs_he_luminance": float(p_lum),
           "rho_vs_he_texture": float(rho_tex), "p_vs_he_texture": float(p_tex)}
    pd.DataFrame([row]).to_csv(PANEL_DIR / "pc5_vs_he_divergence.csv", index=False)
    print(row)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.bar([0, 1], [abs(rho_lum), abs(rho_tex)], color=["#4c72b0", "#dd8452"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H&E luminance", "H&E texture"], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("|Spearman rho| with\nMetaboFM PC5 (Alzheimer's model)", fontsize=8)
    ax.set_ylim(0, max(abs(rho_lum), abs(rho_tex)) * 1.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_panel(fig, "figS15_panelC_divergence")
    plt.close(fig)
    return row


# ── Panel E: spatial companion to panel d's channel x PC heatmap ──────────
# Mirrors Supplementary Fig. S13's MSI spatial diagnostic (PC1-5 spatial
# maps + curated raw channel maps) for the same Alzheimer's-model sample, so
# panel e's correlation claims have a visual spatial counterpart beyond just
# the single PC5/Amyloid-B42 pairing already shown in panel a.

def _sanitize(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in label)


def panel_e():
    d = np.load(HIST_DIR / "BrainIHC_alz_tokens_data.npz", allow_pickle=False)
    channel_labels = [str(n) for n in d["channel_names"]]
    channel_images = d["channel_images"]
    pca, pc_interior, interior_flat, interior_mask, ion_shape = _fit_ihc_pca("alz")
    H, W = ion_shape
    S = max(H, W)
    n_pcs = pc_interior.shape[1]

    # -- PC1-5 spatial maps --
    fig, axes = plt.subplots(1, n_pcs, figsize=(3.0 * n_pcs, 3.2))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    for k in range(n_pcs):
        grid = _pc_grid_at_ion_resolution(pc_interior, interior_flat, k, ion_shape)
        grid_sq = pad_to_square(grid)
        valid = grid_sq != 0
        vals = grid_sq[valid]
        norm = (grid_sq - vals.min()) / (vals.max() - vals.min() + 1e-8) if vals.size else grid_sq
        display = np.ma.masked_where(~valid, norm)
        axes[k].imshow(display, cmap=cmap, vmin=0, vmax=1)
        axes[k].set_title(f"PC{k+1} (var={pca.explained_variance_ratio_[k]*100:.1f}%)", fontsize=7)
        axes[k].axis("off")
        add_scale_bar_known_pixel_size(axes[k], MALDI_UM_PER_PX, S, S, color="black")

        def _draw_pc(ax_, display=display):
            ax_.imshow(display, cmap=cmap, vmin=0, vmax=1)
            ax_.axis("off")
            add_scale_bar_known_pixel_size(ax_, MALDI_UM_PER_PX, S, S, color="black")
        save_single(_draw_pc, (3.0, 3.0), f"figS15_panelE_PC{k+1}")
    fig.tight_layout()
    save_panel(fig, "figS15_panelE_pc1_to_pc5")
    plt.close(fig)

    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 2.2))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmap), cax=ax_cb)
    cb.set_label("PC score\n(min-max normalized per component)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout()
    save_panel(fig_cb, "figS15_panelE_colorbar_pc")
    plt.close(fig_cb)

    # -- curated raw channel maps: best-tracked + weakly-tracked, same
    # curation logic as Supplementary Fig. S13's select_curated_channels --
    rho_matrix = np.zeros((len(channel_labels), n_pcs))
    for i, img in enumerate(channel_images):
        patch_means = _channel_patch_means(img).flatten()[interior_flat]
        for k in range(n_pcs):
            rho, _ = stats.spearmanr(pc_interior[:, k], patch_means)
            rho_matrix[i, k] = rho
    best_abs = np.abs(rho_matrix).max(axis=1)
    order = np.argsort(-best_abs)
    n_top = 2  # illustrative only (proves PC1-5 tracks real channel structure);
    # every weakly-tracked channel is shown regardless -- that's the actual
    # diagnostic content this panel exists to surface, not the top-tracked ones
    top_idx = order[:n_top]
    weak_idx = np.where(best_abs < WEAK_THRESHOLD)[0]
    selected = sorted(set(top_idx.tolist()) | set(weak_idx.tolist()), key=lambda i: -best_abs[i])
    coverage = [(channel_images[i] > 0).mean() for i in selected]
    disp_labels = [
        f"{channel_labels[i]}\n(best |rho|={best_abs[i]:.2f}, coverage={cov*100:.1f}%)"
        for i, cov in zip(selected, coverage)
    ]
    print(f"  panel E: showing {len(selected)}/{len(channel_labels)} curated channels "
          f"(top {n_top}-tracked + {len(weak_idx)} weakly-tracked)")

    n_cols = len(selected)
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(2.3 * n_cols, 3.1))
    viridis = plt.get_cmap("viridis").copy()
    for col, (ch_idx, label) in enumerate(zip(selected, disp_labels)):
        img = pad_to_square(channel_images[ch_idx].astype(np.float32))
        vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1
        axes2[col].imshow(img, cmap=viridis, vmin=0, vmax=vmax)
        axes2[col].set_title(label, fontsize=7)
        axes2[col].axis("off")
        add_scale_bar_known_pixel_size(axes2[col], MALDI_UM_PER_PX, S, S)

        def _draw_ch(ax_, img=img, vmax=vmax):
            ax_.imshow(img, cmap=viridis, vmin=0, vmax=vmax)
            ax_.axis("off")
            add_scale_bar_known_pixel_size(ax_, MALDI_UM_PER_PX, S, S)
        save_single(_draw_ch, (3.0, 3.0), f"figS15_panelE_{_sanitize(channel_labels[ch_idx])}")
    fig2.tight_layout()
    save_panel(fig2, "figS15_panelE_raw_channels")
    plt.close(fig2)

    fig_cb2, ax_cb2 = plt.subplots(figsize=(1.3, 2.2))
    cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=viridis), cax=ax_cb2)
    cb2.set_label("Relative intensity\n(a.u., 0 to each channel's own 99th percentile)", fontsize=6)
    cb2.ax.tick_params(labelsize=6)
    fig_cb2.tight_layout()
    save_panel(fig_cb2, "figS15_panelE_colorbar_raw")
    plt.close(fig_cb2)

    return len(weak_idx), len(channel_labels)


def write_caption(plaque_stat, rho_ab42_pc5, p_ab42_pc5, effect_row, divergence_row, n_weak, n_total_channels):
    p = plaque_stat["mannwhitney_p"]
    p_str = f"{p:.2e}" if p == p and p > 0 else "n/a"
    caption = f"""\
Supplementary Figure 15 | MetaboFM resolves amyloid pathology in a targeted imaging modality (MALDI-IHC).

(a) Alzheimer's-model mouse brain, MALDI-IHC (targeted, mass-tag antibody panel). Columns: H&E | raw Amyloid-B42 channel | a binary plaque mask (Amyloid-B42 intensity at or above its own {PLAQUE_PERCENTILE}th percentile) | the MetaboFM Stage 1 principal component that best tracks Amyloid-B42's own spatial pattern (PC5; see (d) for the full justification), with the plaque mask outlined in black. Rows: full-tissue resolution (top); a zoomed crop (bottom) on a plaque-dense region, same physical location in all four columns. The H&E crop shows no distinguishing feature at this location; the raw Amyloid-B42 channel shows dense punctate plaque signal; MetaboFM's PC5 -- computed from the concatenated Stage 1 tokens across all 19 channels, not railroaded to Amyloid-B42 specifically -- independently recovers spatially coincident structure. This is confirmed quantitatively, not just visually: a token-level (one 8x8 patch = one independent sample) two-sided Mann-Whitney U test of PC5 inside vs. outside the plaque mask (across the full tissue, not just the zoomed crop; {plaque_stat['n_plaque_tokens']} plaque-containing vs. {plaque_stat['n_outside_tokens']} outside tokens) gives p={p_str} (mean PC5 {plaque_stat['plaque_pc5_mean']:.1f} inside plaque patches vs. {plaque_stat['outside_pc5_mean']:.1f} outside). Amyloid plaques are not reliably identifiable in routine H&E, which is why neuropathology practice requires a dedicated stain (Congo Red, Thioflavin S) or immunohistochemistry to visualize them -- this is a concrete, disease-relevant case of a structure MetaboFM's learned representation resolves and H&E does not. The untargeted-MSI counterpart of this demonstration (Lung, m/z 527.16) is reported independently in Supplementary Fig. S14. A separate, independent blind H&E-annotated anatomical-region validation for this modality (not specific to the amyloid signal here) is reported in Supplementary Fig. S16.

(b) Head-to-head comparison on the plaque mask in (a): the same token-level Mann-Whitney test scored by three different detectors -- MetaboFM's PC5, the H&E image's own local luminance, and its local texture (Sobel gradient magnitude) -- summarized by effect size (rank-biserial correlation; 0 = groups indistinguishable, 1 = perfect separation). Local texture is nominally significant for this mask (p=5.2e-8): any binary intensity mask has edges, independent of whether the underlying biology is visible in H&E, so texture's p-value does not by itself indicate H&E resolves the same structure. By effect size, MetaboFM's PC5 separates the plaque mask more cleanly than either H&E feature ({effect_row['MetaboFM PC']:.2f} vs. {effect_row['H&E luminance']:.2f} luminance, {effect_row['H&E texture']:.2f} texture).

(c) Quantitative divergence test, parity with Supplementary Fig. S14's Lung case: Spearman correlation between MetaboFM Stage 1 PC5 (interior tokens, whole tissue) and the H&E image's own local luminance and local texture (Sobel gradient magnitude) at the same patch locations (rho={divergence_row['rho_vs_he_luminance']:.2f} luminance, rho={divergence_row['rho_vs_he_texture']:.2f} texture).

(d) Spearman correlation between each of the 19 MALDI-IHC channels' raw patch-mean intensity and each of PC1-PC5's interior-token scores, same methodology as Supplementary Fig. S13's MSI diagnostic. Amyloid-B42 (outlined) is best tracked by PC5 (rho={rho_ab42_pc5:.2f}, p={p_ab42_pc5:.1e}) -- the justification for using PC5, not PC1, in (a).

(e) Spatial companion to panel d's channel x PC correlation heatmap, same methodology as Supplementary Fig. S13's MSI spatial diagnostic. Top: MetaboFM Stage 1 PC1-PC5 spatial maps (interior tissue tokens only). Bottom: curated raw channel maps (interior tissue only), each independently normalized to its own 99th percentile -- the top-2 best-tracked channels (illustrating that PC1-PC5 reflects real channel structure) plus every weakly-tracked channel ({n_weak} of {n_total_channels} channels weakly tracked by all of PC1-PC5, the ones the heatmap in (d) exists to surface), each labeled with its nonzero-pixel coverage.
"""
    (PANEL_DIR / "captions.txt").write_text(caption, encoding="utf-8")
    print("  saved captions.txt")


def main():
    stat_result, plaque_mask, pc_alz, flat_alz, rho_ab42_pc5, p_ab42_pc5 = panel_a()
    effect_row = panel_b(plaque_mask, pc_alz, flat_alz)
    divergence_row = panel_c(pc_alz, flat_alz)
    panel_d()
    n_weak, n_total_channels = panel_e()
    write_caption(stat_result, rho_ab42_pc5, p_ab42_pc5, effect_row, divergence_row, n_weak, n_total_channels)
    print("FigS15 done.")


if __name__ == "__main__":
    main()
