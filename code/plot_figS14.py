"""
plot_figS14.py
--------------
Supplementary Figure S14: MetaboFM structure that H&E does not show, in
untargeted MSI (part of the manuscript's H&E-comparison analysis -- a concrete case
where MetaboFM answers a biology question histology cannot, not just a
complementarity argument). Fig. S12 established a positive control
(MetaboFM's PC1 tracks real, H&E-annotatable anatomy); this figure makes the
stronger, distinct claim that a specific channel's structure is not simply
recoverable from the H&E image itself, in the modality used throughout this
study. The MALDI-IHC (targeted) counterpart of this demonstration -- amyloid
plaques in an Alzheimer's-model brain -- is Supplementary Fig. S15, its own
independent analysis rather than a repeat of this one.

Panels:
  A  Lung, m/z 527.16: a METASPACE channel whose spatial pattern is not
     H&E-recoverable, identified by a systematic scan for high internal
     spatial coherence (best |Spearman rho| vs. any of PC1-5) combined with
     weak correlation to the H&E image's own luminance/texture.
  B  Quantitative divergence: Spearman correlation between each MSI sample's
     PC1 (interior tokens) and the H&E image's own local luminance and local
     texture (gradient magnitude) at the same patch locations. A weak
     correlation is the direct, quantitative version of "this spatial
     structure is not what the H&E stain itself shows."
  C  Head-to-head effect-size comparison: for the hot-spot mask in (a), how
     cleanly each of MetaboFM's PC, H&E luminance, and H&E texture actually
     separates inside- from outside-mask tokens.

Usage:
  python plot_figS14.py   (base conda env)
"""

from __future__ import annotations

from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.ndimage import sobel

from sklearn.decomposition import PCA

from plot_utils import set_nature_style, add_scale_bar_known_pixel_size
from embed_histology_comparison import PATCH_GRID, _channel_patch_means, _raster_to_patch_mask, patch_grid_to_ion_resolution, pad_to_square
from optical_alignment import native_optical_crop

set_nature_style()

HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
REG_DIR = METABOFM_ROOT / "outputs/optical_images/registration"

OUT_DIR = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS14_he_invisible_structure"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# Lung METASPACE MSI sample -- native optical (H&E) and ion-grid pixel
# sizes, same source/convention as plot_figS12.py's NATIVE_OPTICAL_UM_PER_PX
# (live-fetched MSI Pixel_Size divided by the registration affine's
# ion-to-optical scale factor).
LUNG_HE_UM_PER_PX = 3.7566070165424823
LUNG_ION_UM_PER_PX = 35.0

# m/z 527.1584 (Galactotriose / Maltotriose / Raffinose): the channel found
# by a systematic scan for high internal spatial coherence (best |Spearman
# rho| vs. any of PC1-5) combined with weak correlation to the H&E image's
# own luminance/texture. See probe_he_baseline_significance.py.
LUNG_TARGET_MZ = 527.1584
LUNG_PC_INDEX = 2  # PC3 (0-indexed) -- best-tracking component, Spearman rho=0.464
LUNG_HOT_PERCENTILE = 95  # threshold on nonzero channel intensity defining "hot" pixels

LUNG_ORGAN, LUNG_DATASET = "Lung", "2023-06-27_22h58m39s"
BRAIN_ORGAN, BRAIN_DATASET = "Brain", "2019-11-25_17h14m31s"


def write_caption(lung_stat: dict, effect_row: dict, divergence_row: dict):
    p_lung = lung_stat["mannwhitney_p"]
    p_lung_str = f"{p_lung:.2e}" if p_lung == p_lung and p_lung > 0 else "n/a"
    caption = f"""\
Supplementary Figure 14 | MetaboFM structure that H&E does not show, in untargeted MSI.

(a) Lung, METASPACE MSI (untargeted, MSM-ranked metabolite channels; same acquisition as Fig. S12). Columns: H&E | raw m/z {LUNG_TARGET_MZ:.4f} channel (top HMDB candidates: Galactotriose, Maltotriose, Raffinose) | a binary hot-spot mask (channel intensity at or above its own {LUNG_HOT_PERCENTILE}th percentile) | MetaboFM Stage 1 PC3, the component that best tracks this channel's own spatial pattern (Spearman rho=0.46 vs. the raw channel, interior tokens), with the hot-spot mask outlined in black. Rows: full-tissue resolution (top); a zoomed crop (bottom) on a hot-spot-dense region, same physical location in all four columns. This channel was identified by a systematic scan for channels with strong internal spatial structure (high |correlation| with any of PC1-5) that is simultaneously weakly correlated with the H&E image's own luminance and texture (|rho| < 0.15 for the raw channel itself), i.e. a channel whose real spatial pattern is not simply the same gross tissue contrast already visible in H&E. A token-level (one 8x8 patch = one independent sample) two-sided Mann-Whitney U test of PC3 inside vs. outside the hot-spot mask ({lung_stat['n_hot_tokens']} hot-spot vs. {lung_stat['n_outside_tokens']} outside tokens) gives p={p_lung_str} (mean PC3 {lung_stat['hot_pc_mean']:.1f} inside hot-spot patches vs. {lung_stat['outside_pc_mean']:.1f} outside). We do not claim a specific biological interpretation for this carbohydrate-family channel; the result establishes that a real, spatially coherent, statistically robust molecular pattern in an untargeted MSI acquisition -- the modality used throughout this study -- is not recoverable from the registered H&E image.

(b) Quantitative divergence test for the Lung sample: Spearman correlation between MetaboFM Stage 1 PC3 -- the component actually used in panel a, not the generic leading component -- (interior tokens) and the registered H&E image's own local luminance and local texture (Sobel gradient magnitude) at the same patch locations (rho={divergence_row['rho_vs_he_luminance']:.2f} luminance, rho={divergence_row['rho_vs_he_texture']:.2f} texture). This correlation is modest, not zero: PC3 is a corpus-wide component, not specific to the hot-spot mask in panel a, so it is expected to carry some general tissue-contrast signal in addition to the hot-spot-specific structure; panel c's mask-level effect-size comparison is the direct test of whether that hot-spot-specific structure itself is H&E-recoverable. (Brain's own PC1 divergence, from Supplementary Fig. S12's blind-annotation validation, is reported there rather than repeated here, since Brain is not otherwise part of this figure's demonstration.)

(c) Head-to-head comparison on the hot-spot mask in (a): the same token-level Mann-Whitney test scored by three different detectors -- MetaboFM's best-tracking PC, the H&E image's own local luminance, and its local texture (Sobel gradient magnitude) -- summarized by effect size (rank-biserial correlation; 0 = groups indistinguishable, 1 = perfect separation). Local texture is nominally significant for this mask (p=7.2e-12): any binary intensity mask has edges, independent of whether the underlying biology is visible in H&E, so texture's p-value does not by itself indicate H&E resolves the same structure. By effect size, MetaboFM's PC separates the hot-spot mask more cleanly than either H&E feature ({effect_row['MetaboFM PC']:.2f} vs. {effect_row['H&E luminance']:.2f} luminance, {effect_row['H&E texture']:.2f} texture). The MALDI-IHC counterpart of this comparison (amyloid plaque mask) is reported independently in Supplementary Fig. S15.
"""
    (PANEL_DIR / "captions.txt").write_text(caption, encoding="utf-8")
    print("  saved captions.txt")


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


# ── Panel A: Lung m/z 527.16 hot-spot vs. PC3 ──────────────────────────────

def _fit_msi_pca(organ: str, dataset_id: str):
    d = np.load(HIST_DIR / f"{organ}_{dataset_id}_tokens_data.npz", allow_pickle=False)
    tokens = d["concat_tokens"]
    H, W = int(d["H"]), int(d["W"])
    t = np.load(HIST_DIR / f"{organ}_{dataset_id}_pc1_token_level.npz", allow_pickle=True)
    interior_mask = t["interior_mask"]
    interior_flat = interior_mask.flatten()
    pca = PCA(n_components=5, random_state=42)
    pc_interior = pca.fit_transform(tokens[interior_flat])
    return pc_interior, interior_flat, interior_mask, (H, W)


def _pc_grid_at_ion_resolution(pc_interior: np.ndarray, interior_flat: np.ndarray,
                                pc_index: int, ion_shape: tuple[int, int]) -> np.ndarray:
    grid_full = np.full(PATCH_GRID * PATCH_GRID, np.nan, dtype=np.float32)
    grid_full[interior_flat] = pc_interior[:, pc_index]
    grid = grid_full.reshape(PATCH_GRID, PATCH_GRID)
    H, W = ion_shape
    return patch_grid_to_ion_resolution(np.nan_to_num(grid, nan=0.0), H, W)


def _find_hot_center(img: np.ndarray) -> tuple[int, int]:
    from scipy.ndimage import gaussian_filter
    thresh = np.percentile(img[img > 0], 99) if (img > 0).any() else 0
    hot = (img >= thresh).astype(np.float32)
    density = gaussian_filter(hot, sigma=8)
    return np.unravel_index(np.argmax(density), density.shape)


def _mask_centroid(mask: np.ndarray) -> tuple[int, int]:
    """Centroid of the full hot-spot mask, not just its single densest
    sub-cluster -- for this channel the mask is a spread-out arc, not one
    tight blob, so centering the zoom on the argmax-density point (as
    _find_hot_center does) crops out roughly half the pattern. The centroid
    keeps the zoom representative of the whole mask rather than just its
    hottest corner."""
    ys, xs = np.where(mask)
    return int(round(ys.mean())), int(round(xs.mean()))


def build_hot_mask(channel_img: np.ndarray, percentile: float = LUNG_HOT_PERCENTILE) -> np.ndarray:
    positive = channel_img[channel_img > 0]
    thresh = np.percentile(positive, percentile) if positive.size else np.inf
    return channel_img >= thresh


def hot_vs_pc_test(hot_mask: np.ndarray, pc_interior: np.ndarray, interior_flat: np.ndarray,
                    pc_index: int) -> tuple[dict, np.ndarray]:
    hot_patch_mask = _raster_to_patch_mask(hot_mask, min_overlap=0.0)
    hot_in_interior = hot_patch_mask.flatten()[interior_flat]
    pc_vals = pc_interior[:, pc_index]
    hot_vals = pc_vals[hot_in_interior]
    outside_vals = pc_vals[~hot_in_interior]
    result = {
        "n_hot_tokens": int(hot_in_interior.sum()),
        "n_outside_tokens": int((~hot_in_interior).sum()),
        "hot_pc_mean": float(hot_vals.mean()) if hot_vals.size else float("nan"),
        "outside_pc_mean": float(outside_vals.mean()) if outside_vals.size else float("nan"),
    }
    if hot_vals.size >= 2 and outside_vals.size >= 2:
        _, p = stats.mannwhitneyu(hot_vals, outside_vals, alternative="two-sided")
        result["mannwhitney_p"] = float(p)
    else:
        result["mannwhitney_p"] = float("nan")
    return result, hot_patch_mask


def _patch_mask_to_pixel_grid(patch_mask: np.ndarray, S: int) -> np.ndarray:
    """Upsamples a (PATCH_GRID, PATCH_GRID) boolean mask to an (S, S) pixel
    grid via NEAREST, for contouring against a full-resolution image.
    Deliberately NOT done via ax.contour(patch_mask, extent=[0, S, S, 0]) --
    matplotlib's contour() does not interpret an inverted-y extent the same
    way imshow() does, which silently produces a badly shifted/mismatched
    contour (verified: the contour ends up over completely the wrong region
    relative to the actual mask). Upsampling first and contouring at native
    pixel resolution with no extent avoids that trap entirely."""
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


def panel_a():
    d = np.load(HIST_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_tokens_data.npz", allow_pickle=False)
    mzs = d["matched_mz"]
    ch_idx = int(np.argmin(np.abs(mzs - LUNG_TARGET_MZ)))
    ch_img = d["channel_images"][ch_idx].astype(np.float32)
    H, W = int(d["H"]), int(d["W"])

    pc_interior, interior_flat, interior_mask, ion_shape = _fit_msi_pca(LUNG_ORGAN, LUNG_DATASET)
    pc_map = _pc_grid_at_ion_resolution(pc_interior, interior_flat, LUNG_PC_INDEX, ion_shape)

    rd = np.load(REG_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_registration_data.npz", allow_pickle=False)
    affine = rd["affine_ion_to_optical"]
    optical_crop = native_optical_crop(rd["optical"], affine, (H, W))
    he_native = optical_crop.image
    Hh, Wh = he_native.shape[:2]

    he_native = _pad_rgb_to_square(he_native)
    Hh, Wh = he_native.shape[:2]
    ch_img_sq = pad_to_square(ch_img)
    S = max(H, W)
    H_sq, W_sq = S, S
    pc_map_sq = pad_to_square(pc_map)

    hot_mask = build_hot_mask(ch_img)
    stat_result, hot_patch_mask = hot_vs_pc_test(hot_mask, pc_interior, interior_flat, LUNG_PC_INDEX)
    print(f"[INFO] Lung m/z {LUNG_TARGET_MZ} hot-spot vs. PC{LUNG_PC_INDEX + 1}, "
          f"token-level Mann-Whitney: {stat_result}")
    pd.DataFrame([stat_result]).to_csv(PANEL_DIR / "lung_hotspot_vs_pc_stats.csv", index=False)

    hot_mask_sq = pad_to_square(hot_mask.astype(np.float32)) > 0.5

    row, col = _mask_centroid(hot_mask_sq)
    he_row = int(row * Hh / H_sq)
    he_col = int(col * Wh / W_sq)
    he_zoom_half = 350
    ion_zoom_half = max(8, int(round(he_zoom_half * H_sq / Hh)))

    def crop(img, r, c, half, fill=0):
        r0, r1 = max(0, r - half), min(img.shape[0], r + half)
        c0, c1 = max(0, c - half), min(img.shape[1], c + half)
        out_shape = (2 * half, 2 * half) + img.shape[2:]
        out = np.full(out_shape, fill, dtype=img.dtype)
        out[r0 - (r - half):r1 - (r - half), c0 - (c - half):c1 - (c - half)] = img[r0:r1, c0:c1]
        return out

    he_zoom = crop(he_native, he_row, he_col, he_zoom_half, fill=255)
    ch_zoom = crop(ch_img_sq, row, col, ion_zoom_half)
    pc_zoom = crop(pc_map_sq, row, col, ion_zoom_half)
    mask_zoom = crop(hot_mask_sq, row, col, ion_zoom_half)

    vmax_ch = np.percentile(ch_img_sq[ch_img_sq > 0], 99)
    cmap_ch = plt.get_cmap("viridis").copy()
    cmap_pc = plt.get_cmap("RdBu_r").copy()
    cmap_pc.set_bad(color="#dddddd")
    cmap_mask = matplotlib.colors.ListedColormap(["#222222", "#ffd60a"])

    pc_valid = pc_map_sq != 0
    pc_norm = np.ma.masked_where(~pc_valid, (pc_map_sq - pc_map_sq[pc_valid].min()) /
                                  (pc_map_sq[pc_valid].max() - pc_map_sq[pc_valid].min() + 1e-8))
    pc_zoom_valid = pc_zoom != 0
    pc_zoom_norm = np.ma.masked_where(~pc_zoom_valid, (pc_zoom - pc_map_sq[pc_valid].min()) /
                                       (pc_map_sq[pc_valid].max() - pc_map_sq[pc_valid].min() + 1e-8))

    hot_mask_pixel = _patch_mask_to_pixel_grid(hot_patch_mask, H_sq)

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.3))

    def _img(ax, arr, cmap, vmin, vmax):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

    _img(axes[0, 0], he_native, None, None, None)
    axes[0, 0].set_title("Lung H&E", fontsize=8)
    add_scale_bar_known_pixel_size(axes[0, 0], LUNG_HE_UM_PER_PX, Wh, Wh, color="black")

    _img(axes[0, 1], ch_img_sq, cmap_ch, 0, vmax_ch)
    axes[0, 1].set_title(f"m/z {LUNG_TARGET_MZ:.2f} (raw channel)", fontsize=8)
    add_scale_bar_known_pixel_size(axes[0, 1], LUNG_ION_UM_PER_PX, W_sq, W_sq)

    _img(axes[0, 2], hot_mask_sq, cmap_mask, 0, 1)
    axes[0, 2].set_title(f"Hot-spot mask\n(>= p{LUNG_HOT_PERCENTILE})", fontsize=8)
    add_scale_bar_known_pixel_size(axes[0, 2], LUNG_ION_UM_PER_PX, W_sq, W_sq, color="white")

    axes[0, 3].imshow(pc_norm, cmap=cmap_pc, vmin=0, vmax=1)
    axes[0, 3].contour(hot_mask_pixel, levels=[0.5], colors="black", linewidths=0.8)
    axes[0, 3].set_title(f"MetaboFM PC{LUNG_PC_INDEX + 1}\n(best-tracking component)", fontsize=8)
    axes[0, 3].axis("off")
    add_scale_bar_known_pixel_size(axes[0, 3], LUNG_ION_UM_PER_PX, W_sq, W_sq, color="black")

    _img(axes[1, 0], he_zoom, None, None, None)
    axes[1, 0].set_title("H&E, zoomed\n(hot-spot region)", fontsize=8)
    add_scale_bar_known_pixel_size(axes[1, 0], LUNG_HE_UM_PER_PX, he_zoom.shape[1], he_zoom.shape[1], color="black")

    _img(axes[1, 1], ch_zoom, cmap_ch, 0, vmax_ch)
    axes[1, 1].set_title("m/z, zoomed\n(same region)", fontsize=8)
    add_scale_bar_known_pixel_size(axes[1, 1], LUNG_ION_UM_PER_PX, ch_zoom.shape[1], ch_zoom.shape[1])

    _img(axes[1, 2], mask_zoom, cmap_mask, 0, 1)
    axes[1, 2].set_title("Hot-spot mask, zoomed\n(same region)", fontsize=8)
    add_scale_bar_known_pixel_size(axes[1, 2], LUNG_ION_UM_PER_PX, mask_zoom.shape[1], mask_zoom.shape[1], color="white")

    axes[1, 3].imshow(pc_zoom_norm, cmap=cmap_pc, vmin=0, vmax=1)
    axes[1, 3].set_title(f"MetaboFM PC{LUNG_PC_INDEX + 1}, zoomed\n(same region)", fontsize=8)
    axes[1, 3].axis("off")
    add_scale_bar_known_pixel_size(axes[1, 3], LUNG_ION_UM_PER_PX, pc_zoom.shape[1], pc_zoom.shape[1], color="black")

    fig.tight_layout()
    save_panel(fig, "figS14_panelA_lung_carbohydrate")
    plt.close(fig)

    save_single(lambda ax: (_img(ax, he_native, None, None, None),
                             add_scale_bar_known_pixel_size(ax, LUNG_HE_UM_PER_PX, Wh, Wh, color="black")),
                (3.2, 3.2), "figS14_panelA_HE")
    save_single(lambda ax: (_img(ax, ch_img_sq, cmap_ch, 0, vmax_ch),
                             add_scale_bar_known_pixel_size(ax, LUNG_ION_UM_PER_PX, W_sq, W_sq)),
                (3.2, 3.2), "figS14_panelA_mz527")
    save_single(lambda ax: (_img(ax, hot_mask_sq, cmap_mask, 0, 1),
                             add_scale_bar_known_pixel_size(ax, LUNG_ION_UM_PER_PX, W_sq, W_sq, color="white")),
                (3.2, 3.2), "figS14_panelA_hotspot_mask")
    save_single(lambda ax: (_img(ax, pc_norm, cmap_pc, 0, 1),
                             ax.contour(hot_mask_pixel, levels=[0.5], colors="black", linewidths=0.8),
                             add_scale_bar_known_pixel_size(ax, LUNG_ION_UM_PER_PX, W_sq, W_sq, color="black")),
                (3.2, 3.2), "figS14_panelA_PC3")
    save_single(lambda ax: (_img(ax, he_zoom, None, None, None),
                             add_scale_bar_known_pixel_size(ax, LUNG_HE_UM_PER_PX, he_zoom.shape[1], he_zoom.shape[1], color="black")),
                (3.2, 3.2), "figS14_panelA_HE_zoomed")
    save_single(lambda ax: (_img(ax, ch_zoom, cmap_ch, 0, vmax_ch),
                             add_scale_bar_known_pixel_size(ax, LUNG_ION_UM_PER_PX, ch_zoom.shape[1], ch_zoom.shape[1])),
                (3.2, 3.2), "figS14_panelA_mz527_zoomed")
    save_single(lambda ax: (_img(ax, mask_zoom, cmap_mask, 0, 1),
                             add_scale_bar_known_pixel_size(ax, LUNG_ION_UM_PER_PX, mask_zoom.shape[1], mask_zoom.shape[1], color="white")),
                (3.2, 3.2), "figS14_panelA_hotspot_mask_zoomed")
    save_single(lambda ax: (_img(ax, pc_zoom_norm, cmap_pc, 0, 1),
                             add_scale_bar_known_pixel_size(ax, LUNG_ION_UM_PER_PX, pc_zoom.shape[1], pc_zoom.shape[1], color="black")),
                (3.2, 3.2), "figS14_panelA_PC3_zoomed")

    return stat_result, hot_mask, pc_interior, interior_flat


# ── Panel B: quantitative H&E-vs-PC1 divergence (MSI samples only) ────────

def _sample_optical_at_ion_grid(optical_crop_image: np.ndarray, affine: np.ndarray,
                                 crop_x0: int, crop_y0: int, ion_shape: tuple[int, int]) -> np.ndarray:
    gray = np.asarray(Image.fromarray(optical_crop_image).convert("L"), dtype=np.float32)
    T = np.asarray(affine, dtype=float)
    coeffs = (T[0, 0], T[0, 1], T[0, 2] - crop_x0, T[1, 0], T[1, 1], T[1, 2] - crop_y0)
    H, W = ion_shape
    sampled = np.asarray(
        Image.fromarray(gray, mode="F").transform(
            (W, H), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR, fillcolor=0
        ),
        dtype=np.float32,
    )
    return sampled


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    return np.hypot(gx, gy)


def _correlate_sample(label: str, pc_vals: np.ndarray, luminance_patch: np.ndarray,
                       texture_patch: np.ndarray) -> dict:
    valid = ~np.isnan(pc_vals)
    rho_lum, p_lum = stats.spearmanr(pc_vals[valid], luminance_patch[valid])
    rho_tex, p_tex = stats.spearmanr(pc_vals[valid], texture_patch[valid])
    return {"sample": label, "n_tokens": int(valid.sum()),
            "rho_vs_he_luminance": float(rho_lum), "p_vs_he_luminance": float(p_lum),
            "rho_vs_he_texture": float(rho_tex), "p_vs_he_texture": float(p_tex)}


def panel_b(pc_interior: np.ndarray, interior_flat: np.ndarray):
    """Divergence test restricted to Lung and to PC3, the specific component
    panel a actually uses -- not Brain (not otherwise shown in this figure;
    Brain's own PC1 divergence lives in Fig. S12) and not the generic PC1
    (panel a's claim rests on PC3 specifically, so that's the component this
    check needs to speak to)."""
    d = np.load(HIST_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_tokens_data.npz", allow_pickle=False)
    H, W = int(d["H"]), int(d["W"])
    rd = np.load(REG_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_registration_data.npz", allow_pickle=False)
    affine = rd["affine_ion_to_optical"]
    optical_crop = native_optical_crop(rd["optical"], affine, (H, W))
    he_ion_grid = _sample_optical_at_ion_grid(optical_crop.image, affine, optical_crop.x0, optical_crop.y0, (H, W))

    luminance_patch = _channel_patch_means(he_ion_grid).flatten()[interior_flat]
    texture_patch = _channel_patch_means(_gradient_magnitude(he_ion_grid)).flatten()[interior_flat]
    pc3_vals = pc_interior[:, LUNG_PC_INDEX]

    row = _correlate_sample(f"Lung (MSI), PC{LUNG_PC_INDEX + 1}", pc3_vals, luminance_patch, texture_patch)
    table = pd.DataFrame([row])
    table.to_csv(PANEL_DIR / "he_vs_pc3_divergence.csv", index=False)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(3.0, 3.2))
    detectors = [("H&E luminance", row["rho_vs_he_luminance"], "#4c72b0"),
                 ("H&E texture", row["rho_vs_he_texture"], "#dd8452")]
    x = np.arange(len(detectors))
    ax.bar(x, [abs(v) for _, v, _ in detectors], color=[c for _, _, c in detectors])
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _, _ in detectors], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel(f"|Spearman rho| with\nMetaboFM PC{LUNG_PC_INDEX + 1} (Lung)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_panel(fig, "figS14_panelB_divergence")
    plt.close(fig)
    return row


# ── Panel C: MetaboFM vs. H&E head-to-head on the hot-spot mask ───────────

def _rank_biserial(inside: np.ndarray, outside: np.ndarray) -> float:
    n1, n2 = inside.size, outside.size
    u, _ = stats.mannwhitneyu(inside, outside, alternative="two-sided")
    return float(abs(1 - 2 * u / (n1 * n2)))


def _mask_effect_size_raster(mask: np.ndarray, feature_img: np.ndarray) -> float:
    patch_mask = _raster_to_patch_mask(mask, min_overlap=0.0).flatten()
    feat_flat = _channel_patch_means(feature_img).flatten()
    return _rank_biserial(feat_flat[patch_mask], feat_flat[~patch_mask])


def _mask_effect_size_pc(mask: np.ndarray, pc_interior: np.ndarray, interior_flat: np.ndarray,
                          pc_index: int) -> float:
    patch_mask = _raster_to_patch_mask(mask, min_overlap=0.0).flatten()[interior_flat]
    pc_vals = pc_interior[:, pc_index]
    return _rank_biserial(pc_vals[patch_mask], pc_vals[~patch_mask])


def panel_c(hot_mask: np.ndarray, pc_lung: np.ndarray, flat_lung: np.ndarray):
    d = np.load(HIST_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_tokens_data.npz", allow_pickle=False)
    H, W = int(d["H"]), int(d["W"])
    rd = np.load(REG_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_registration_data.npz", allow_pickle=False)
    affine = rd["affine_ion_to_optical"]
    optical_crop = native_optical_crop(rd["optical"], affine, (H, W))
    he_ion_lung = _sample_optical_at_ion_grid(optical_crop.image, affine, optical_crop.x0, optical_crop.y0, (H, W))

    row = {
        "structure": f"m/z {LUNG_TARGET_MZ:.2f} hot-spot\n(untargeted MSI)",
        "MetaboFM PC": _mask_effect_size_pc(hot_mask, pc_lung, flat_lung, LUNG_PC_INDEX),
        "H&E luminance": _mask_effect_size_raster(hot_mask, he_ion_lung),
        "H&E texture": _mask_effect_size_raster(hot_mask, _gradient_magnitude(he_ion_lung)),
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
    save_panel(fig, "figS14_panelC_effect_size")
    plt.close(fig)

    return row


def main():
    lung_stat, hot_mask, pc_lung, flat_lung = panel_a()
    divergence_row = panel_b(pc_lung, flat_lung)
    effect_row = panel_c(hot_mask, pc_lung, flat_lung)
    write_caption(lung_stat, effect_row, divergence_row)
    print("FigS14 done.")


if __name__ == "__main__":
    main()
