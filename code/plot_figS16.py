"""
plot_figS16.py
--------------
Supplementary Figure S16: MetaboFM vs. H&E blind validation, MALDI-IHC
-- the MALDI-IHC counterpart of Supplementary
Fig. S12's untargeted-MSI blind validation, split into its own figure
because it is a self-contained, independent test (does MetaboFM's PC1
track real anatomy in this modality at all, not just the amyloid-specific
signal in Supplementary Fig. S15) rather than a sub-part of the amyloid
demonstration -- see this repo's CLAUDE.md's "H&E / optical-image comparison
pipeline" section.

Data sources (see embed_ihc_histology_comparison.py for how these
were produced):
  - BrainIHC alz / wt: MALDI-IHC (mass-tag antibody) mouse brain (protein
    markers), regions annotated as ImageJ .roi files in
    data_external/miralys_mb_ihc/.
  - outputs/optical_images/annotation_overlay/all_regions_annotation_vs_pc1_fdr.csv:
    the combined, BH-FDR-corrected region-vs-outside-tissue test across all
    4 samples / 27 regions (both modalities); this script uses only the
    BrainIHC_alz / BrainIHC_wt rows.

All region annotations were drawn blind to any MALDI-IHC channel, from the
H&E image alone -- the entire logic of the test depends on this; state it
explicitly in the caption.

Panels:
  A  Small multiples: H&E | MetaboFM Stage-1 PC1 map (with annotated region
     outlines) for each of the 2 MALDI-IHC conditions (Alzheimer's model,
     wild-type).
  B  Summary: effect size vs. -log10(FDR q) for all IHC regions, highlighting
     the regions significant at FDR q<0.05.

Usage:
  python plot_figS16.py   (base conda env -- matplotlib/scipy only, no metaspace/GPU)
"""

from __future__ import annotations

from pathlib import Path
from metabofm_paths import METABOFM_ROOT, IHC_RAW_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from plot_utils import set_nature_style, add_scale_bar_known_pixel_size
from embed_histology_comparison import patch_grid_to_ion_resolution

set_nature_style()

HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
OVERLAY_DIR = METABOFM_ROOT / "outputs/optical_images/annotation_overlay"
IHC_DATA_ROOT = IHC_RAW_DIR
FDR_CSV = OVERLAY_DIR / "all_regions_annotation_vs_pc1_fdr.csv"

OUT_DIR = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS16_ihc_validation_summary"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
HE_UM_PER_PX = 2.6
THUMB_MAX = 1200

IHC_SAMPLE_LABELS = {
    "BrainIHC_alz": "Brain MALDI-IHC (Alzheimer's model)",
    "BrainIHC_wt": "Brain MALDI-IHC (wild-type)",
}


def write_caption(n_sig: int, n_total: int):
    caption = f"""\
Supplementary Figure 16 | MetaboFM vs. H&E blind validation, MALDI-IHC.

(a) For each of 2 independent MALDI-IHC conditions (Alzheimer's model, wild-type; targeted protein-marker panel) -- the registered H&E image (left) and MetaboFM Stage 1's first principal component (PC1) computed over interior tissue tokens only (right), with hand-drawn anatomical region outlines overlaid in green. Every region was annotated directly from the H&E image alone, blind to any MALDI-IHC channel, before any MetaboFM output was consulted -- this blinding is what makes the region-vs-surrounding-tissue comparison in (b) a genuine test of whether MetaboFM recovers real anatomy, not a circular one. This is a self-contained test of MetaboFM's spatial embedding in this modality, independent of the amyloid-specific demonstration in Supplementary Fig. S15. The same validation in the untargeted MSI modality used throughout this study is Supplementary Fig. S12.

(b) Effect size (PC1 mean inside the annotated region minus PC1 mean in the rest of the interior tissue) plotted against statistical significance (-log10 of the Benjamini-Hochberg FDR-corrected q-value, corrected jointly with the MSI regions in Supplementary Fig. S12 across all 27 regions from both modalities) for the {n_total} MALDI-IHC regions in (a), from a token-level (one 8x8 patch = one independent sample) two-sided Mann-Whitney U test. {n_sig}/{n_total} regions are significant at FDR q<0.05 (filled markers, dashed line), spanning both conditions.
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


def _thumb(img: np.ndarray, max_side: int = THUMB_MAX):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = np.array(Image.fromarray(img).resize(new_size, Image.BILINEAR))
    return img, scale


def load_ihc_panel(condition: str):
    cond_dir = IHC_DATA_ROOT / condition
    he_full = tifffile.imread(cond_dir / "he_resized_affine.tif")
    he_thumb, scale = _thumb(he_full)

    t = np.load(HIST_DIR / f"BrainIHC_{condition}_pc1_token_level.npz", allow_pickle=False)
    pc1_grid, interior_mask = t["pc1_grid"], t["interior_mask"]
    valid_grid = ~np.isnan(pc1_grid) & interior_mask

    d = np.load(HIST_DIR / f"BrainIHC_{condition}_tokens_data.npz", allow_pickle=False)
    H, W = int(d["H"]), int(d["W"])
    pc1_ion_res = patch_grid_to_ion_resolution(np.nan_to_num(pc1_grid, nan=0.0), H, W)
    valid_ion_res = patch_grid_to_ion_resolution(valid_grid.astype(np.float32), H, W) > 0.5
    pc1_thumb = np.array(Image.fromarray(pc1_ion_res).resize(
        (he_thumb.shape[1], he_thumb.shape[0]), Image.NEAREST))
    valid_thumb = np.array(Image.fromarray(valid_ion_res.astype(np.uint8) * 255).resize(
        (he_thumb.shape[1], he_thumb.shape[0]), Image.NEAREST)) > 127

    import roifile
    region_polys = []
    for roi_path in sorted(cond_dir.glob(f"{condition}_roi*.roi")):
        roi = roifile.ImagejRoi.fromfile(str(roi_path))
        region_polys.append(roi.coordinates().astype(np.float64) * scale)
    return he_thumb, pc1_thumb, valid_thumb, region_polys, scale


def _square_he_pc1(he, pc1, valid, polys):
    H, W = he.shape[:2]
    S = max(H, W)
    top, left = (S - H) // 2, (S - W) // 2
    he_sq = np.full((S, S, he.shape[2]), 255, dtype=he.dtype)
    he_sq[top:top + H, left:left + W] = he
    pc1_sq = np.zeros((S, S), dtype=pc1.dtype)
    pc1_sq[top:top + H, left:left + W] = pc1
    valid_sq = np.zeros((S, S), dtype=bool)
    valid_sq[top:top + H, left:left + W] = valid
    polys_sq = [poly + np.array([left, top]) for poly in polys]
    return he_sq, pc1_sq, valid_sq, polys_sq


def _draw_he(ax, he_thumb, region_polys, um_per_display_px, display_width, with_title=None):
    ax.imshow(he_thumb); ax.axis("off")
    if with_title is not None:
        ax.set_title(with_title, fontsize=8, loc="left")
    for poly in region_polys:
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#2ca02c", linewidth=1.0)
    add_scale_bar_known_pixel_size(ax, um_per_native_px=um_per_display_px,
                                    native_width_px=display_width, display_width_px=display_width, color="black")


def _draw_pc1(ax, pc1_thumb, valid_thumb, region_polys, um_per_display_px, display_width, with_title=None):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    valid = valid_thumb & np.isfinite(pc1_thumb)
    vals = pc1_thumb[valid]
    norm = (pc1_thumb - vals.min()) / (vals.max() - vals.min() + 1e-8) if vals.size else pc1_thumb
    display = np.ma.masked_where(~valid, norm)
    ax.imshow(display, cmap=cmap, vmin=0, vmax=1); ax.axis("off")
    if with_title is not None:
        ax.set_title(with_title, fontsize=8, loc="left")
    for poly in region_polys:
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#2ca02c", linewidth=1.0)
    add_scale_bar_known_pixel_size(ax, um_per_native_px=um_per_display_px,
                                    native_width_px=display_width, display_width_px=display_width, color="black")


def panel_a():
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.8))
    for i, condition in enumerate(["alz", "wt"]):
        sample = f"BrainIHC_{condition}"
        he, pc1, valid, polys, scale = load_ihc_panel(condition)
        he, pc1, valid, polys = _square_he_pc1(he, pc1, valid, polys)
        um_per_display_px = HE_UM_PER_PX / scale
        display_width = he.shape[1]
        _draw_he(axes[i][0], he, polys, um_per_display_px, display_width, with_title=IHC_SAMPLE_LABELS[sample])
        _draw_pc1(axes[i][1], pc1, valid, polys, um_per_display_px, display_width, with_title="MetaboFM Stage 1 PC1")
        save_single(lambda ax, he=he, polys=polys, u=um_per_display_px, w=display_width:
                     _draw_he(ax, he, polys, u, w), (3.2, 3.2), f"figS16_panelA_{sample}_HE")
        save_single(lambda ax, pc1=pc1, valid=valid, polys=polys, u=um_per_display_px, w=display_width:
                     _draw_pc1(ax, pc1, valid, polys, u, w), (3.2, 3.2), f"figS16_panelA_{sample}_PC1")
    fig.tight_layout()
    save_panel(fig, "figS16_panelA_samples")
    plt.close(fig)

    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 2.2))
    cmap = plt.get_cmap("RdBu_r").copy()
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=cmap), cax=ax_cb)
    cb.set_label("MetaboFM PC1\n(min-max normalized per sample)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout()
    save_panel(fig_cb, "figS16_panelA_colorbar")
    plt.close(fig_cb)


def panel_b():
    df = pd.read_csv(FDR_CSV)
    df = df[df["sample"].isin(["BrainIHC_alz", "BrainIHC_wt"])].copy()
    df["effect_size"] = df["region_pc1_mean"] - df["outside_pc1_mean"]
    df["neg_log10_q"] = -np.log10(df["fdr_q"].clip(lower=1e-300))
    sample_colors = {"BrainIHC_alz": "#9467bd", "BrainIHC_wt": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for sample, group in df.groupby("sample"):
        sig = group["significant_fdr05"]
        ax.scatter(group.loc[sig, "effect_size"], group.loc[sig, "neg_log10_q"],
                   color=sample_colors[sample], marker="o", s=36,
                   edgecolor="black", linewidth=0.5, label=IHC_SAMPLE_LABELS[sample], zorder=3)
        ax.scatter(group.loc[~sig, "effect_size"], group.loc[~sig, "neg_log10_q"],
                   facecolor="none", edgecolor=sample_colors[sample], marker="o", s=30,
                   linewidth=0.8, zorder=2)
    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(ax.get_xlim()[1], -np.log10(0.05), " FDR q=0.05", fontsize=6, va="bottom", ha="right", color="gray")
    ax.set_xlabel("Effect size (region PC1 mean - outside-tissue PC1 mean)", fontsize=8)
    ax.set_ylabel(r"$-\log_{10}$(FDR q-value)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_panel(fig, "figS16_panelB_summary")
    plt.close(fig)

    return int(df["significant_fdr05"].sum()), int(len(df))


def main():
    panel_a()
    n_sig, n_total = panel_b()
    write_caption(n_sig, n_total)
    print("FigS16 done.")


if __name__ == "__main__":
    main()
