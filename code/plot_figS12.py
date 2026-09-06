"""
plot_figS12.py
--------------
Supplementary Figure S12: MetaboFM vs. H&E blind validation, untargeted MSI
(the manuscript's H&E-comparison analysis) -- does MetaboFM's learned spatial structure track
real, blind-H&E-annotated anatomy, without ever having seen the H&E image, in
the untargeted MSI modality used throughout this study? MALDI-IHC (targeted)
gets the same treatment, as its own independent analysis, in plot_figS15.py
-- see that script and this repo's CLAUDE.md's "H&E / optical-image comparison
pipeline" section for why the modalities are split into separate figures
rather than interleaved. FDR correction is still computed jointly across all
27 regions from both modalities (see FDR_CSV below), even though the two
figures present their own regions separately.

Data sources (see embed_histology_comparison.py for how these were
produced):
  - Brain, Lung: METASPACE MSI, MSM-ranked metabolite channels, region
    annotated in outputs/optical_images/annotations/*_regions.geojson.
  - outputs/optical_images/annotation_overlay/all_regions_annotation_vs_pc1_fdr.csv:
    the combined, BH-FDR-corrected region-vs-outside-tissue test across all
    4 samples / 27 regions (built in an earlier chat session); this script
    uses only the Brain/Lung (MSI) rows.

All region annotations were drawn blind to any MSI channel, from the H&E
image alone -- the entire logic of the test depends on this; state it
explicitly in the caption.

Panels:
  A  Small multiples: H&E | MetaboFM Stage-1 PC1 map (with annotated region
     outlines) for each of the 2 MSI samples (Brain, Lung).
  B  Token-level PC1 distributions (region vs. surrounding tissue) for each
     of the 3 MSI regions, with FDR q-values annotated (FDR computed jointly
     across all 27 regions from both modalities, filtered here to the MSI
     rows).

Usage:
  python plot_figS12.py   (base conda env -- matplotlib/scipy only, no metaspace/GPU)
"""

from __future__ import annotations

from pathlib import Path
from metabofm_paths import METABOFM_ROOT, IHC_RAW_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from plot_utils import set_nature_style, add_scale_bar_known_pixel_size
from embed_histology_comparison import patch_grid_to_ion_resolution

set_nature_style()

# Native-resolution physical pixel sizes (um/px), for scale bars. METASPACE
# organs: derived from the live-fetched MSI Pixel_Size (Brain=100, Lung=35
# um/px ion grid) divided by the registration affine's ion-to-optical scale
# factor. MALDI-IHC: given directly (H&E scanner=2.6 um/px, MALDI raster
# step=20 um/px) -- no METASPACE record exists for these samples.
NATIVE_OPTICAL_UM_PER_PX = {
    "Brain": 2.413409356838321,
    "Lung": 3.7566070165424823,
    "BrainIHC_alz": 2.6,
    "BrainIHC_wt": 2.6,
}
MALDI_ION_UM_PER_PX = 20.0

# -- CONFIG -------------------------------------------------------------------
HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
ANNOT_DIR = METABOFM_ROOT / "outputs/optical_images/annotations"
OVERLAY_DIR = METABOFM_ROOT / "outputs/optical_images/annotation_overlay"
IHC_DATA_ROOT = IHC_RAW_DIR
FDR_CSV = OVERLAY_DIR / "all_regions_annotation_vs_pc1_fdr.csv"

OUT_DIR = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS12_he_validation_summary"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
THUMB_MAX = 1200  # display-only downsize for the H&E panel

METASPACE_SAMPLES = {
    "Brain": "2019-11-25_17h14m31s",
    "Lung": "2023-06-27_22h58m39s",
}

SAMPLE_ORDER = ["Brain", "Lung"]
SAMPLE_LABELS = {
    "Brain": "Brain (MSI)",
    "Lung": "Lung (MSI)",
}


def write_caption(n_sig: int, n_total: int):
    caption = f"""\
Supplementary Figure 12 | MetaboFM vs. H&E blind validation, untargeted MSI.

(a) For each of 2 independent METASPACE MSI samples (Brain, Lung; MSM-ranked metabolite channels) -- the registered H&E image (left) and MetaboFM Stage 1's first principal component (PC1) computed over interior tissue tokens only (right), with hand-drawn anatomical region outlines overlaid in green, each numbered to match its label in (b) (Lung has 2 annotated regions, Brain has 1). Every region was annotated directly from the H&E image alone, blind to the MSI channels, before any MetaboFM output was consulted -- this blinding is what makes the region-vs-surrounding-tissue comparison in (b) a genuine test of whether MetaboFM recovers real anatomy, not a circular one. The same validation is repeated in a second, targeted imaging modality (MALDI-IHC) in Supplementary Fig. S15.

(b) Token-level PC1 distributions for each of the {n_total} MSI regions in (a) (one 8x8 patch = one independent sample), dim points/box = the rest of the interior tissue, bold points/box = inside the annotated region, from a two-sided Mann-Whitney U test (Benjamini-Hochberg FDR-corrected jointly across all 27 regions from both modalities; q-values annotated above each region). {n_sig}/{n_total} MSI regions are significant at FDR q<0.05, spanning both organs.
"""
    (PANEL_DIR / "captions.txt").write_text(caption, encoding="utf-8")
    print("  saved captions.txt")


def save_panel(fig, stem):
    for ax in fig.get_axes():
        ax.set_title(ax.get_title())  # keep sample titles in this combined panel
    fig.suptitle("")
    path = PANEL_DIR / stem
    fig.savefig(str(path) + ".svg", bbox_inches="tight", pad_inches=0.05, dpi=DPI)
    print(f"  saved panel {stem}.svg")


def _thumb(img: np.ndarray, max_side: int = THUMB_MAX) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = np.array(Image.fromarray(img).resize(new_size, Image.BILINEAR))
    return img, scale


def load_metaspace_panel(organ: str, dataset_id: str):
    """H&E native crop (downsized for display) + PC1 native map + region
    polygons (already in native-optical-crop pixel space, from the GeoJSON --
    no coordinate transform needed here, unlike the MALDI-IHC samples)."""
    stem = f"{organ}_{dataset_id}"
    he_path = HIST_DIR / "native_optical_panels" / f"{stem}_HE_native.png"
    he_full = np.array(Image.open(he_path))
    he_thumb, scale = _thumb(he_full)

    m = np.load(HIST_DIR / f"{stem}_pc1_native_map.npz", allow_pickle=False)
    pc1, valid = m["pc1"], m["valid"]
    pc1_thumb = np.array(Image.fromarray(pc1).resize(
        (he_thumb.shape[1], he_thumb.shape[0]), Image.NEAREST))
    valid_thumb = np.array(Image.fromarray(valid.astype(np.uint8) * 255).resize(
        (he_thumb.shape[1], he_thumb.shape[0]), Image.NEAREST)) > 127

    import json
    fc = json.loads((ANNOT_DIR / f"{stem}_regions.geojson").read_text(encoding="utf-8"))
    region_feats = [f for f in fc["features"] if f["properties"]["classification"] == "anatomical_region"]
    region_polys = [
        (np.asarray(f["geometry"]["coordinates"][0], dtype=np.float64) * scale)
        for f in region_feats
    ]
    # same name/order convention as embed_histology_comparison.py's
    # load_annotated_tissue_patch_mask -- so a label here matches the
    # "region_1"/"region_2" identity used in panel b and the FDR CSV.
    region_names = [f["properties"]["region_name"] or "region" for f in region_feats]
    return he_thumb, pc1_thumb, valid_thumb, region_polys, region_names, scale


def _region_short_label(sample: str, region_name: str) -> str:
    """'region_1' -> '1', for a compact on-image label; only needed when a
    sample has more than one region (Lung), so a single-region sample
    (Brain) isn't cluttered with a redundant '1'."""
    return region_name.rsplit("_", 1)[-1]


def _draw_he(ax, he_thumb, region_polys, region_names, sample, um_per_display_px, display_width, with_title=None):
    ax.imshow(he_thumb)
    ax.axis("off")
    if with_title is not None:
        ax.set_title(with_title, fontsize=8, loc="left")
    for poly, name in zip(region_polys, region_names):
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#2ca02c", linewidth=1.0)
        cx, cy = poly[:, 0].max(), poly[:, 1].min()
        ax.annotate(_region_short_label(sample, name), (cx, cy), color="#2ca02c", fontsize=7,
                    fontweight="bold", ha="left", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    add_scale_bar_known_pixel_size(ax, um_per_native_px=um_per_display_px,
                                    native_width_px=display_width, display_width_px=display_width, color="black")


def _draw_pc1(ax, pc1_thumb, valid_thumb, region_polys, region_names, sample, um_per_display_px, display_width, with_title=None):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    valid = valid_thumb & np.isfinite(pc1_thumb)
    vals = pc1_thumb[valid]
    if vals.size:
        vmin, vmax = vals.min(), vals.max()
        norm = (pc1_thumb - vmin) / (vmax - vmin + 1e-8)
    else:
        norm = pc1_thumb
    display = np.ma.masked_where(~valid, norm)
    ax.imshow(display, cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    if with_title is not None:
        ax.set_title(with_title, fontsize=8, loc="left")
    for poly, name in zip(region_polys, region_names):
        closed = np.vstack([poly, poly[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#2ca02c", linewidth=1.0)
        cx, cy = poly[:, 0].max(), poly[:, 1].min()
        ax.annotate(_region_short_label(sample, name), (cx, cy), color="#2ca02c", fontsize=7,
                    fontweight="bold", ha="left", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    add_scale_bar_known_pixel_size(ax, um_per_native_px=um_per_display_px,
                                    native_width_px=display_width, display_width_px=display_width, color="black")


def save_single(draw_fn, figsize, stem):
    """Saves one standalone image (no title, no padding) -- per-panel export
    convention, so every sub-image of a combined grid can be repositioned
    independently in PowerPoint."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.savefig(str(PANEL_DIR / stem) + ".svg", bbox_inches="tight", pad_inches=0, dpi=DPI)
    plt.close(fig)
    print(f"  saved individual panel {stem}.svg")


def _square_he_pc1(he, pc1, valid, polys):
    """Pads H&E (white fill, matching real slide background) and PC1/valid
    (zero fill -- masked/invalid, same as everywhere else) to a shared
    square, centered the same way pad_to_square() pads for the encoder, and
    offsets the region-outline polygons to match. Squaring both sides this
    way keeps real-content alignment intact -- it only adds a matching
    blank border around each, it doesn't stretch either to fit the other."""
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


def panel_a():
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.8))
    for i, sample in enumerate(SAMPLE_ORDER):
        he, pc1, valid, polys, region_names, scale = load_metaspace_panel(sample, METASPACE_SAMPLES[sample])
        he, pc1, valid, polys = _square_he_pc1(he, pc1, valid, polys)
        um_per_display_px = NATIVE_OPTICAL_UM_PER_PX[sample] / scale
        display_width = he.shape[1]

        _draw_he(axes[i][0], he, polys, region_names, sample, um_per_display_px, display_width,
                  with_title=SAMPLE_LABELS[sample])
        _draw_pc1(axes[i][1], pc1, valid, polys, region_names, sample, um_per_display_px, display_width,
                   with_title="MetaboFM Stage 1 PC1")

        save_single(lambda ax, he=he, polys=polys, rn=region_names, s=sample, u=um_per_display_px, w=display_width:
                     _draw_he(ax, he, polys, rn, s, u, w), (3.2, 3.2), f"figS12_panelA_{sample}_HE")
        save_single(lambda ax, pc1=pc1, valid=valid, polys=polys, rn=region_names, s=sample, u=um_per_display_px, w=display_width:
                     _draw_pc1(ax, pc1, valid, polys, rn, s, u, w), (3.2, 3.2), f"figS12_panelA_{sample}_PC1")
    fig.tight_layout()
    save_panel(fig, "figS12_panelA_samples")
    plt.close(fig)

    # standalone colorbar for the per-sample min-max-normalized PC1 scale
    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 2.2))
    cmap = plt.get_cmap("RdBu_r").copy()
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb)
    cb.set_label("MetaboFM PC1\n(min-max normalized per sample)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout()
    save_panel(fig_cb, "figS12_panelA_colorbar")
    plt.close(fig_cb)


def _region_token_values(sample: str, dataset_id: str):
    """Token-level PC1 values split by region (one array per annotated
    region) vs. the rest of the interior tissue -- the raw distributions the
    Mann-Whitney test in panel b actually compares. With only 3 MSI regions
    total, a scatter of 3 summary points is too sparse to be a compelling
    panel on its own; showing the real per-token distributions instead is
    both more honest (it's the actual data behind the test, not an
    abstraction of it) and visually substantive."""
    t = np.load(HIST_DIR / f"{sample}_{dataset_id}_pc1_token_level.npz", allow_pickle=True)
    pc1_grid, interior_mask = t["pc1_grid"], t["interior_mask"]
    region_names = [str(n) for n in t["region_names"]]
    region_masks = t["region_masks"]
    valid = ~np.isnan(pc1_grid) & interior_mask
    in_any_region = np.zeros_like(valid)
    for i in range(len(region_names)):
        in_any_region |= region_masks[i].astype(bool)
    outside_vals = pc1_grid[valid & ~in_any_region]
    out = {}
    for i, name in enumerate(region_names):
        region_valid = region_masks[i].astype(bool) & valid
        out[f"{sample} {name}"] = (pc1_grid[region_valid], outside_vals)
    return out


def panel_b():
    df = pd.read_csv(FDR_CSV)
    df = df[df["sample"].isin(SAMPLE_ORDER)].copy()

    groups = {}
    for sample, dataset_id in METASPACE_SAMPLES.items():
        groups.update(_region_token_values(sample, dataset_id))

    labels = list(groups.keys())
    sample_colors = {"Brain": "#1f77b4", "Lung": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    rng = np.random.default_rng(42)
    positions = np.arange(len(labels))
    box_data = []
    for i, label in enumerate(labels):
        region_vals, outside_vals = groups[label]
        sample = label.split(" ")[0]
        color = sample_colors[sample]
        for offset, (vals, alpha, jitter_w) in enumerate([
            (outside_vals, 0.25, 0.14),
            (region_vals, 0.9, 0.14),
        ]):
            x = positions[i] + (offset - 0.5) * 0.42 + rng.uniform(-jitter_w, jitter_w, size=vals.size)
            ax.scatter(x, vals, s=6, color=color, alpha=alpha, linewidth=0, zorder=2)
        box_data.append((positions[i] - 0.21, outside_vals))
        box_data.append((positions[i] + 0.21, region_vals))

    bp = ax.boxplot([d for _, d in box_data], positions=[p for p, _ in box_data], widths=0.3,
                     showfliers=False, patch_artist=True, zorder=3,
                     boxprops=dict(facecolor="none", linewidth=1.0),
                     medianprops=dict(color="black", linewidth=1.2),
                     whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))

    ax.set_ylabel("MetaboFM Stage 1 PC1\n(token-level, interior tissue)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.margins(y=0.22)
    ymax = ax.get_ylim()[1]

    row_by_label = {f"{r['sample']} {r['region_name']}": r for _, r in df.iterrows()}
    for i, label in enumerate(labels):
        row = row_by_label[label]
        q = row["fdr_q"]
        p_str = f"q={q:.1e}" if q >= 1e-4 else f"q<10$^{{{int(np.ceil(np.log10(q)))}}}$"
        ax.text(positions[i], ymax, p_str, fontsize=6.5, ha="center", va="bottom")
        ax.text(positions[i] - 0.21, ymax * 0.88, "outside", fontsize=6, ha="center", color="#555555")
        ax.text(positions[i] + 0.21, ymax * 0.88, "region", fontsize=6, ha="center", fontweight="bold", color="#333333")

    ax.set_xticks(positions)
    ax.set_xticklabels([l.replace("_", " ") for l in labels], fontsize=7)

    fig.tight_layout()
    save_panel(fig, "figS12_panelB_summary")
    plt.close(fig)

    return int(df["significant_fdr05"].sum()), int(len(df))


def main():
    panel_a()
    n_sig, n_total = panel_b()
    write_caption(n_sig, n_total)
    print("FigS12 done.")


if __name__ == "__main__":
    main()
