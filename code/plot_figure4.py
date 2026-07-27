"""
plot_figure4.py
---------------
Figure 4: Spatial patch representations of MetaboFM Stage 1.

Panels:
  A  ROI-annotated PCA spatial maps  -  Lymph node / Brain / Liver
     3 columns per sample: ion image | PC1 spatial map | annotated ROI overlay
     Source: spatial_top3ch_pca_*.png from probe_resnet_umap.py (best channel row)
  B  Spatial contiguity distribution across all 5,600 samples (unchanged)
  C  Within- vs between-organ cosine similarity (unchanged)

Supplementary figures:
  S6a  Percentile grid of spatial coherence (existing figS6_spatial_samples)
  S6b  Unsupervised K-means metabolic microregion maps (moved from old Panel A)

Usage:
  conda run -n torch_gpu --no-capture-output python -u code_v2/plot_figure4.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import textwrap as _tw

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from plot_utils import set_nature_style, draw_pipeline_diagram
set_nature_style()

# â"€â"€ CONFIG â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

RUN_DIR     = METABOFM_ROOT / "metabofm_v2/stage1_resnet/run_20260708_181629"
SPATIAL_DIR = METABOFM_ROOT / "outputs/spatial_patches"
MOL_DIR     = METABOFM_ROOT / "outputs/molecule_spatial_consistency"
OUT_DIR     = METABOFM_ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR   = OUT_DIR / "figure4"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

# Samples for Panel A  -  (display label, tiff stem prefix to match)
PANEL_A_SAMPLES = [
    ("Lymph node", "2021-06-30_20h06m19s__Lymph_no"),
    ("Brain",      "2023-10-02_17h16m22s__Brain"),
    ("Liver",      "2025-12-05_00h57m15s__Liver"),
    ("Stomach",    "2023-11-27_04h09m07s__Stomach"),
]


ORGANS_B = ["Kidney", "Brain", "Lung", "Liver", "Skin", "Breast"]
RANDOM_BASELINE = 0.167

COLOR_PRIMARY = "#2166ac"
COLOR_WITHIN  = "#2166ac"
COLOR_BETWEEN = "#d6604d"


# â"€â"€ HELPERS â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def save_fig(fig, stem):
    for ext in ("svg", "png"):
        fig.savefig(str(OUT_DIR / f"{stem}.{ext}"), dpi=DPI, bbox_inches="tight")
    print(f"  saved {stem}")

def save_panel(fig, stem):
    """Save individual panel as SVG without titles or padding."""
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    fig.savefig(str(PANEL_DIR / f"{stem}.svg"), bbox_inches="tight", pad_inches=0)
    print(f"  saved panel {stem}.svg")


def load_arrays(stem_prefix: str) -> dict:
    """Load saved .npz arrays from probe_resnet_umap for a given sample stem prefix."""
    matches = sorted(RUN_DIR.glob(f"arrays_{stem_prefix}*encoder_final.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No arrays .npz found for prefix '{stem_prefix}' in {RUN_DIR}\n"
            f"Re-run probe_resnet_umap.py with the new checkpoint.")
    return dict(np.load(str(matches[0])))


# â"€â"€ PANEL A: ROI-annotated PCA spatial maps â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _to_display(arr2d: np.ndarray) -> np.ndarray:
    return arr2d.astype(np.float32)


def _mask_to_display(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def draw_panel_a(fig, gs_cell):
    COL_LABELS = ["Ion image\n(best channel)", "PC1 spatial map", "UMAP-1 spatial map", "Annotated ROI"]
    CMAP_ION   = "viridis"
    CMAP_PC1   = "RdBu_r"
    CMAP_UMAP1 = "RdBu_r"

    data = [(label, load_arrays(stem)) for label, stem in PANEL_A_SAMPLES]
    n    = len(data)

    # Row heights proportional to each image's H/W ratio so no distortion
    h_ratios = [arr["ion_image"].shape[0] / arr["ion_image"].shape[1]
                for _, arr in data]

    # 5 columns: ion | pc1 | umap1 | roi | narrow cbar slot
    gs = gs_cell.subgridspec(n, 5, hspace=0.06, wspace=0.04,
                             width_ratios=[1, 1, 1, 1, 0.06],
                             height_ratios=h_ratios)

    last_im = None

    for row_i, (label, arr) in enumerate(data):
        # resize to common display size so all cells fill equally
        ion     = _to_display(arr["ion_image"])
        pc1     = _to_display(arr["pc1_map"])
        umap1   = _to_display(arr["umap1_map"])
        mask    = _mask_to_display(arr["roi_mask"].astype(bool))
        score   = float(arr["best_score"])
        best_ch = int(arr["best_ch"])

        slug = label.lower().replace(" ", "_")

        # Col 0: ion image
        ax0 = fig.add_subplot(gs[row_i, 0])
        ax0.imshow(ion, cmap=CMAP_ION, aspect="equal", interpolation="antialiased")
        ax0.axis("off")
        ax0.text(0.03, 0.97, f"Ch {best_ch}", transform=ax0.transAxes,
                 fontsize=7, va="top", color="white",
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))
        if row_i == 0:
            ax0.text(0.5, 1.01, COL_LABELS[0], transform=ax0.transAxes,
                     fontsize=9, ha="center", va="bottom")
        ax0.set_ylabel(label, fontsize=10, fontweight="bold")
        # save individual ion image
        _f, _a = plt.subplots(figsize=(3, 3 * ion.shape[0]/ion.shape[1]))
        _a.imshow(ion, cmap=CMAP_ION, aspect="equal", interpolation="antialiased")
        _a.axis("off")
        _f.savefig(str(PANEL_DIR / f"figure4_panelB_{slug}_ion.svg"), bbox_inches="tight", pad_inches=0)
        plt.close(_f)
        print(f"  saved panel figure4_panelB_{slug}_ion.svg")

        # Col 1: PC1 spatial map
        ax1 = fig.add_subplot(gs[row_i, 1])
        im  = ax1.imshow(pc1, cmap=CMAP_PC1, vmin=0, vmax=1,
                         aspect="equal", interpolation="antialiased")
        ax1.axis("off")
        if row_i == 0:
            ax1.text(0.5, 1.01, COL_LABELS[1], transform=ax1.transAxes,
                     fontsize=9, ha="center", va="bottom")
        last_im = im
        # save individual PC1 map
        _f, _a = plt.subplots(figsize=(3, 3 * pc1.shape[0]/pc1.shape[1]))
        _a.imshow(pc1, cmap=CMAP_PC1, vmin=0, vmax=1, aspect="equal", interpolation="antialiased")
        _a.axis("off")
        _f.savefig(str(PANEL_DIR / f"figure4_panelB_{slug}_pc1.svg"), bbox_inches="tight", pad_inches=0)
        plt.close(_f)
        print(f"  saved panel figure4_panelB_{slug}_pc1.svg")

        # Col 2: UMAP-1 spatial map
        ax2 = fig.add_subplot(gs[row_i, 2])
        ax2.imshow(umap1, cmap=CMAP_UMAP1, vmin=0, vmax=1,
                   aspect="equal", interpolation="antialiased")
        ax2.axis("off")
        if row_i == 0:
            ax2.text(0.5, 1.01, COL_LABELS[2], transform=ax2.transAxes,
                     fontsize=9, ha="center", va="bottom")
        # save individual UMAP-1 map
        _f, _a = plt.subplots(figsize=(3, 3 * umap1.shape[0]/umap1.shape[1]))
        _a.imshow(umap1, cmap=CMAP_UMAP1, vmin=0, vmax=1, aspect="equal", interpolation="antialiased")
        _a.axis("off")
        _f.savefig(str(PANEL_DIR / f"figure4_panelB_{slug}_umap1.svg"), bbox_inches="tight", pad_inches=0)
        plt.close(_f)
        print(f"  saved panel figure4_panelB_{slug}_umap1.svg")

        # Col 3: ROI overlay on grayscale ion image
        ax3 = fig.add_subplot(gs[row_i, 3])
        ax3.imshow(ion, cmap="gray", aspect="equal", interpolation="antialiased")
        overlay = np.zeros((*ion.shape, 4), dtype=np.float32)
        overlay[mask] = [0.85, 0.12, 0.12, 0.50]
        ax3.imshow(overlay, aspect="equal", interpolation="antialiased")
        ax3.axis("off")
        if row_i == 0:
            ax3.text(0.5, 1.01, COL_LABELS[3], transform=ax3.transAxes,
                     fontsize=9, ha="center", va="bottom")
        # save individual ROI overlay
        _f, _a = plt.subplots(figsize=(3, 3 * ion.shape[0]/ion.shape[1]))
        _a.imshow(ion, cmap="gray", aspect="equal", interpolation="antialiased")
        _ov = np.zeros((*ion.shape, 4), dtype=np.float32)
        _ov[mask] = [0.85, 0.12, 0.12, 0.50]
        _a.imshow(_ov, aspect="equal", interpolation="antialiased")
        _a.axis("off")
        _f.savefig(str(PANEL_DIR / f"figure4_panelB_{slug}_roi.svg"), bbox_inches="tight", pad_inches=0)
        plt.close(_f)
        print(f"  saved panel figure4_panelB_{slug}_roi.svg")

    # Dedicated colorbar axes in col 4, spanning all rows
    cbar_ax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label("PC1 (norm.)", fontsize=8)
    cb.ax.tick_params(labelsize=7)


# â"€â"€ PANEL B: contiguity violin per organ â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def draw_panel_b(ax):
    df   = pd.read_csv(SPATIAL_DIR / "spatial_coherence_all_samples.csv")
    data = [df[df["organ"] == o]["contiguity"].values for o in ORGANS_B]

    vp = ax.violinplot(data, positions=range(len(ORGANS_B)),
                       showmedians=True, showextrema=False, widths=0.7)
    for body in vp["bodies"]:
        body.set_facecolor(COLOR_PRIMARY)
        body.set_alpha(0.55)
        body.set_edgecolor(COLOR_PRIMARY)
        body.set_linewidth(0.8)
    vp["cmedians"].set_color(COLOR_PRIMARY)
    vp["cmedians"].set_linewidth(2)

    ax.axhline(RANDOM_BASELINE, color="#cc0000", lw=1.8, ls="--", zorder=5,
               label=f"Random baseline ({RANDOM_BASELINE})")

    global_mean = df["contiguity"].mean()
    ax.axhline(global_mean, color=COLOR_PRIMARY, lw=1.2, ls=":", zorder=3)
    ax.text(len(ORGANS_B) + 0.1, global_mean,
            f"Mean\n{global_mean:.3f}",
            fontsize=8, color=COLOR_PRIMARY, ha="left", va="center", clip_on=False)

    ax.set_xticks(range(len(ORGANS_B)))
    ax.set_xticklabels(ORGANS_B, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Spatial contiguity", fontsize=11)
    ax.set_title("C   Patch Spatial Contiguity\n(k=6 clusters, n=5,600 samples)",
                 fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


# â"€â"€ PANEL C: within vs between organ cosine similarity â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def draw_panel_e(fig, gs_cell):
    """Panel E: K-means metabolic microregion maps for 4 organs (1x4 grid)."""
    ORGANS_E = ["Brain", "Kidney", "Liver", "Lung"]
    CROP_X0, CROP_Y0 = 790, 152

    gs   = gs_cell.subgridspec(1, len(ORGANS_E), hspace=0.04, wspace=0.05)
    axes = []
    for i, organ in enumerate(ORGANS_E):
        ax = fig.add_subplot(gs[0, i])
        path = SPATIAL_DIR / f"metabolic_microregions_{organ}.png"
        img  = np.array(Image.open(path))
        img  = img[CROP_Y0:, CROP_X0:, :3]
        ax.imshow(img, aspect="equal", interpolation="nearest")
        ax.set_title(organ, fontsize=11, fontweight="bold", pad=3)
        ax.axis("off")
        axes.append(ax)

    axes[0].text(-0.06, 1.18, "E", transform=axes[0].transAxes,
                 fontsize=14, fontweight="bold", va="top")
    axes[0].set_title(
        "E   Metabolic Microregion Maps  (K-means, k=6, Stage 1 patch embeddings)",
        fontsize=12, fontweight="bold", pad=22, loc="left")


def draw_panel_c(ax):
    df_pairs  = pd.read_csv(MOL_DIR / "consistency_all_pairs.csv")
    df_global = pd.read_csv(MOL_DIR / "consistency_global.csv")

    organs  = [r for r in df_global["organ"].tolist() if r != "OVERALL"]
    within  = [float(df_global[df_global["organ"] == o]["within_cosine"].iloc[0])
               for o in organs]
    between = []
    for o in organs:
        mask = (df_pairs["comparison"] == "between") & \
               ((df_pairs["organ_a"] == o) | (df_pairs["organ_b"] == o))
        between.append(float(df_pairs[mask]["mean_cosine"].mean()))

    x     = np.arange(len(organs))
    bar_w = 0.35

    ax.bar(x - bar_w / 2, within,  width=bar_w, color=COLOR_WITHIN,
           label="Within-organ",  edgecolor="white", linewidth=0.4)
    ax.bar(x + bar_w / 2, between, width=bar_w, color=COLOR_BETWEEN,
           label="Between-organ", edgecolor="white", linewidth=0.4)

    ymax_all = max(max(within), max(between))
    ymin_all = min(min(within), min(between))
    for i, (w, b) in enumerate(zip(within, between)):
        ax.text(x[i], max(w, b) + 0.010, f"d={w - b:+.2f}",
                ha="center", va="bottom", fontsize=8, color="#222", fontweight="bold")

    ax.axhline(0, color="#aaa", lw=0.8)
    ax.set_ylim(min(ymin_all - 0.04, -0.08), ymax_all + 0.10)
    ax.set_xticks(x)
    ax.set_xticklabels(organs, fontsize=10)
    ax.set_ylabel("Mean cosine similarity\n(spatial patch embeddings)", fontsize=11)
    ax.set_title("D   Within- vs Between-organ\nSpatial Map Similarity",
                 fontsize=12, fontweight="bold", pad=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    ov = df_global[df_global["organ"] == "OVERALL"].iloc[0]
    ax.text(0.98, -0.22,
            f"Overall: within={ov['within_cosine']:.3f}, "
            f"between={ov['between_cosine']:.3f},  d = +{ov['delta']:.3f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", color="#444")



# â"€â"€ MAIN â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def main():
    # â"€â"€ individual Panel A â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    fig_a = plt.figure(figsize=(18, 9))
    gs_a  = fig_a.add_gridspec(1, 1)[0, 0]
    draw_panel_a(fig_a, gs_a)
    fig_a.suptitle(
        "A   Spatially Resolved Patch Representations  -  ROI-Annotated Samples\n"
        "Best ion channel (highest annotated-vs-background silhouette score)",
        fontsize=11, y=1.01)
    save_panel(fig_a, "figure4_panelB_roi_combined")
    plt.close(fig_a)

    # ── individual Panel C ─â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    fig_b, ax_b = plt.subplots(figsize=(6, 4.5))
    draw_panel_b(ax_b)
    save_panel(fig_b, "figure4_panelC_contiguity")
    plt.close(fig_b)

    # ── individual Panel D ─â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    fig_c, ax_c = plt.subplots(figsize=(5.5, 4.5))
    draw_panel_c(ax_c)
    save_panel(fig_c, "figure4_panelD_consistency")
    plt.close(fig_c)

    print("[DONE] Figure 4 panels ->", PANEL_DIR)


if __name__ == "__main__":
    main()


