"""
save_legends.py
---------------
Saves standalone legend and colorbar SVGs (no padding, no titles) for each
figure subfolder, matching the colors / labels defined in the figure scripts.

Run from code_v2:
  python save_legends.py
"""

from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib as mpl
import numpy as np

FIG_DIR = METABOFM_ROOT / "outputs/figures"


def _legend_fig(handles, labels, ncol=1, figsize=None, loc="center",
                handlelength=1.4, handleheight=1.0):
    """Return a tight figure containing only the legend."""
    if figsize is None:
        figsize = (max(2.5, ncol * 2.5), max(1.0, len(labels) / ncol * 0.38 + 0.3))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.legend(handles, labels, frameon=False, fontsize=9,
              ncol=ncol, loc=loc,
              handlelength=handlelength, handleheight=handleheight,
              borderpad=0, labelspacing=0.4, handletextpad=0.5,
              columnspacing=1.0)
    return fig


def _colorbar_fig(cmap, vmin, vmax, label, orientation="vertical",
                  figsize=None, n_ticks=5):
    """Return a tight figure containing only a colorbar."""
    if figsize is None:
        figsize = (1.0, 3.5) if orientation == "vertical" else (3.5, 0.8)
    fig, ax = plt.subplots(figsize=figsize)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                   orientation=orientation)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.set_ticks(np.linspace(vmin, vmax, n_ticks))
    return fig


def save(fig, path):
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved {path.name}")


# ── FIGURE 2 ────────────────────────────────────────────────────────────────

def figure2():
    d = FIG_DIR / "figure2"
    d.mkdir(exist_ok=True)

    VARIANT_LABELS = {
        "stage2_ch_refined__unambig[all]": "Stage 2 (channel-refined)",
        "resnet+smiles[all]":              "ResNet + SMILES",
        "resnet_only__unambig[all]":       "Stage 1 (channel)",
        "mz_only[unambiguous]":            "m/z only",
        "metadata_only[unambiguous]":      "Metadata only",
        "smiles_only[all]":                "SMILES only (structure baseline)",
        "imagenet__unambig[all]":          "ImageNet ResNet",
    }
    COLORS = {
        "stage2_ch_refined__unambig[all]": "#2166ac",
        "resnet+smiles[all]":              "#4dac26",
        "resnet_only__unambig[all]":       "#74add1",
        "mz_only[unambiguous]":            "#d6604d",
        "metadata_only[unambiguous]":      "#b2b2b2",
        "smiles_only[all]":                "#f4a582",
        "imagenet__unambig[all]":          "#c2c2c2",
    }
    HATCH = {"mz_only[unambiguous]": "//", "smiles_only[all]": "//"}

    handles, labels = [], []
    for k, lbl in VARIANT_LABELS.items():
        h = mpatches.Patch(facecolor=COLORS[k],
                           hatch=HATCH.get(k, None),
                           edgecolor=COLORS[k] if k in HATCH else "white",
                           linewidth=0.4)
        handles.append(h); labels.append(lbl)
    save(_legend_fig(handles, labels, figsize=(3.5, 2.5)), d / "figure2_legend_variants.svg")

    # figure-level highlight legend (3 entries)
    hl_handles = [
        mpatches.Patch(facecolor="#2166ac", label="MetaboFM Stage 2 (ours)"),
        mpatches.Patch(facecolor="#d6604d", hatch="//", edgecolor="#d6604d",
                       label="m/z only (trivial baseline)"),
        mpatches.Patch(facecolor="#f4a582", hatch="//", edgecolor="#f4a582",
                       label="SMILES only (structure baseline)"),
    ]
    save(_legend_fig(hl_handles, [h.get_label() for h in hl_handles],
                     ncol=1, figsize=(3.5, 1.2)),
         d / "figure2_legend_highlights.svg")

    # viridis colorbar for ion images (panel E)
    save(_colorbar_fig("viridis", 0, 1, "Ion intensity (norm.)",
                       figsize=(0.55, 2.8)),
         d / "figure2_legend_viridis_colorbar.svg")


# ── FIGURE 3, 5 ───────────────────────────────────────────────────────────
# NOTE: Figure 3 (molecular variation, within/between-molecule legend) and
# Figure 5 (organ/organism UMAP + retrieval legends) now draw their legends
# INLINE within the saved panel axes (plot_figure3.py draw_panel_a; plot_
# figure5.py draw_panel_b/c/d), so no standalone legend SVGs are needed for
# those two figures. Do not regenerate separate legend files for them.


# ── FIGURE 4 ────────────────────────────────────────────────────────────────
# Spatial patch figure (main Fig. 4). Individual per-sample PC1 / UMAP-1
# spatial-map tiles are exported WITHOUT a colorbar (only the combined
# "figure4_panelB_roi_combined.svg" preview has one baked in). Both the PC1
# and UMAP-1 maps share the same RdBu_r, 0-1 range, so one shared colorbar
# covers both column types.

def figure4():
    d = FIG_DIR / "figure4"
    d.mkdir(exist_ok=True)

    save(_colorbar_fig("RdBu_r", 0, 1, "PC1 / UMAP-1 (norm.)", figsize=(0.55, 2.8)),
         d / "figure4_legend_pc1_umap1_colorbar.svg")


# ── FIGURE 6 ────────────────────────────────────────────────────────────────

def figure6():
    d = FIG_DIR / "figure6"
    d.mkdir(exist_ok=True)

    CLASS_LABELS = [
        "Lipids and lipid-like molecules",
        "Organoheterocyclic compounds",
        "Organic acids and derivatives",
        "Organic oxygen compounds",
        "Benzenoids",
        "Phenylpropanoids and polyketides",
        "Nucleosides, nucleotides, and analogues",
        "Organic nitrogen compounds",
        "Other / Unknown",
    ]
    CLASS_SHORT = {
        "Lipids and lipid-like molecules":            "Lipids",
        "Organoheterocyclic compounds":               "Organoheterocyclics",
        "Organic acids and derivatives":              "Organic acids",
        "Organic oxygen compounds":                   "Org. oxygen cpds.",
        "Benzenoids":                                 "Benzenoids",
        "Phenylpropanoids and polyketides":           "Phenylpropanoids",
        "Nucleosides, nucleotides, and analogues":    "Nucleosides/nts.",
        "Organic nitrogen compounds":                 "Org. nitrogen cpds.",
        "Other / Unknown":                            "Other / Unknown",
    }
    CLASS_PALETTE = [
        "#2166ac", "#d6604d", "#4dac26", "#fdae6b", "#9970ab",
        "#1b7837", "#e08214", "#74add1", "#c8c8c8",
    ]

    handles = [mpatches.Patch(color=c, label=CLASS_SHORT[l])
               for c, l in zip(CLASS_PALETTE, CLASS_LABELS)]
    save(_legend_fig(handles, [h.get_label() for h in handles],
                     ncol=1, figsize=(3.0, 2.8)),
         d / "figure6_legend_hmdb_classes.svg")


# NOTE: old figure7() (drug-likeness colorbar + drug-matched scatter legend)
# was removed -- that content is now part of main Figure 6 (panel d), whose
# own draw_panel_d() in plot_figure6.py already draws and saves both the
# continuous colorbar and the categorical legend inline, so no separate
# standalone files are needed.


# ── FIGURE 7 (cross-study retrieval; formerly Figure 8) ─────────────────────

def figure7():
    d = FIG_DIR / "figure7"
    d.mkdir(exist_ok=True)

    # border legend for retrieval image panels
    handles = [
        mpatches.Patch(color="#2ca02c", label="Correct tissue match"),
        mpatches.Patch(color="#d62728", label="Tissue mismatch"),
    ]
    save(_legend_fig(handles, [h.get_label() for h in handles], figsize=(2.5, 0.8)),
         d / "figure7_legend_retrieval_border.svg")

    # query color legend (Kidney / Brain / Lung)
    QUERY_COLORS = ["#2166ac", "#d6604d", "#4dac26"]
    QUERY_LABELS = ["Kidney (query)", "Brain (query)", "Lung (query)"]
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(QUERY_COLORS, QUERY_LABELS)]
    save(_legend_fig(handles, [h.get_label() for h in handles], figsize=(2.5, 1.0)),
         d / "figure7_legend_query_colors.svg")

    # purity curve legend (Stage 2 / Stage 1 / Random)
    handles = [
        mlines.Line2D([0], [0], color="#2166ac", lw=2, label="Stage 2 (MetaboFM)"),
        mlines.Line2D([0], [0], color="#d6604d", lw=2, label="Stage 1 (ResNet)"),
        mlines.Line2D([0], [0], color="#aaaaaa", lw=1.5, ls="--", label="Random baseline"),
    ]
    save(_legend_fig(handles, [h.get_label() for h in handles], figsize=(2.8, 1.0)),
         d / "figure7_legend_purity.svg")

    # viridis colorbar for retrieval ion images
    save(_colorbar_fig("viridis", 0, 1, "Ion intensity (norm.)",
                       figsize=(0.55, 2.8)),
         d / "figure7_legend_viridis_colorbar.svg")


# ── SUPPLEMENTARY FIGURE S6 (extended spatial analysis) ─────────────────────
# Panel a's k=6 unsupervised microregion cluster map has no legend anywhere.
# Cluster identity is arbitrary per-sample (cluster 1 in Brain is not the
# same biological region as cluster 1 in Kidney), so labels are generic
# "Region 1-6" rather than named tissue compartments.

def figS6():
    d = FIG_DIR / "figS6_spatial_extended"
    d.mkdir(exist_ok=True)

    REGION_COLORS = ["#2166ac", "#d6604d", "#4dac26", "#fdae6b", "#9970ab", "#74add1"]
    handles = [mpatches.Patch(color=c, label=f"Region {i+1}")
               for i, c in enumerate(REGION_COLORS)]
    save(_legend_fig(handles, [h.get_label() for h in handles],
                     ncol=2, figsize=(3.0, 1.4)),
         d / "figS6_legend_microregions.svg")


# ── SUPPLEMENTARY FIGURE S7 (annotated patch separation) ────────────────────
# Panel c (top-3 channel ion / PC1 / UMAP-1 spatial maps) has no colorbar at
# all, even in the combined preview. PC1 and UMAP-1 rows share an RdBu_r,
# 0-1 range identical to main Fig. 4's colorbar.

def figS7():
    d = FIG_DIR / "figS7_annotated_patch_separation"
    d.mkdir(exist_ok=True)

    save(_colorbar_fig("RdBu_r", 0, 1, "PC1 / UMAP-1 (norm.)", figsize=(0.55, 2.8)),
         d / "figS7_legend_pc1_umap1_colorbar.svg")


# ── SUPPLEMENTARY FIGURE S9 (similarity-map null-model comparison) ──────────
# Panels b/c share a -1 to 1 RdBu_r cosine-similarity scale.

def figS9():
    d = FIG_DIR / "figS9_similarity_map_null_model"
    d.mkdir(exist_ok=True)

    save(_colorbar_fig("RdBu_r", -1, 1, "Cosine similarity to organ centroid",
                       figsize=(0.55, 2.8)),
         d / "figS9_legend_similarity_colorbar.svg")


# ── SHARED ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from plot_utils import set_nature_style
    set_nature_style()

    print("[Figure 2]"); figure2()
    print("[Figure 4]"); figure4()
    print("[Figure 6]"); figure6()
    print("[Figure 7]"); figure7()
    print("[Supp Figure S6]"); figS6()
    print("[Supp Figure S7]"); figS7()
    print("[Supp Figure S9]"); figS9()
    print("[DONE]")
