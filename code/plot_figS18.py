"""
plot_figS18.py
--------------
Supplementary Figure S18: Drug Landscape — Extended Analysis.

Extends Figure 6 (panel d) beyond the 6 highlighted drugs.

Panels:
  A  Distribution of drug-similarity scores across all molecules
  B  Top-20 highest drug-similar non-drug molecules (ranked bar)
  C  PCA scatter coloured by drug similarity score (continuous, all molecules)

Usage:
  conda run -n torch_gpu python plot_figS18.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from plot_utils import set_nature_style
set_nature_style()

# -- CONFIG -------------------------------------------------------------------
FIG7_DIR  = METABOFM_ROOT / "outputs/figure7_data"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS18_drug_extended"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI        = 300
DRUG_COLOR = "#d6604d"
NDRUG_COLOR = "#2166ac"
CMAP       = "viridis"


CAPTION = """\
Supplementary Figure 18 | Extended drug proximity analysis.

Extended drug proximity analysis across the full set of molecules in the MetaboFM channel embedding space.

a, Distribution of drug-similarity scores for all annotated molecules, split by whether the molecule has a confirmed drug match (red) or not (blue). Drug-confirmed molecules cluster at higher similarity scores, validating that the embedding space organises molecules by pharmacological relevance.

b, Top-20 non-drug molecules ranked by drug-proximity score, labelled with molecule name and rounded m/z (x-axis zoomed to the 0.97-1.00 range in which these top-ranked scores fall, to make rank differences visible). These represent candidate endogenous metabolites with structural or metabolic proximity to known drugs, and may be of interest for pharmacological or biomarker studies.

c, PCA scatter of all molecules in the Stage 2 channel embedding space, coloured continuously by drug-similarity score (viridis scale). Known drug molecules are overlaid as filled circles (red). High drug-similarity regions of the PCA space correspond to known pharmacological compound classes, confirming that the embedding captures chemically meaningful structure.
"""

def write_caption():
    (PANEL_DIR / "captions.txt").write_text(CAPTION, encoding="utf-8")
    print("  saved captions.txt")


def save_panel(fig, stem):
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    path = PANEL_DIR / stem
    fig.savefig(str(path.with_suffix(".svg")), bbox_inches="tight", pad_inches=0)
    print(f"  saved panel {stem}.svg")


def draw_panel_a(ax, df):
    """Histogram of drug_similarity scores, drugs vs non-drugs."""
    drugs  = df[df["has_drug"] == True]["drug_similarity"].dropna()
    ndrugs = df[df["has_drug"] == False]["drug_similarity"].dropna()

    bins = np.linspace(0, 1, 40)
    ax.hist(ndrugs, bins=bins, color=NDRUG_COLOR, alpha=0.6,
            label=f"Non-drug molecules (n={len(ndrugs):,})", density=True)
    ax.hist(drugs, bins=bins, color=DRUG_COLOR, alpha=0.7,
            label=f"Drug molecules (n={len(drugs):,})", density=True)

    ax.set_xlabel("Drug similarity score", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("A   Drug Similarity Score Distribution",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)


def draw_panel_b(ax, df):
    """Top-20 non-drug molecules by drug similarity (ranked bar)."""
    top = (df[df["has_drug"] == False]
           .nlargest(20, "drug_similarity")
           .reset_index(drop=True))
    y = np.arange(len(top))
    vals = top["drug_similarity"].values
    # These top-20 values are all tightly clustered near the top of the
    # [0,1] score range, so bars starting at x=0 make the (real) rank
    # differences invisible; zoom the x-axis to the actual value range
    # and add numeric labels instead.
    pad = max((vals.max() - vals.min()) * 0.15, 0.002)
    xlo, xhi = vals.min() - pad, vals.max() + pad
    ax.barh(y, vals - xlo, left=xlo, color=NDRUG_COLOR,
            alpha=0.8, height=0.7, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(v + pad * 0.08, i, f"{v:.3f}", va="center",
                fontsize=7, color="#333")
    ax.set_xlim(xlo, xhi + pad * 0.9)
    labels = [f"{row.mol_name}  ({row.mz_r:.4f})" for _, row in top.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Drug similarity score", fontsize=9)
    ax.set_title("B   Top-20 Non-drug Molecules by Drug Proximity",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def draw_panel_c(ax, df, fig):
    """PCA scatter coloured continuously by drug similarity score."""
    sc = ax.scatter(df["PC1"], df["PC2"],
                    c=df["drug_similarity"], cmap=CMAP,
                    s=5, alpha=0.6, linewidths=0, rasterized=True)
    # Overlay drug-confirmed molecules
    drugs = df[df["has_drug"] == True]
    ax.scatter(drugs["PC1"], drugs["PC2"],
               c=DRUG_COLOR, s=18, alpha=0.9,
               linewidths=0.5, edgecolors="white",
               label="Known drugs", rasterized=True, zorder=3)

    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Drug similarity score", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.set_title("C   Molecule Embedding Space (PCA)\nColoured by Drug Similarity",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.tick_params(labelsize=8)


def main():
    df = pd.read_csv(FIG7_DIR / "figure7_mol_df.csv")
    df["has_drug"] = df["has_drug"].astype(str).str.strip().str.lower() == "true"
    print(f"Loaded {len(df):,} molecules, {df['has_drug'].sum()} with drug match")

    # Panel A
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    draw_panel_a(ax, df)
    save_panel(fig, "figS18_panelA_drug_similarity_dist")
    plt.close(fig)

    # Panel B
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    draw_panel_b(ax, df)
    save_panel(fig, "figS18_panelB_top20_nondrug")
    plt.close(fig)

    # Panel C
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    draw_panel_c(ax, df, fig)
    save_panel(fig, "figS18_panelC_pca_drug_similarity")
    plt.close(fig)

    write_caption()
    print("FigS18 done.")


if __name__ == "__main__":
    main()



