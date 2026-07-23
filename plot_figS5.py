"""
plot_figS5.py
-------------
Supplementary Figure S5: Full Molecule Variance Analysis.

Extends Figure 3 from 4 selected molecules to the full m/z set.

Panels:
  A  Within-molecule similarity vs observation count (scatter, Stage 2 highlighted)
  B  Top-20 most consistent and bottom-20 least consistent m/z features (Stage 2)

Note: the former panel A (violin of within-molecule similarity stratified by
variant, all m/z features) was removed -- it duplicated Fig. 3c, which is the
same violin plot restricted to m/z groups with >=10 samples each.

Usage:
  conda run -n torch_gpu python plot_figS5.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_utils import set_nature_style
set_nature_style()

# -- CONFIG -------------------------------------------------------------------
MOL_DIR   = METABOFM_ROOT / "outputs/molecule_variance"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS5_molecule_variance_full"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

VARIANT_COLORS = {
    "MetaboFM Stage 2": "#2166ac",
    "Stage 1 (ResNet)": "#d6604d",
    "ResNet + SMILES":  "#4dac26",
    "SMILES only":      "#aaaaaa",
}
VARIANT_ORDER = ["MetaboFM Stage 2", "Stage 1 (ResNet)", "ResNet + SMILES", "SMILES only"]


CAPTION = """\
Supplementary Figure 5 | Full molecule-level embedding consistency analysis across all m/z features.

Full molecule-level embedding consistency across the complete set of observed m/z features, extending the sample-size-thresholded analysis in Fig. 3c.

a, Within-molecule similarity as a function of observation count (number of datasets in which the m/z was detected, capped at 50 per molecule for computational tractability; points jittered horizontally to show density at each integer count). Molecules observed in many contexts tend to show stable within-molecule similarity under Stage 2, suggesting the representation is robust to dataset-specific variation.

b, Top-20 most consistent (highest within-molecule similarity) and bottom-20 least consistent m/z features under Stage 2, labelled with rounded m/z and observation count. Highly consistent molecules tend to be abundant, well-annotated metabolites; least consistent features typically correspond to rare or context-specific ions.
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


def draw_panel_a(ax, df, rng=None):
    """Scatter: observation count vs within-molecule similarity for Stage 2."""
    if rng is None:
        rng = np.random.default_rng(0)
    s2 = df[df["variant"] == "MetaboFM Stage 2"].copy()
    s1 = df[df["variant"] == "Stage 1 (ResNet)"].copy()
    col_s2 = VARIANT_COLORS["MetaboFM Stage 2"]
    col_s1 = VARIANT_COLORS["Stage 1 (ResNet)"]

    # n_obs only takes ~40 discrete integer values, so points at each x
    # stack into dense vertical columns; add a small horizontal jitter
    # (dithering the integer count, not the underlying data) to reveal
    # point density instead of a solid vertical smear.
    jitter_s1 = rng.uniform(-0.35, 0.35, size=len(s1))
    jitter_s2 = rng.uniform(-0.35, 0.35, size=len(s2))

    ax.scatter(s1["n_obs"] + jitter_s1, s1["within_sim"], s=5, alpha=0.2,
               color=col_s1, linewidths=0, label="Stage 1", rasterized=True)
    ax.scatter(s2["n_obs"] + jitter_s2, s2["within_sim"], s=5, alpha=0.3,
               color=col_s2, linewidths=0, label="Stage 2", rasterized=True)

    ax.set_xlabel("Number of observations (datasets)", fontsize=9)
    ax.set_ylabel("Within-molecule cosine similarity", fontsize=9)
    ax.set_title("A   Molecule Consistency vs Observation Count",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)


def draw_panel_b(ax_top, ax_bot, df):
    """Top-20 and bottom-20 m/z features by Stage 2 within-molecule similarity."""
    s2 = (df[df["variant"] == "MetaboFM Stage 2"]
          .sort_values("within_sim", ascending=False)
          .reset_index(drop=True))
    top20 = s2.head(20)
    bot20 = s2.tail(20).sort_values("within_sim", ascending=True)

    col = VARIANT_COLORS["MetaboFM Stage 2"]
    for sub_ax, sub_df, title in [
        (ax_top, top20, "B (i)   Top-20 Most Consistent m/z Features"),
        (ax_bot, bot20, "B (ii)  Bottom-20 Least Consistent m/z Features"),
    ]:
        labels = [f"{row.mz_r:.4f} (n={int(row.n_obs)})" for _, row in sub_df.iterrows()]
        vals = sub_df["within_sim"].values
        # Bars all start at x=0 by default, which compresses the visible
        # difference between values that are all clustered near the top
        # (or bottom) of the [0,1] range; zoom each subplot's x-axis to
        # its own value range so the within-group variation is visible.
        pad = max((vals.max() - vals.min()) * 0.15, 0.005)
        xlo, xhi = vals.min() - pad, vals.max() + pad
        sub_ax.barh(range(len(sub_df)), vals - xlo, left=xlo,
                    color=col, alpha=0.75, height=0.7, edgecolor="none")
        for i, v in enumerate(vals):
            sub_ax.text(v + pad * 0.08, i, f"{v:.3f}", va="center",
                        fontsize=6.5, color="#333")
        sub_ax.set_xlim(xlo, xhi + pad * 0.9)
        sub_ax.set_yticks(range(len(sub_df)))
        sub_ax.set_yticklabels(labels, fontsize=7)
        sub_ax.invert_yaxis()
        sub_ax.set_xlabel("Within-molecule cosine similarity", fontsize=8)
        sub_ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        sub_ax.spines["top"].set_visible(False)
        sub_ax.spines["right"].set_visible(False)
        sub_ax.tick_params(axis="x", labelsize=7.5)


def main():
    df = pd.read_csv(MOL_DIR / "molecule_variance_per_mz.csv")
    print(f"Loaded {len(df):,} rows, {df['mz_r'].nunique():,} unique m/z, "
          f"{df['variant'].nunique()} variants")

    # Panel A
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    draw_panel_a(ax, df)
    save_panel(fig, "figS5_panelA_similarity_vs_obs")
    plt.close(fig)

    # Panel B
    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(13.0, 6.5))
    draw_panel_b(ax_top, ax_bot, df)
    save_panel(fig, "figS5_panelB_top_bottom_mz")
    plt.close(fig)

    write_caption()
    print("FigS5 done.")


if __name__ == "__main__":
    main()

