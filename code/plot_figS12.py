"""
plot_figS12.py
-------------
Supplementary Figure S12: HMDB Class Retrieval — Full Breakdown.

Extends Figure 6 panel C from selected classes to all HMDB super-classes.

Panels:
  A  mAP@10 for all HMDB super-classes, sorted descending

Usage:
  conda run -n torch_gpu python plot_figS12.py
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
CENT_DIR  = METABOFM_ROOT / "outputs/molecule_centroids"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS12_hmdb_extended"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI       = 300
S2_COLOR  = "#2166ac"

PALETTE = [
    "#2166ac", "#d6604d", "#4dac26", "#fdae6b", "#9970ab",
    "#1b7837", "#e08214", "#74add1", "#a50026", "#35978f",
    "#8c510a", "#01665e", "#c51b7d", "#4d9221", "#762a83",
]


CAPTION = """\
Supplementary Figure 12 | HMDB super-class retrieval performance across all annotated classes.

HMDB super-class retrieval performance across the complete set of annotated classes with sufficient representation.

a, Mean Average Precision at 10 (MAP@10) for each HMDB super class, comparing Stage 2 (blue) and Stage 1 (red), sorted by Stage 2 performance. Stage 2 outperforms Stage 1 for nearly every class, including the most represented class (lipids, n = 841: 0.840 versus 0.797) and structurally diverse classes such as organoheterocyclics (0.507 versus 0.459). Benzenoids is the sole near-tie (0.306 versus 0.309), and singleton or near-singleton classes (n <= 7) score near zero for both stages, reflecting insufficient within-class examples for retrieval. n labels indicate the number of molecule groups per class.

b, Same comparison as panel (a), shown as a Stage 2 versus Stage 1 scatter plot with one point per HMDB class (point size proportional to n) and a diagonal reference line. Nearly all classes fall above the diagonal, confirming that Stage 2 improves retrieval consistently across chemical classes rather than for a small subset.
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


def draw_panel_a(ax, df_s2, df_s1):
    """Grouped horizontal bar chart: Stage 2 vs Stage 1 mAP@10 per HMDB class."""
    merged = df_s2.merge(df_s1[["hmdb_super_class", "map_at_10"]].rename(
                             columns={"map_at_10": "map_s1"}),
                         on="hmdb_super_class", how="left")
    merged = merged.sort_values("map_at_10", ascending=True)
    labels = merged["hmdb_super_class"].tolist()
    y      = np.arange(len(labels))
    h      = 0.35

    ax.barh(y + h / 2, merged["map_at_10"].values, h * 0.9,
            color=S2_COLOR, alpha=0.85, label="Stage 2")
    ax.barh(y - h / 2, merged["map_s1"].values,    h * 0.9,
            color="#d6604d", alpha=0.85, label="Stage 1")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mAP@10", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.axvline(0.5, color="#888", lw=0.8, ls="--")
    ax.legend(fontsize=8, frameon=False)

    n_grp = merged["n_groups"].tolist() if "n_groups" in merged.columns else [None] * len(merged)
    for i, (v, n) in enumerate(zip(merged["map_at_10"].values, n_grp)):
        txt = f"n={n}" if n is not None else ""
        ax.text(v + 0.01, y[i] + h / 2, txt, va="center", fontsize=6.5, color="#555")

    ax.set_title("A   HMDB Super-class Retrieval mAP@10 (Stage 2 vs Stage 1)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def draw_panel_b(ax, df_s2, df_s1):
    """Stage 2 vs Stage 1 mAP@10 scatter, one point per HMDB class, sized by
    n_groups, with a diagonal (y=x) reference line. Points above the line
    indicate classes where Stage 2 outperforms Stage 1."""
    merged = df_s2.merge(df_s1[["hmdb_super_class", "map_at_10"]].rename(
                             columns={"map_at_10": "map_s1"}),
                         on="hmdb_super_class", how="left")

    sizes = 20 + 4 * np.sqrt(merged["n_groups"].values.astype(float))
    ax.scatter(merged["map_s1"].values, merged["map_at_10"].values,
               s=sizes, color=S2_COLOR, alpha=0.75, edgecolor="white", linewidth=0.5)

    lim = [0, 1.0]
    ax.plot(lim, lim, color="#888", lw=0.8, ls="--", zorder=0)

    texts = [
        ax.text(row["map_s1"], row["map_at_10"], row["hmdb_super_class"],
                 fontsize=6, color="#444")
        for _, row in merged.iterrows()
    ]
    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            x=merged["map_s1"].values, y=merged["map_at_10"].values,
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.5),
            expand_points=(1.4, 1.4), expand_text=(1.2, 1.2),
            force_text=(0.4, 0.6), force_points=(0.3, 0.3),
        )
    except ImportError:
        pass

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Stage 1 mAP@10", fontsize=9)
    ax.set_ylabel("Stage 2 mAP@10", fontsize=9)
    ax.set_title("B   Stage 2 vs Stage 1, per HMDB Class\n(points above diagonal: Stage 2 wins; size = n_groups)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")


def main():
    df_s2 = pd.read_csv(CENT_DIR / "perclass_map10.csv")
    df_s2.columns = [c.strip() for c in df_s2.columns]
    df_s1 = pd.read_csv(CENT_DIR / "perclass_map10_stage1.csv")
    print(f"Loaded {len(df_s2)} Stage 2 / {len(df_s1)} Stage 1 HMDB classes")

    # Panel A
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, len(df_s2) * 0.55)))
    draw_panel_a(ax, df_s2, df_s1)
    save_panel(fig, "figS12_panelA_perclass_map10")
    plt.close(fig)

    # Panel B
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    draw_panel_b(ax, df_s2, df_s1)
    save_panel(fig, "figS12_panelB_scatter")
    plt.close(fig)

    write_caption()
    print("FigS7 done.")


if __name__ == "__main__":
    main()


