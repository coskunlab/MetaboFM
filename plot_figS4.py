"""
plot_figS4.py
-------------
Supplementary Figure S4: Ablation study — training data scale and embedding variant comparisons.

Panels:
  A  Data-size ablation: cross-acquisition Recall@1 vs fraction of Stage 2 training data
  B  Molecule retrieval Tanimoto similarity for 4 embedding variants

Note: panel C (cross-platform tissue discriminability delta by variant) was
moved into Fig. 2d(ii) in the main text, since it's the direct explanation for
the cross-platform consistency numbers discussed there; kept here only as
duplicated content otherwise.

Usage:
  conda run -n torch_gpu python plot_figS4.py
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
ABL_DIR   = METABOFM_ROOT / "outputs/ablation_datasize"
XPLAT_DIR = METABOFM_ROOT / "outputs/crossplatform_consistency"
SMILES_DIR = METABOFM_ROOT / "outputs/smiles_retrieval"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS4_ablation_stage1_vs_stage2"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

COLORS = {
    "stage2":   "#2166ac",
    "stage1":   "#4dac26",
    "smiles":   "#d6604d",
    "random":   "#aaaaaa",
    "mz":       "#762a83",
}

CAPTION = """\
Supplementary Figure 4 | Ablation study: training data scale and embedding variant comparisons.

a, Data-size ablation for cross-acquisition transfer. Macro Recall@1 under leave-one-acquisition-out evaluation as a function of Stage 2 training set size (shaded band = +/-1 s.d. over seeds). Stage 2 with 50% of the data already matches or exceeds Stage 1 trained on 100%, demonstrating data efficiency. Dashed line is the random baseline; dot-dash line shows Stage 1 full-data performance.

b, Molecule retrieval performance (Tanimoto similarity at k = 1, 5, 10) for four embedding variants: MetaboFM Stage 2, Stage 1 (ResNet-18 mean-pool), SMILES-only (chemical structure ceiling, not part of MetaboFM), and random. Stage 2 outperforms Stage 1 and random, confirming that cross-channel aggregation improves molecule-level embedding quality without access to structure information.
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


def draw_panel_a(ax):
    """Data-size ablation: macro Recall@1 vs training fraction."""
    df = pd.read_csv(ABL_DIR / "ablation_multiseed_summary.csv")

    # Full data (fraction==1.0 or max fraction) Stage 2 line
    full = df[df["fraction"] == 1.0]
    ablation = df[df["fraction"] > 0]

    ax.fill_between(
        ablation["fraction"],
        ablation["macro_r1"] - ablation["macro_std_r1"].fillna(0),
        ablation["macro_r1"] + ablation["macro_std_r1"].fillna(0),
        alpha=0.2, color=COLORS["stage2"],
    )
    ax.plot(ablation["fraction"], ablation["macro_r1"],
            color=COLORS["stage2"], lw=2, marker="o", ms=5, label="Stage 2")

    # Stage 1 full-data reference line (fraction==0 row is random baseline)
    random_r1 = float(df[df["fraction"] == 0.0]["macro_r1"].iloc[0])
    stage1_r1 = float(
        pd.read_csv(XPLAT_DIR / "summary.csv")
        .query("variant=='stage1_meanpool'")["mean"].mean()
    ) if (XPLAT_DIR / "summary.csv").exists() else None

    ax.axhline(random_r1, color=COLORS["random"], lw=1.5, ls="--", label="Random baseline")
    if stage1_r1 is not None:
        ax.axhline(stage1_r1, color=COLORS["stage1"], lw=1.5, ls="-.", label="Stage 1 (100% data)")

    ax.set_xlabel("Fraction of Stage 2 training data", fontsize=9)
    ax.set_ylabel("Macro Recall@1", fontsize=9)
    ax.set_title("A   Data-size Ablation (cross-acquisition transfer)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)


def draw_panel_b(ax):
    """Tanimoto molecule retrieval for 4 variants."""
    df = pd.read_csv(SMILES_DIR / "smiles_retrieval_results.csv")
    # Columns: Tan@1, Tan@5, Tan@10, variant
    ks = ["Tan@1", "Tan@5", "Tan@10"]
    x = np.arange(len(ks))
    width = 0.2

    # Raw variant strings in the data file use informal analysis-script naming
    # ("Stage 2 (ours)", "SMILES (upper bound)") that doesn't match the paper's
    # naming conventions elsewhere (plain "Stage 2", "SMILES-only") -- translate
    # to display labels rather than showing the raw strings in the legend.
    display_labels = {
        "Stage 2 (ours)": "Stage 2",
        "Stage 1": "Stage 1",
        "SMILES (upper bound)": "SMILES-only",
        "Random": "Random",
    }
    color_map = {
        "Stage 2 (ours)":   COLORS["stage2"],
        "Stage 1":          COLORS["stage1"],
        "SMILES (upper bound)": COLORS["smiles"],
        "Random":           COLORS["random"],
    }

    for i, (_, row) in enumerate(df.iterrows()):
        raw_label = row["variant"]
        label = display_labels.get(raw_label, raw_label)
        vals = [row[k] for k in ks]
        color = color_map.get(raw_label, "#888888")
        hatch = "//" if "SMILES" in raw_label else ""
        ax.bar(x + (i - 1.5) * width, vals, width, label=label,
               color=color, alpha=0.85, hatch=hatch, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(ks, fontsize=8)
    ax.set_ylabel("Tanimoto similarity", fontsize=9)
    ax.set_title("B   Molecule Retrieval (Tanimoto similarity)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(labelsize=8)


def main():
    # Panel A
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    draw_panel_a(ax)
    save_panel(fig, "figS4_panelA_datasize_ablation")
    plt.close(fig)

    # Panel B
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    draw_panel_b(ax)
    save_panel(fig, "figS4_panelB_tanimoto_variants")
    plt.close(fig)

    write_caption()
    print("FigS4 done.")


if __name__ == "__main__":
    main()
