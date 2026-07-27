"""
plot_figS1.py
-------------
Supplementary Figure S1: Dataset Composition and Diversity.

Panels:
  A  Sample count by ionisation source
  B  Sample count by analyzer type
  C  Sample count by organism
  D  Top-15 organs by sample count
  E  Polarity breakdown

Usage:
  conda run -n torch_gpu python plot_figS1.py
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
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS1_dataset_composition"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI       = 300
BAR_COLOR = "#2166ac"
ALT_COLOR = "#d6604d"

PALETTE = [
    "#2166ac", "#d6604d", "#4dac26", "#fdae6b", "#9970ab",
    "#1b7837", "#e08214", "#74add1", "#a50026", "#35978f",
    "#8c510a", "#01665e", "#c51b7d", "#4d9221", "#762a83",
]


CAPTION = """\
Supplementary Figure 1 | Dataset composition and acquisition diversity of the MetaboFM training corpus.

a, Sample counts by ionisation source across the full dataset. MALDI dominates but DESI, AP-SMALDI, and several other sources are represented.

b, Sample counts by mass analyser family. Instruments span FT-ICR, Orbitrap, timsTOF, and reflector-TOF platforms, covering a broad range of mass resolving powers and acquisition workflows.

c, Sample counts by organism. Homo sapiens and Mus musculus account for the majority of samples; additional organisms include rat, zebrafish, and non-human primates.

d, Top-15 organs by sample count. Kidney, brain, lung, and liver are the most represented tissues.

e, Polarity breakdown. Both positive- and negative-ion mode acquisitions are included, reflecting the complementary metabolite coverage of each mode.

f, Disease-condition breakdown, grouped as tumor/cancer, healthy/wildtype, other named conditions, or unknown (descriptive corpus composition only; condition is not used as a classification target in this study).
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


def load_meta():
    ch = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                     usecols=["sample_path", "organism", "polarity",
                               "Organism_Part", "analyzerType", "ionisationSource",
                               "Condition"])
    return ch.drop_duplicates("sample_path").reset_index(drop=True)


def _hbar(ax, labels, values, color, title, xlabel="Samples (n)"):
    idx = range(len(labels))
    bars = ax.barh(list(idx), values, color=color, height=0.65, edgecolor="none")
    ax.set_yticks(list(idx))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    for bar, v in zip(bars, values):
        ax.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(v):,}", va="center", fontsize=7)


def draw_panel_a(ax, df):
    counts = (df["ionisationSource"]
              .str.strip().replace({"SIMS (Bi3+)": "SIMS"})
              .replace("", "Unknown").fillna("Unknown")
              .value_counts())
    counts = counts[counts.index != "Unknown"].head(10)
    _hbar(ax, counts.index.tolist(), counts.values.tolist(), BAR_COLOR,
          "A   Samples by Ionisation Source")


def draw_panel_b(ax, df):
    # Simplify analyzer names to broad families
    def _family(s):
        s = str(s).strip()
        if "Orbitrap" in s or "Exploris" in s:    return "Orbitrap"
        if "FTICR" in s or "FT-ICR" in s or "FTMS" in s: return "FT-ICR"
        if "timsTOF" in s:                         return "timsTOF"
        if "TOF" in s:                             return "TOF"
        if "qTOF" in s or "Q-TOF" in s:           return "Q-TOF"
        return "Other"
    df2 = df.copy()
    df2["analyzer_family"] = df2["analyzerType"].apply(_family)
    counts = df2["analyzer_family"].value_counts()
    colors = PALETTE[:len(counts)]
    _hbar(ax, counts.index.tolist(), counts.values.tolist(), colors[0],
          "B   Samples by Analyzer Type")
    for i, (bar_val, col) in enumerate(zip(counts.values, colors)):
        ax.patches[i].set_facecolor(col)


def draw_panel_c(ax, df):
    def _org(s):
        s = str(s).strip()
        if "sapiens" in s or "human" in s.lower():   return "Homo sapiens"
        if "musculus" in s or "Mouse" in s:           return "Mus musculus"
        if "norvegicus" in s:                         return "Rattus norvegicus"
        if "rerio" in s:                              return "Danio rerio"
        if "primate" in s.lower():                    return "Non-human primate"
        if s in ("", "nan", "Mixed"):                 return "Mixed / Unknown"
        return s
    df2 = df.copy()
    df2["org"] = df2["organism"].apply(_org)
    counts = df2["org"].value_counts()
    _hbar(ax, counts.index.tolist(), counts.values.tolist(), ALT_COLOR,
          "C   Samples by Organism")


def draw_panel_d(ax, df):
    def _norm(s):
        return {"Kideny": "Kidney", "colon": "Colon"}.get(str(s).strip(), str(s).strip())
    df2 = df.copy()
    df2["organ"] = df2["Organism_Part"].apply(_norm)
    counts = df2["organ"].value_counts().head(15)
    colors = PALETTE[:len(counts)]
    _hbar(ax, counts.index.tolist(), counts.values.tolist(), colors[0],
          "D   Top-15 Organs by Sample Count")
    for i, col in enumerate(colors):
        ax.patches[i].set_facecolor(col)


def draw_panel_e(ax, df):
    counts = df["polarity"].str.strip().value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values.tolist()
    colors = [BAR_COLOR, ALT_COLOR][:len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax.set_title("E   Ionisation Polarity", fontsize=11, fontweight="bold", pad=6)


def draw_panel_f(ax, df):
    """Disease-condition breakdown, grouped the same way as the tumor/cancer
    vs. healthy/wildtype rule described in Methods (descriptive corpus
    composition only -- condition is not used as a classification target)."""
    def _group(s):
        s = str(s).strip().lower()
        if s in ("tumor", "cancer"):
            return "Tumor / cancer"
        if s in ("healthy", "wildtype", "wtype", "wild type", "normal"):
            return "Healthy / wildtype"
        if s in ("", "nan", "unknown"):
            return "Unknown"
        return "Other"
    df2 = df.copy()
    df2["condition_group"] = df2["Condition"].apply(_group)
    counts = df2["condition_group"].value_counts()
    order = [c for c in ["Tumor / cancer", "Healthy / wildtype", "Other", "Unknown"] if c in counts.index]
    counts = counts.reindex(order)
    colors = [PALETTE[i] for i in range(len(counts))]
    _hbar(ax, counts.index.tolist(), counts.values.tolist(), colors[0],
          "F   Samples by Disease Condition")
    for i, col in enumerate(colors):
        ax.patches[i].set_facecolor(col)


def main():
    df = load_meta()
    print(f"Loaded {len(df):,} unique samples")

    # Panel A
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    draw_panel_a(ax, df)
    save_panel(fig, "figS1_panelA_ionisation_source")
    plt.close(fig)

    # Panel B
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    draw_panel_b(ax, df)
    save_panel(fig, "figS1_panelB_analyzer_type")
    plt.close(fig)

    # Panel C
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    draw_panel_c(ax, df)
    save_panel(fig, "figS1_panelC_organism")
    plt.close(fig)

    # Panel D
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    draw_panel_d(ax, df)
    save_panel(fig, "figS1_panelD_organs")
    plt.close(fig)

    # Panel E
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_panel_e(ax, df)
    save_panel(fig, "figS1_panelE_polarity")
    plt.close(fig)

    # Panel F
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    draw_panel_f(ax, df)
    save_panel(fig, "figS1_panelF_condition")
    plt.close(fig)

    write_caption()
    print("FigS1 done.")


if __name__ == "__main__":
    main()

