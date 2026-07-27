"""
plot_figS11.py
-------------
Supplementary Figure S11: UMAP Coloured by Technical Covariates.

Shows that Stage 2 sample embeddings are NOT driven by acquisition platform,
polarity, or study origin — complementing Figure 5B/C which shows organ/organism
structure.

Panels:
  A  UMAP coloured by ionisation source
  B  UMAP coloured by polarity (positive / negative)
  C  UMAP coloured by analyzer family
  D  LISI scores per covariate (Stage 2 vs Stage 1) — quantitative mixing metric

Usage:
  conda run -n torch_gpu python plot_figS11.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plot_utils import set_nature_style
set_nature_style()

# -- CONFIG -------------------------------------------------------------------
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
UMAP_DIR = METABOFM_ROOT / "outputs/sample_umap"
OUT_DIR  = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS11_umap_technical_covariates"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI      = 300
PT_SIZE  = 3
PT_ALPHA = 0.55
CLIP_PCT = 1

PALETTE = [
    "#2166ac", "#d6604d", "#4dac26", "#fdae6b", "#9970ab",
    "#1b7837", "#e08214", "#74add1", "#a50026", "#35978f",
    "#8c510a", "#01665e",
]
S2_COLOR = "#2166ac"
S1_COLOR = "#d6604d"


CAPTION = """\
Supplementary Figure 11 | Stage 2 sample embeddings show partial mixing of technical covariates.

a, UMAP projection of Stage 2 CLS embeddings coloured by ionisation source. Samples from different ionisation modalities are broadly interleaved, though MALDI dominates the corpus (~85% of samples), so apparent clustering partly reflects this imbalance rather than platform bias.

b, UMAP coloured by measurement polarity (positive / negative ion mode). Both Stage 1 and Stage 2 achieve high polarity mixing (LISI 1.72/2.0 and 1.62/2.0, respectively; panel d), indicating that the embedding captures metabolic identity rather than polarity-specific spectral patterns.

c, UMAP coloured by mass analyser family. Samples from different instrument classes are largely interleaved, with remaining structure attributable to instrument–tissue co-occurrence in the training corpus rather than instrument-specific encoding.

d, Local Inverse Simpson Index (LISI) for each technical covariate in Stage 2 and Stage 1 UMAP space. LISI is normalised by the number of unique labels per covariate (dashed line = perfect mixing). Higher LISI indicates greater mixing of the covariate across local neighbourhoods. Polarity is well mixed in both stages (≥81% of maximum); ionisation source and analyser show lower mixing, consistent with the corpus imbalance observed in panel a.
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


def _clip(ax, df):
    x, y = df["umap_x"].values, df["umap_y"].values
    ax.set_xlim(np.percentile(x, CLIP_PCT), np.percentile(x, 100 - CLIP_PCT))
    ax.set_ylim(np.percentile(y, CLIP_PCT), np.percentile(y, 100 - CLIP_PCT))


def _umap_categorical(ax, df, col, title, max_cats=10):
    counts = df[col].value_counts()
    top    = counts.head(max_cats).index.tolist()
    df2    = df.copy()
    df2[col] = df2[col].where(df2[col].isin(top), other="Other")
    cats   = df2[col].value_counts().index.tolist()
    # put Other last
    if "Other" in cats:
        cats = [c for c in cats if c != "Other"] + ["Other"]
    color_map = {c: (PALETTE[i % len(PALETTE)] if c != "Other" else "#cccccc")
                 for i, c in enumerate(cats)}

    ax.scatter(df2["umap_x"], df2["umap_y"],
               c=[color_map[v] for v in df2[col]],
               s=PT_SIZE, alpha=PT_ALPHA, linewidths=0, rasterized=True)
    _clip(ax, df2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    patches = [mpatches.Patch(color=color_map[c], label=c) for c in cats]
    ax.legend(handles=patches, fontsize=6.5, frameon=False,
              loc="lower right", ncol=1, markerscale=1.4,
              handlelength=1.0, handleheight=0.8)




def draw_panel_d(ax):
    """LISI scores per covariate — Stage 2 vs Stage 1."""
    lisi = pd.read_csv(UMAP_DIR / "lisi_scores.csv")

    covariates = lisi["covariate"].unique().tolist()
    y          = np.arange(len(covariates))
    h          = 0.35

    s2 = lisi[lisi["stage"] == "Stage 2"].set_index("covariate")
    s1 = lisi[lisi["stage"] == "Stage 1"].set_index("covariate")

    for i, cov in enumerate(covariates):
        n_lab = int(s2.loc[cov, "n_labels"])
        s2_norm = s2.loc[cov, "lisi_mean"] / n_lab
        s1_norm = s1.loc[cov, "lisi_mean"] / n_lab
        ax.barh(y[i] + h / 2, s2_norm, h * 0.9, color=S2_COLOR, alpha=0.85,
                label="Stage 2" if i == 0 else "")
        ax.barh(y[i] - h / 2, s1_norm, h * 0.9, color=S1_COLOR, alpha=0.85,
                label="Stage 1" if i == 0 else "")

    ax.axvline(1.0, color="#888", lw=0.9, ls="--", label="Perfect mixing")
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(covariates, fontsize=9)
    ax.set_xlabel("LISI / max possible (normalised)", fontsize=9)
    ax.set_title("D   Quantitative Mixing: LISI by Covariate\n(Stage 2 vs Stage 1)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", labelsize=8)


def load_data():
    coords = np.load(str(UMAP_DIR / "umap2d_stage2.npy"))
    ch = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                     usecols=["sample_path", "organism", "polarity",
                               "Organism_Part", "analyzerType", "ionisationSource"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    sm   = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv").merge(
               samp, on="sample_path", how="left")
    assert len(sm) == len(coords), f"UMAP length mismatch: {len(coords)} vs {len(sm)}"
    sm["umap_x"] = coords[:, 0]
    sm["umap_y"] = coords[:, 1]

    def _family(s):
        s = str(s).strip()
        if "Orbitrap" in s or "Exploris" in s:         return "Orbitrap"
        if "FTICR" in s or "FT-ICR" in s or "FTMS" in s: return "FT-ICR"
        if "timsTOF" in s:                              return "timsTOF"
        if "Q-TOF" in s or "qTOF" in s:               return "Q-TOF"
        if "TOF" in s:                                  return "TOF"
        return "Other"
    sm["analyzer_family"] = sm["analyzerType"].apply(_family)
    sm["ionisation_clean"] = sm["ionisationSource"].str.strip().fillna("Unknown")
    sm["polarity_clean"]   = sm["polarity"].str.strip().fillna("Unknown")
    return sm


def main():
    df = load_data()

    # Panel A — ionisation source
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    _umap_categorical(ax, df, "ionisation_clean",
                      "A   Sample Embeddings (UMAP)\nColoured by Ionisation Source")
    save_panel(fig, "figS11_panelA_umap_ionisation")
    plt.close(fig)

    # Panel B — polarity
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    _umap_categorical(ax, df, "polarity_clean",
                      "B   Sample Embeddings (UMAP)\nColoured by Polarity",
                      max_cats=5)
    save_panel(fig, "figS11_panelB_umap_polarity")
    plt.close(fig)

    # Panel C — analyzer family
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    _umap_categorical(ax, df, "analyzer_family",
                      "C   Sample Embeddings (UMAP)\nColoured by Analyzer Type")
    save_panel(fig, "figS11_panelC_umap_analyzer")
    plt.close(fig)

    # Panel D — LISI quantitative mixing
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    draw_panel_d(ax)
    save_panel(fig, "figS11_panelD_lisi_mixing")
    plt.close(fig)

    write_caption()
    print("FigS4 done.")


if __name__ == "__main__":
    main()


