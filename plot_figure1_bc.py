"""
plot_figure1_bc.py
-------------------
Builds Figure 1 panels b (corpus composition Sankey) and c (HMDB
super-class prevalence) from the full training corpus.

Panel b: a treemap of Organism -> Organ (joint composition, n=5,600 samples,
deduplicated by sample_path from stage2_channel_meta.csv), plus large-font
stat badges for ionisationSource/analyzerType/polarity diversity. This
deliberately does NOT duplicate Supplementary Fig. S1, which shows organism
and organ as separate marginal bar charts (not their joint distribution) --
a Sankey of all six metadata fields was tried first and rejected: with up to
19 analyzer-type categories and 88 organs, no font size stayed legible once
shrunk to a compact multi-panel figure slot, since text size on a Sankey
node is fixed regardless of panel size. A treemap's text auto-scales to box
area, so large categories stay legible and small ones just shrink instead
of colliding. Same "Kideny"->"Kidney" and "Mouse Brain" organism fixes
applied elsewhere in this project are applied here too.

Panel c: channel-level HMDB super-class counts (n=158,405), matching the
corpus scale used everywhere else in the manuscript. A small number of
channels (6/158,405, 0.004%) have a corrupted/truncated label in the
source cache file and are dropped rather than guessed.
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_utils import set_nature_style

set_nature_style()

EMB_DIR = METABOFM_ROOT / "outputs/embeddings_v2"
HMDB_CACHE = METABOFM_ROOT / "outputs/benchmarks_v2/_hmdb_cache/hmdb__super_class.csv"
OUT_DIR = METABOFM_ROOT / "outputs/figures/figure1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300


def save_panel(fig, stem):
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    fig.savefig(str(OUT_DIR / f"{stem}.svg"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved panel {stem}.svg")


# ============================================================
# PANEL B: Organism -> Organ treemap + technical-diversity stat badges
# ============================================================

FACTORS_FOR_META = ["ionisationSource", "analyzerType", "polarity", "organism", "Organism_Part"]
TOPK_ORGANS_PER_ORGANISM = 6
ORGANISM_COLORS = {"Homo sapiens": "#2166ac", "Mus musculus": "#d6604d", "Other organism": "#999999"}


def _load_sample_meta() -> pd.DataFrame:
    ch = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                     usecols=["sample_path"] + FACTORS_FOR_META)
    sm = ch.drop_duplicates("sample_path").reset_index(drop=True)
    assert len(sm) == 5600, f"expected 5600 samples, got {len(sm)}"

    sm["ionisationSource"] = sm["ionisationSource"].replace({"SIMS (Bi3+)": "SIMS"})
    sm["analyzerType"] = sm["analyzerType"].replace({"FTICR": "FT-ICR", "qTOF": "Q-TOF"})

    # Same organism/organ data-entry fix applied elsewhere in this project:
    # a handful of rows have organism mistakenly set to "Mouse Brain".
    crossed = sm["organism"].astype(str).str.fullmatch(r"Mouse\s+Brain", case=False, na=False)
    sm.loc[crossed, "organism"] = "Mus musculus"
    sm.loc[crossed & sm["Organism_Part"].isna(), "Organism_Part"] = "Brain"
    sm["Organism_Part"] = sm["Organism_Part"].replace({"Kideny": "Kidney", "colon": "Colon"})

    for f in FACTORS_FOR_META:
        sm[f] = sm[f].fillna("Other")

    return sm


def draw_panel_b_class_counts(ax, sm: pd.DataFrame):
    """Simple bar chart: number of distinct classes per metadata field."""
    fields = [
        ("ionisationSource", "Ionization source"),
        ("analyzerType", "Analyzer type"),
        ("polarity", "Polarity"),
        ("organism", "Organism"),
        ("Organism_Part", "Organ"),
    ]
    names = [disp for _, disp in fields]
    counts = [sm[col].nunique() for col, _ in fields]

    y = np.arange(len(fields))
    ax.barh(y, counts, color="#4C72B0", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Number of distinct categories", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    for yi, c in zip(y, counts):
        ax.text(c + max(counts) * 0.01, yi, f"{c}", va="center", fontsize=9, color="#333")


# ============================================================
# PANEL C: HMDB super-class prevalence (channel-level, n=158,405)
# ============================================================

def draw_panel_c_hmdb(ax):
    df = pd.read_csv(HMDB_CACHE)
    df = df[~df["label"].str.contains(r"\[TRUNC\]", regex=True, na=False)]
    vc = df["label"].value_counts()
    vc = vc.rename(index={"unknown": "Unclassified / no HMDB match"})

    MIN_N = 20  # fold genuinely negligible tail categories (<0.02% of 158,405) into "Other"
    kept = vc[vc >= MIN_N]
    other_n = int(vc[vc < MIN_N].sum())
    if other_n:
        kept["Other"] = other_n
    kept = kept.sort_values(ascending=True)

    colors = ["#bdbdbd" if k in ("Unclassified / no HMDB match", "Other") else "#4C72B0" for k in kept.index]
    y = np.arange(len(kept))
    ax.barh(y, kept.values, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(kept.index, fontsize=8)
    ax.set_xlabel("Number of ion channels", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    total = int(vc.sum())
    for yi, v in zip(y, kept.values):
        ax.text(v + total * 0.005, yi, f"{v:,}", va="center", fontsize=7, color="#333")
    print(f"  n={total:,} channels total; {len(kept)} classes shown "
          f"({'Other' if other_n else 'no'} tail bucket = {other_n:,})")


def main():
    print("=== Figure 1 panel B: organism/organ bars + stat badges ===")
    sm = _load_sample_meta()
    print(f"  {len(sm)} samples; ionisationSource={sm['ionisationSource'].nunique()} categories, "
          f"analyzerType={sm['analyzerType'].nunique()} categories")

    fig, ax = plt.subplots(figsize=(5, 4))
    draw_panel_b_class_counts(ax, sm)
    fig.tight_layout()
    save_panel(fig, "figure1_panelB_class_counts")

    print("=== Figure 1 panel C: HMDB super-class prevalence ===")
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_panel_c_hmdb(ax)
    fig.tight_layout()
    save_panel(fig, "figure1_panelC_hmdb_prevalence")

    print(f"[DONE] Figure 1 panels b/c -> {OUT_DIR}")


if __name__ == "__main__":
    main()
