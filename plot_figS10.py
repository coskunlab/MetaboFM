"""
plot_figS10.py
-------------
Supplementary Figure S10: Retrieval Robustness.

Panels:
  A  Leave-platform-out channel retrieval R@1 per platform (Stage 2 vs Stage 1 vs Random)
  B  Leave-acquisition-out R@1 per organ (Stage 2 vs Stage 1)
  C  Cross-platform channel retrieval R@k curves by platform
  D  Strict leave-one-study-out retrieval (METASPACE submitter-level exclusion),
     directly responding to the reviewer request for leave-study-out /
     leave-laboratory-out validation

Usage:
  conda run -n torch_gpu python plot_figS10.py
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
RET_DIR    = METABOFM_ROOT / "outputs/crossdataset_retrieval"
PATCH_DIR  = METABOFM_ROOT / "outputs/spatial_patches"
XPLAT_DIR  = METABOFM_ROOT / "outputs/crossplatform_retrieval"
LSO_DIR    = METABOFM_ROOT / "outputs/leave_study_out"
OUT_DIR    = METABOFM_ROOT / "outputs/figures"
PANEL_DIR  = OUT_DIR / "figS10_retrieval_robustness"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI       = 300
S2_COLOR  = "#2166ac"
S1_COLOR  = "#d6604d"
RND_COLOR = "#aaaaaa"

IMAGENET_COLOR = "#9970ab"

VARIANT_STYLE = {
    "stage2_ch_refined": (S2_COLOR,  "Stage 2 (channel-refined)", "o"),
    "stage1_cls":        (S1_COLOR,  "Stage 1 (channel)",         "s"),
    "imagenet":          (IMAGENET_COLOR, "ImageNet ResNet",      "d"),
    "mz_soft":           ("#4dac26", "m/z baseline",               "^"),
    "random":            (RND_COLOR, "Random",                     "x"),
    "Stage 2":           (S2_COLOR,  "Stage 2 (channel-refined)", "o"),
    "Stage 1":           (S1_COLOR,  "Stage 1 (channel)",         "s"),
    "ImageNet":          (IMAGENET_COLOR, "ImageNet ResNet",      "d"),
    "Random":            (RND_COLOR, "Random",                     "x"),
}


CAPTION = """\
Supplementary Figure 10 | Retrieval robustness across acquisition platforms, organ types, and real study identity.

Organ labels serve as a verifiable ground-truth proxy for cross-dataset transfer. The goal of this figure is to demonstrate that MetaboFM embeddings are consistent across instruments and acquisitions, and to report a stricter, real study-level validation directly in response to reviewer request.

a, Leave-platform-out channel retrieval Recall@1 by platform and embedding variant (Stage 2, Stage 1, m/z baseline, random). For each held-out acquisition platform, all samples from that platform serve as queries and the remaining samples form the gallery. Recall@1 remains above chance across all platforms for the learned representations, demonstrating that embeddings transfer to instruments not seen during retrieval-index construction; the m/z baseline is shown for direct comparison.

b, Per-organ leave-one-acquisition-out Recall@1 for all annotated organs, comparing Stage 2 and Stage 1, sorted ascending by Stage 2 performance. Stage 2 outperforms Stage 1 for the majority of organs; organs where Stage 1 leads are also shown, including rare tissue types with fewer cross-acquisition examples.

c, Cross-platform channel-level retrieval Recall@1 by platform and embedding variant (Stage 2, Stage 1, m/z baseline, random). MetaboFM Stage 2 outperforms baselines across the majority of platforms for the channel-level retrieval task, confirming that instrument-agnostic metabolic channel embeddings are learned at both the sample and channel levels.

d, Strict leave-one-study-out retrieval, directly responding to the request for leave-study-out, leave-laboratory-out, or leave-platform-out validation. Study identity is the METASPACE submitting researcher, which excludes all other acquisitions by the same researcher from the query's gallery, not only the query's own file (335 distinct submitters span the corpus). Under this stricter exclusion, absolute Recall@1 is substantially lower than under leave-one-acquisition-out (panel b) for both Stage 2 (macro 0.188, weighted 0.562) and Stage 1 (macro 0.170, weighted 0.559), and an m/z-only baseline performs comparably or better (macro 0.274, weighted 0.687), indicating that a meaningful share of the leave-one-acquisition-out retrieval margin over baselines reflects researcher/acquisition-batch signatures rather than organ biology alone, and that mass alone remains informative of organ identity within this corpus. All variants remain well above the random baseline (dashed lines).
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


def draw_panel_a(ax):
    """Cross-platform channel retrieval R@1 per platform, Stage 2 vs Stage 1 vs m/z vs Random."""
    df = pd.read_csv(XPLAT_DIR / "crossplatform_retrieval_results.csv")
    show_variants = ["stage2_ch_refined", "stage1_cls", "mz_soft", "random"]
    labels        = {"stage2_ch_refined": "Stage 2 (channel-refined)", "stage1_cls": "Stage 1 (channel)",
                      "mz_soft": "m/z baseline", "random": "Random"}
    colors        = {"stage2_ch_refined": S2_COLOR, "stage1_cls": S1_COLOR,
                      "mz_soft": "#4dac26", "random": RND_COLOR}

    platforms = (df[df["variant"] == "stage2_ch_refined"]
                 .sort_values("R@1", ascending=True)["platform"].tolist())
    y = np.arange(len(platforms))
    n = len(show_variants)
    w = 0.8 / n

    for i, var in enumerate(show_variants):
        sub  = df[df["variant"] == var].set_index("platform")
        vals = [sub.loc[p, "R@1"] if p in sub.index else np.nan for p in platforms]
        ax.barh(y + (i - (n - 1) / 2) * w, vals, w * 0.9,
                color=colors[var], alpha=0.85, label=labels[var])

    ax.set_yticks(y)
    ax.set_yticklabels(platforms, fontsize=8)
    ax.set_xlabel("R@1 (leave-platform-out)", fontsize=9)
    ax.set_title("A   Leave-platform-out Channel Retrieval R@1\n(Stage 2 vs Stage 1 vs m/z vs Random)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0.5, color="#888", lw=0.8, ls="--")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8, frameon=False)


def draw_panel_b(ax):
    """Per-organ leave-acquisition-out R@1 with Stage 2 vs Stage 1."""
    df = pd.read_csv(RET_DIR / "crossdataset_retrieval_pivot_r1.csv")
    # columns: organ, Stage 1, Stage 2 (and possibly more)
    s2_col = [c for c in df.columns if "Stage 2" in c or "stage2" in c.lower()]
    s1_col = [c for c in df.columns if "Stage 1" in c or "stage1" in c.lower()]
    organ_col = df.columns[0]

    if not s2_col or not s1_col:
        ax.text(0.5, 0.5, "Could not find Stage 1 / Stage 2 columns",
                ha="center", va="center", transform=ax.transAxes)
        return
    s2_col, s1_col = s2_col[0], s1_col[0]

    df = df.dropna(subset=[s2_col, s1_col])
    df = df[df[organ_col] != "MACRO_AVG"]
    df = df.sort_values(s2_col, ascending=True)

    y = np.arange(len(df))
    h = 0.35
    ax.barh(y + h/2, df[s2_col].values, h, color=S2_COLOR, alpha=0.85, label="Stage 2")
    ax.barh(y - h/2, df[s1_col].values, h, color=S1_COLOR, alpha=0.85, label="Stage 1")
    ax.set_yticks(y)
    ax.set_yticklabels(df[organ_col].tolist(), fontsize=7.5)
    ax.set_xlabel("Recall@1 (leave-one-acquisition-out)", fontsize=9)
    ax.set_title("B   Per-organ Leave-acquisition-out Retrieval R@1",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", labelsize=8)


def draw_panel_c(ax):
    """Cross-platform channel retrieval R@1 by platform and variant."""
    df = pd.read_csv(XPLAT_DIR / "crossplatform_retrieval_results.csv")
    variants = [v for v in VARIANT_STYLE if v in df["variant"].unique() and v != "imagenet"]
    platforms = df["platform"].unique()
    x = np.arange(len(platforms))
    n = len(variants)
    width = 0.8 / n

    for i, var in enumerate(variants):
        sub = df[df["variant"] == var].set_index("platform")
        vals = [sub.loc[p, "R@1"] if p in sub.index else np.nan for p in platforms]
        col, label, _ = VARIANT_STYLE[var]
        ax.bar(x + (i - n/2 + 0.5) * width, vals, width * 0.9,
               color=col, alpha=0.85, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(platforms, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("R@1", fontsize=9)
    ax.set_title("C   Cross-platform Channel Retrieval R@1 by Platform",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="y", labelsize=8)


def draw_panel_d(ax):
    """Strict leave-one-study-out retrieval (METASPACE submitter-level
    exclusion): macro and weighted Recall@1 for Stage 2, Stage 1, and an
    m/z-only baseline, with random-baseline reference lines."""
    df = pd.read_csv(LSO_DIR / "leavestudyout_overall.csv").set_index("variant")
    variants = ["Stage 2", "Stage 1", "m/z"]
    colors   = {"Stage 2": S2_COLOR, "Stage 1": S1_COLOR, "m/z": "#4dac26"}

    metrics = ["macro_recall@1", "weighted_recall@1"]
    metric_labels = ["Macro R@1", "Weighted R@1"]
    x = np.arange(len(metrics))
    n = len(variants)
    width = 0.8 / n

    for i, var in enumerate(variants):
        vals = [df.loc[var, m] for m in metrics]
        ax.bar(x + (i - n/2 + 0.5) * width, vals, width * 0.9,
               color=colors[var], alpha=0.85, label=var)
        for xi, v in zip(x + (i - n/2 + 0.5) * width, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # random baselines as reference lines per metric group
    rnd_macro = df.loc["Stage 2", "macro_random@1"]
    rnd_weighted = df.loc["Stage 2", "weighted_random@1"]
    ax.plot([x[0] - 0.4, x[0] + 0.4], [rnd_macro]*2, color="#888", lw=1.2, ls="--")
    ax.plot([x[1] - 0.4, x[1] + 0.4], [rnd_weighted]*2, color="#888", lw=1.2, ls="--",
            label="Random baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Recall@1", fontsize=9)
    ax.set_title("D   Strict Leave-one-study-out Retrieval\n(METASPACE submitter-level exclusion, 335 studies)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, max(df[["macro_recall@1", "weighted_recall@1"]].values.max(), rnd_weighted) + 0.08)


def main():
    # Panel A
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    draw_panel_a(ax)
    save_panel(fig, "figS10_panelA_leave_platform_out")
    plt.close(fig)

    # Panel B
    n_organs = len(pd.read_csv(RET_DIR / "crossdataset_retrieval_pivot_r1.csv").dropna()) - 1
    fig_h    = max(7.0, n_organs * 0.28)
    fig, ax  = plt.subplots(figsize=(7.0, fig_h))
    draw_panel_b(ax)
    save_panel(fig, "figS10_panelB_per_organ_leave_study")
    plt.close(fig)

    # Panel C
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    draw_panel_c(ax)
    save_panel(fig, "figS10_panelC_crossplatform_r1")
    plt.close(fig)

    # Panel D
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    draw_panel_d(ax)
    save_panel(fig, "figS10_panelD_leave_study_out_strict")
    plt.close(fig)

    write_caption()
    print("FigS10 done.")


if __name__ == "__main__":
    main()

