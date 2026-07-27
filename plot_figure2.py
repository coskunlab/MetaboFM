"""
plot_figure2.py
---------------
Figure 2: Benchmark performance across all variants.

Panels (saved individually, no composite figure):
  B  HMDB super_class macro-F1  (horizontal bars, linear probe, unambiguous subset)
  C  HMDB super_class MAP@10    (horizontal bars, retrieval, unambiguous subset)
  D  Cross-platform consistency (within-platform minus cross-platform cosine delta)
  E  Representative tissue ion images (one per organ: Brain, Kidney, Liver, Lung)

Supplementary content previously embedded here has been split out:
  - Full-dataset HMDB benchmark (all channels)      -> plot_figS3.py
  - Leave-analyzerType-out organ classification F1  -> plot_fig_leave_platform_organ.py
Run those separately if you need to regenerate them.

Usage:
  python plot_figure2.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_utils import set_nature_style, load_best_channel, pick_best_sample, add_scale_bar
set_nature_style()

# ── CONFIG ───────────────────────────────────────────────────────────────────
BENCH_DIR   = METABOFM_ROOT / "outputs/benchmarks_v2"
XPLAT_DIR   = METABOFM_ROOT / "outputs/crossplatform_consistency"
EMB_DIR     = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR     = METABOFM_ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR   = OUT_DIR / "figure2"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

# ── VARIANT DISPLAY CONFIG (Panels B & C) ────────────────────────────────────
# The "__unambig"-suffixed variants (and smiles_only) were extracted only for the
# n_cand==1 rows (n=35,856) at the time of extraction. Their own row_ids ARE that
# unambiguous set, so the "[all]" bracket (no additional masking) is the correct,
# non-redundant evaluation. Re-applying the CURRENT unambiguous_mask on top via the
# "[unambiguous]" bracket double-filters against a since-drifted candidate-matching
# pipeline (only ~22% row overlap with the original extraction), which silently
# evaluates on a tiny, arbitrary subset instead of the full unambiguous set — this
# was a real bug (see July 2026 data-consistency investigation) and must NOT be used
# for the __unambig-suffixed variants. mz_only and metadata_only were computed fresh
# over the full current corpus, so their own "[unambiguous]" bracket IS the single,
# correct current-mask application and should be kept as-is.
VARIANT_LABELS = {
    "stage2_ch_refined__unambig[all]": "Stage 2 (channel-refined)",
    "resnet+smiles[all]":              "ResNet + SMILES",
    "resnet_only__unambig[all]":       "Stage 1 (channel)",
    "mz_only[unambiguous]":            "m/z only",
    "metadata_only[unambiguous]":      "Metadata only",
    "smiles_only[all]":                "SMILES only (structure baseline)",
    "imagenet__unambig[all]":          "ImageNet ResNet",
}
VARIANT_ORDER = list(VARIANT_LABELS.keys())

COLORS = {
    "stage2_ch_refined__unambig[all]": "#2166ac",
    "resnet+smiles[all]":              "#4dac26",
    "resnet_only__unambig[all]":       "#74add1",
    "mz_only[unambiguous]":            "#d6604d",
    "metadata_only[unambiguous]":      "#b2b2b2",
    "smiles_only[all]":                "#f4a582",
    "imagenet__unambig[all]":          "#c2c2c2",
}
HATCH = {
    "mz_only[unambiguous]":            "//",
    "smiles_only[all]":                "//",
}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_probe():
    df = pd.read_csv(BENCH_DIR / "linear_probe" / "summary.csv")
    return df[df["field"] == "super_class"].set_index("variant")

def load_retrieval():
    df = pd.read_csv(BENCH_DIR / "retrieval" / "summary.csv")
    return df[df["field"] == "super_class"].set_index("variant")

def load_crossplatform():
    df = pd.read_csv(XPLAT_DIR / "summary.csv")
    return df.set_index(["variant", "group"])["mean"]

# ── PANELS B & C: horizontal bar chart ───────────────────────────────────────

def draw_bars(ax, summary_df, val_col, err_col, title, xlabel, xlim):
    y = np.arange(len(VARIANT_ORDER))

    for i, v in enumerate(VARIANT_ORDER):
        if v not in summary_df.index:
            continue
        val = float(summary_df.loc[v, val_col])
        err = float(summary_df.loc[v, err_col]) if err_col in summary_df.columns else 0.0
        color = COLORS.get(v, "#888")
        hatch = HATCH.get(v, None)

        ax.barh(
            i, val, xerr=err,
            height=0.62,
            color=color, hatch=hatch,
            edgecolor="white" if hatch is None else color,
            linewidth=0.4,
            error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444"),
        )
        ax.text(
            min(val + max(err, 0) + 0.005, xlim - 0.01), i,
            f"{val:.3f}",
            va="center", ha="left", fontsize=9, color="#222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0, xlim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

# ── PANEL D: cross-platform consistency ──────────────────────────────────────

def draw_crossplatform(ax_raw, ax_delta, xplat):
    """Two-part cross-platform consistency panel.

    (i)  Raw mean cosine similarity for Stage 1 mean-pool sample embeddings
         across the three same/cross-tissue x same/cross-platform pair groups
         -- the values discussed directly in the main text.
    (ii) Same-tissue-diff-platform minus diff-tissue-diff-platform delta
         (group B - group C) for Stage 1 vs Stage 2, i.e. how much better each
         representation separates tissue identity from platform identity.
         Deltas are used here (not raw values) because Stage 1 mean-pooled
         embeddings have a uniformly higher baseline similarity unrelated to
         tissue discriminability, which makes a raw-value comparison across
         variants misleading -- see plot_figS4.py draw_panel_c docstring for
         the original diagnosis of this issue.
    """
    groups = [
        ("A_same_tissue_same_platform", "Same tissue,\nsame platform"),
        ("B_same_tissue_diff_platform", "Same tissue,\ndiff. platform"),
        ("C_diff_tissue_diff_platform", "Diff. tissue,\ndiff. platform"),
    ]
    variant = "stage1_meanpool"
    vals = [float(xplat.get((variant, g), np.nan)) for g, _ in groups]
    x = np.arange(len(groups))

    bars = ax_raw.bar(x, vals, color="#74add1", width=0.55,
                      edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax_raw.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#222")

    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels([lbl for _, lbl in groups], fontsize=8.5)
    ax_raw.set_ylabel("Mean cosine similarity\n(Stage 1 mean-pool)", fontsize=9.5)
    ax_raw.set_title("D (i)   Cross-platform\nConsistency", fontsize=11,
                     fontweight="bold", pad=6)
    ax_raw.spines["top"].set_visible(False)
    ax_raw.spines["right"].set_visible(False)
    ax_raw.tick_params(axis="both", labelsize=8.5)
    valid = [v for v in vals if not np.isnan(v)]
    ax_raw.set_ylim(0, (max(valid) if valid else 1.0) + 0.06)

    # (ii) B - C delta, Stage 1 vs Stage 2
    b = xplat.xs("B_same_tissue_diff_platform", level="group")
    c = xplat.xs("C_diff_tissue_diff_platform", level="group")
    delta = (b - c)
    cp_variants = ["stage1_meanpool", "stage2"]
    cp_labels   = ["Stage 1\n(mean-pool)", "Stage 2\n(sample)"]
    cp_colors   = ["#74add1", "#2166ac"]
    cp_vals     = [float(delta.get(v, np.nan)) for v in cp_variants]

    bars2 = ax_delta.bar(cp_labels, cp_vals, color=cp_colors, width=0.5,
                         edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, cp_vals):
        if not np.isnan(val):
            ax_delta.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                          f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#222")

    ax_delta.set_ylabel("Same-tissue $-$ diff-tissue\ncosine similarity ($\\Delta$)", fontsize=9.5)
    ax_delta.set_title("D (ii)   Tissue Discriminability\nAcross Platforms", fontsize=11,
                       fontweight="bold", pad=6)
    ax_delta.spines["top"].set_visible(False)
    ax_delta.spines["right"].set_visible(False)
    ax_delta.tick_params(axis="both", labelsize=8.5)
    valid2 = [v for v in cp_vals if not np.isnan(v)]
    ax_delta.set_ylim(0, (max(valid2) if valid2 else 0.1) * 1.4)

# ── PANEL E: representative tissue ion images ────────────────────────────────

GALLERY_ORGANS  = ["Brain", "Kidney", "Liver", "Lung", "Skin", "Breast"]
GALLERY_COLORS  = ["#2166ac", "#9970ab", "#e08214", "#4dac26", "#d6604d", "#35978f"]


def save_spatial_gallery_panels(ch_meta):
    """Panel E: one best-channel ion image per organ (Brain/Kidney/Liver/Lung),
    saved individually — no composite figure required."""
    for organ, col in zip(GALLERY_ORGANS, GALLERY_COLORS):
        rows = ch_meta[ch_meta["Organism_Part"] == organ]
        sp   = pick_best_sample(rows["sample_path"].unique())
        img  = load_best_channel(sp) if sp else None
        if img is None:
            print(f"  [SKIP] panel E organ={organ}: no sample found")
            continue
        fig_s, ax_s = plt.subplots(figsize=(3, 3))
        ax_s.imshow(img, cmap="viridis", aspect="equal", interpolation="antialiased")
        ax_s.axis("off")
        add_scale_bar(ax_s, sp)
        fig_s.savefig(str(PANEL_DIR / f"figure2_panelE_{organ.lower()}.svg"),
                      bbox_inches="tight", pad_inches=0)
        plt.close(fig_s)
        print(f"  saved panel figure2_panelE_{organ.lower()}.svg")


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_panel(fig, stem):
    """Save individual panel as SVG without titles or padding."""
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    fig.savefig(str(PANEL_DIR / f"{stem}.svg"), bbox_inches="tight", pad_inches=0)
    print(f"  saved panel {stem}.svg")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    probe = load_probe()
    ret   = load_retrieval()
    xplat = load_crossplatform()

    print(f"[Panels B/C] {len(probe)} variants for HMDB super_class")

    TITLE_B = "B   HMDB Classification — Unambiguous Channels\n(macro-F1, linear probe, n_cand=1 only, n=35,484)"
    TITLE_C = "C   HMDB Retrieval — Unambiguous Channels\n(MAP@10, n_cand=1 only, n=35,484)"

    fig_b, ax_b = plt.subplots(figsize=(6.5, 5.5))
    draw_bars(ax_b, probe, "mean_f1", "std_f1", TITLE_B, "Macro-F1", 0.40)
    save_panel(fig_b, "figure2_panelB_f1")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(6.5, 5.5))
    draw_bars(ax_c, ret, "map_mean", "map_std", TITLE_C, "MAP@10", 0.86)
    save_panel(fig_c, "figure2_panelC_map")
    plt.close(fig_c)

    fig_d, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(8.0, 4.5))
    draw_crossplatform(ax_d1, ax_d2, xplat)
    save_panel(fig_d, "figure2_panelD_crossplatform")
    plt.close(fig_d)

    ch_meta = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                          usecols=["sample_path", "Organism_Part"])
    save_spatial_gallery_panels(ch_meta)

    print("[DONE] all outputs →", PANEL_DIR)


if __name__ == "__main__":
    main()
