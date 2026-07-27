"""
plot_figS3.py
--------------
Supplementary Figure S3: Full-dataset HMDB benchmark (all channels, mixed
annotation confidence). Split out of plot_figure2.py so that main-figure
scripts only save their own panels; supplementary panels live in their own
dedicated script + output folder.

Panels:
  A  Linear probe macro-F1 at the super_class level (all channels)
  B  Retrieval MAP@10 at the super_class level (all channels)

Usage:

  python plot_figS3.py
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

# ── CONFIG ───────────────────────────────────────────────────────────────────
BENCH_DIR = METABOFM_ROOT / "outputs/benchmarks_v2"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS3_full_hmdb_benchmark"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

VARIANT_LABELS_FULL = {
    "stage2_ch_refined[all]": "Stage 2 (channel-refined)",
    "resnet+smiles[all]":     "ResNet + SMILES",
    "resnet_only[all]":       "Stage 1 (channel)",
    "mz_only[all]":           "m/z only",
    "metadata_only[all]":     "Metadata only",
    "smiles_only[all]":       "SMILES only",
    "imagenet_resnet[all]":   "ImageNet ResNet",
}
VARIANT_ORDER_FULL = list(VARIANT_LABELS_FULL.keys())
COLORS_FULL = {
    "stage2_ch_refined[all]": "#2166ac",
    "resnet+smiles[all]":     "#4dac26",
    "resnet_only[all]":       "#74add1",
    "mz_only[all]":           "#d6604d",
    "metadata_only[all]":     "#b2b2b2",
    "smiles_only[all]":       "#f4a582",
    "imagenet_resnet[all]":   "#c2c2c2",
}
HATCH_FULL = {
    "mz_only[all]":     "//",
    "smiles_only[all]": "//",
}


def draw_bars_full(ax, summary_df, val_col, err_col, title, xlabel, xlim):
    y = np.arange(len(VARIANT_ORDER_FULL))
    for i, v in enumerate(VARIANT_ORDER_FULL):
        if v not in summary_df.index:
            continue
        val = float(summary_df.loc[v, val_col])
        err = float(summary_df.loc[v, err_col]) if err_col in summary_df.columns else 0.0
        color = COLORS_FULL.get(v, "#888")
        hatch = HATCH_FULL.get(v, None)
        ax.barh(i, val, xerr=err, height=0.62,
                color=color, hatch=hatch,
                edgecolor="white" if hatch is None else color,
                linewidth=0.4,
                error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#444"))
        ax.text(min(val + max(err, 0) + 0.005, xlim - 0.01), i,
                f"{val:.3f}", va="center", ha="left", fontsize=9, color="#222")
    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_LABELS_FULL[v] for v in VARIANT_ORDER_FULL], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0, xlim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)


def main():
    probe_full = pd.read_csv(BENCH_DIR / "linear_probe" / "summary.csv")
    probe_full = probe_full[probe_full["field"] == "super_class"].set_index("variant")
    ret_full   = pd.read_csv(BENCH_DIR / "retrieval" / "summary.csv")
    ret_full   = ret_full[ret_full["field"] == "super_class"].set_index("variant")

    fig_sf, ax_sf = plt.subplots(figsize=(6.5, 5.5))
    draw_bars_full(ax_sf, probe_full, "mean_f1", "std_f1",
                   "HMDB Classification — Full Dataset\n(macro-F1, linear probe, all channels)",
                   "Macro-F1", 0.38)
    fig_sf.savefig(str(PANEL_DIR / "figS3_panelA_lp_superclass_full.svg"),
                   bbox_inches="tight", pad_inches=0)
    plt.close(fig_sf)
    print("  saved figS3_panelA_lp_superclass_full")

    fig_sr, ax_sr = plt.subplots(figsize=(6.5, 5.5))
    draw_bars_full(ax_sr, ret_full, "map_mean", "map_std",
                   "HMDB Retrieval — Full Dataset\n(MAP@10, all channels)",
                   "MAP@10", 0.95)
    fig_sr.savefig(str(PANEL_DIR / "figS3_panelB_ret_superclass_full.svg"),
                   bbox_inches="tight", pad_inches=0)
    plt.close(fig_sr)
    print("  saved figS3_panelB_ret_superclass_full")

    full_caption = (
        "Supplementary Figure 3 | Full-dataset HMDB benchmark including channels with multiple annotation candidates.\n\n"
        "Classification and retrieval benchmarks on the complete channel set "
        "(n = 158,405). Variants differ in the subsets they evaluate: SMILES-only and ResNet+SMILES "
        "are restricted to channels with a single unambiguous HMDB candidate (n = 35,484), while image-only variants "
        "use all 158,405 channels. Channels with multiple candidates are assigned labels by majority vote, "
        "which introduces annotation noise. The unambiguous benchmark (n_cand = 1 only) is the primary "
        "comparison reported in the main text; this figure is provided for completeness.\n\n"
        "a, Linear probe macro-F1 at the HMDB super-class level across all representation variants.\n\n"
        "b, Retrieval MAP@10 at the HMDB super-class level across all representation variants."
    )
    (PANEL_DIR / "captions.txt").write_text(full_caption, encoding="utf-8")
    print("  saved figS3 captions.txt")
    print("[DONE]", PANEL_DIR)


if __name__ == "__main__":
    main()
