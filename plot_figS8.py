"""
plot_figS8.py
--------------
Supplementary Figure S8: Organ retrieval from raw pixel content alone is
substantially worse than from a learned representation.

Motivation: the strong organ-level structure of Stage 1/2 embeddings (Fig. 5b,
Fig. 5d) is only evidence of something non-trivial if organ identity is not
already trivially recoverable from raw visual appearance. This figure adds a
raw-pixel, no-learning baseline (computed in compute_rawpixel_baseline.py) to
the leave-one-dataset-out organ retrieval benchmark and shows it sits well
above random but well below the learned models.

Panels:
  a  Pipeline schematic (separate .mmd file, not drawn here)
  b  Overall weighted/macro Recall@1: random / raw pixels / Stage 1 / Stage 2
  c  UMAP of raw-pixel features, coloured by organ
  d  UMAP of Stage 2 embeddings, coloured by organ (same organs/colours as c)
  e  Per-organ raw-pixel Recall@1 (which organs are visually distinguishable
     from gross raw appearance alone, and which are not)

(A per-organ spatial similarity map comparison, Stage 1 vs. raw pixels, was
explored and dropped: the spatial-contiguity metric it relied on does not
favour Stage 1 -- raw pixel intensity trivially forms spatially contiguous
regions since background is exactly zero, and even restricting to foreground-
only patches did not consistently favour Stage 1 either. That analysis was
not included here since it does not support a clean claim in either
direction; a related, better-motivated patch-level organ-classification
analysis is presented separately in plot_figS14.py.)

Usage:
  conda run -n torch_gpu python plot_figS8.py
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
from sklearn.preprocessing import normalize
from plot_utils import set_nature_style
set_nature_style()

# ── CONFIG ──────────────────────────────────────────────────────────────
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
UMAP_DIR  = METABOFM_ROOT / "outputs/sample_umap"
RET_DIR   = METABOFM_ROOT / "outputs/crossdataset_retrieval"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR = OUT_DIR / "figS8_organ_rawpixel_baseline"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
TOP_ORGANS   = 8     # organs shown in UMAP panels (c, d)
MIN_ORGAN_N  = 10    # min samples per organ for panel e
PT_SIZE      = 4
PT_ALPHA     = 0.70
CLIP_PCT     = 1

ORGAN_PALETTE = [
    "#2166ac", "#d6604d", "#4dac26", "#fdae6b", "#9970ab",
    "#1b7837", "#e08214", "#74add1",
]
OTHER_COLOR = "#d0d0d0"

VARIANT_COLORS = {
    "Random":     "#aaaaaa",
    "Raw pixels": "#c1583f",
    "Stage 1":    "#d6604d",
    "Stage 2":    "#2166ac",
}

CAPTION = """\
Supplementary Figure 8 | Organ retrieval from raw pixel content alone is substantially worse than from a learned representation.

a, Schematic comparison of the raw-pixel baseline (mean-intensity projection across channels, downsampled, no trained encoder, no m/z identity) against the Stage 1 / Stage 2 learned pipeline, both evaluated with the identical leave-one-dataset-out cosine-retrieval protocol.

b, Overall Recall@1 under leave-one-dataset-out organ retrieval. Random and raw-pixel baselines are well below Stage 1 and Stage 2, confirming that organ identity is not an obvious property of raw visual appearance and that the learned representation captures substantial additional structure.

c, UMAP of raw-pixel features (same features used in panel b), coloured by organ (top 8 by sample count). Organ classes are poorly separated.

d, UMAP of Stage 2 sample embeddings, same organs and colours as panel c, for direct visual comparison. Organ classes form substantially more coherent clusters than in panel c.

e, Per-organ raw-pixel Recall@1 (organs with >=10 samples). A small number of organs with distinctive gross morphology (e.g., kidney, brain) are partially recoverable from raw appearance alone; most organs are not, consistent with the aggregate result in panel b.
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


# ── PANEL B: overall Recall@1 bar chart ─────────────────────────────────────

def draw_panel_b(ax):
    overall_std = pd.read_csv(RET_DIR / "crossdataset_retrieval_overall.csv")
    overall_raw = pd.read_csv(RET_DIR / "crossdataset_retrieval_rawpixel_overall.csv")

    stage2 = overall_std[overall_std["variant"] == "Stage 2"].iloc[0]
    stage1 = overall_std[overall_std["variant"] == "Stage 1"].iloc[0]
    rawpix = overall_raw.iloc[0]

    labels  = ["Random", "Raw pixels", "Stage 1", "Stage 2"]
    weighted = [rawpix["weighted_random@1"], rawpix["weighted_recall@1"],
                stage1["weighted_recall@1"], stage2["weighted_recall@1"]]
    macro    = [rawpix["macro_random@1"], rawpix["macro_recall@1"],
                stage1["macro_recall@1"], stage2["macro_recall@1"]]

    y = np.arange(len(labels))
    h = 0.35
    colors = [VARIANT_COLORS[l] for l in labels]

    ax.barh(y + h / 2, weighted, height=h, color=colors, alpha=0.95,
            edgecolor="white", linewidth=0.5, label="Weighted")
    ax.barh(y - h / 2, macro, height=h, color=colors, alpha=0.55,
            edgecolor="white", linewidth=0.5, label="Macro")

    for i, (w, m) in enumerate(zip(weighted, macro)):
        ax.text(w + 0.01, i + h / 2, f"{w:.3f}", va="center", fontsize=8, color="#222")
        ax.text(m + 0.01, i - h / 2, f"{m:.3f}", va="center", fontsize=8, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Recall@1 (leave-one-dataset-out organ retrieval)", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    handles = [mpatches.Patch(facecolor="#888", alpha=0.95, label="Weighted"),
               mpatches.Patch(facecolor="#888", alpha=0.55, label="Macro")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")


# ── PANELS C/D: UMAP comparison ─────────────────────────────────────────────

def _load_sample_meta():
    ch = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                     usecols=["sample_path", "Organism_Part"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    sm = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv").merge(
        samp, on="sample_path", how="left")
    fix = {"Kideny": "Kidney", "colon": "Colon",
           "gratric cancer tissue": "gastric cancer tissue"}
    sm["organ"] = sm["Organism_Part"].apply(lambda s: fix.get(str(s), str(s)))
    return sm


def _clip(ax, x, y, pct=CLIP_PCT):
    ax.set_xlim(np.percentile(x, pct), np.percentile(x, 100 - pct))
    ax.set_ylim(np.percentile(y, pct), np.percentile(y, 100 - pct))


def draw_umap_panel(ax, coords, sm, top_organs, cmap, title):
    x, y = coords[:, 0], coords[:, 1]
    organs = sm["organ"].values
    mask_other = ~np.isin(organs, top_organs)
    ax.scatter(x[mask_other], y[mask_other], s=PT_SIZE, c=OTHER_COLOR,
               alpha=0.4, linewidths=0, rasterized=True)
    for organ in top_organs:
        mask = organs == organ
        ax.scatter(x[mask], y[mask], s=PT_SIZE, c=cmap[organ], alpha=PT_ALPHA,
                   linewidths=0, rasterized=True)
    _clip(ax, x, y)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── PANEL E: per-organ raw-pixel Recall@1 ───────────────────────────────────

def draw_panel_e(ax):
    df = pd.read_csv(RET_DIR / "crossdataset_retrieval_rawpixel_per_organ.csv")
    df = df[df["n_samples"] >= MIN_ORGAN_N].sort_values("recall@1", ascending=True)
    df = df.tail(15)   # top 15 by recall@1 among qualifying organs, avoid label overlap

    y = np.arange(len(df))
    ax.barh(y, df["recall@1"], height=0.65, color=VARIANT_COLORS["Raw pixels"],
            alpha=0.85, edgecolor="white", linewidth=0.4, label="Raw pixels R@1")
    ax.scatter(df["random@1"], y, marker="|", s=200, color="#333", zorder=5,
               label="Random baseline (this organ)")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{o} (n={int(n)})" for o, n in zip(df["organ"], df["n_samples"])],
        fontsize=8)
    ax.set_xlabel("Raw-pixel Recall@1", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")


# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    fig_b, ax_b = plt.subplots(figsize=(6.5, 3.5))
    draw_panel_b(ax_b)
    save_panel(fig_b, "figS8_panelB_overall_recall")
    plt.close(fig_b)

    sm = _load_sample_meta()
    top_organs = sm["organ"].value_counts().head(TOP_ORGANS).index.tolist()
    cmap = {o: ORGAN_PALETTE[i] for i, o in enumerate(top_organs)}

    # panel c: UMAP of raw-pixel features (compute fresh)
    import umap as umap_lib
    emb_raw = np.load(str(RET_DIR / "rawpixel_embeddings.npy"))
    meta_raw = pd.read_csv(RET_DIR / "rawpixel_embeddings_meta.csv")
    raw_normed = normalize(emb_raw, norm="l2")
    print("  [UMAP] fitting raw-pixel embedding space …")
    coords_raw = umap_lib.UMAP(n_components=2, random_state=42,
                                n_neighbors=15, min_dist=0.1).fit_transform(raw_normed)
    fig_c, ax_c = plt.subplots(figsize=(6.0, 5.5))
    draw_umap_panel(ax_c, coords_raw, meta_raw, top_organs, cmap,
                     "C   Raw-pixel Feature Space\n(UMAP, coloured by organ)")
    save_panel(fig_c, "figS8_panelC_umap_rawpixel")
    plt.close(fig_c)

    # panel d: Stage 2 UMAP (precomputed), same organs/colours
    coords_s2 = np.load(str(UMAP_DIR / "umap2d_stage2.npy"))
    assert len(coords_s2) == len(sm)
    fig_d, ax_d = plt.subplots(figsize=(6.0, 5.5))
    draw_umap_panel(ax_d, coords_s2, sm, top_organs, cmap,
                     "D   Stage 2 Embedding Space\n(UMAP, coloured by organ)")
    save_panel(fig_d, "figS8_panelD_umap_stage2")
    plt.close(fig_d)

    fig_e, ax_e = plt.subplots(figsize=(6.5, 5.5))
    draw_panel_e(ax_e)
    save_panel(fig_e, "figS8_panelE_perorgan_rawpixel")
    plt.close(fig_e)

    write_caption()
    print("[DONE] outputs ->", PANEL_DIR)


if __name__ == "__main__":
    main()
