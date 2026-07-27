

"""
plot_figure5.py
---------------
Figure 5: Sample-level Embedding Space.

Panels:
  A  Stage 2 UMAP coloured by organ/tissue (top 10, no Other)
  B  Leave-one-study-out organ retrieval R@1 per organ,
     Stage 2 vs Stage 1 vs random baseline
  C  Same UMAP coloured by organism (Homo sapiens / Mus musculus)

Usage:
  conda run -n torch_gpu python plot_figure5.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import textwrap as _tw

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from plot_utils import set_nature_style, load_best_channel, draw_pipeline_diagram, add_scale_bar, _excluded
set_nature_style()

# â"€â"€ CONFIG â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
UMAP_DIR  = METABOFM_ROOT / "outputs/sample_umap"
RET_DIR   = METABOFM_ROOT / "outputs/crossdataset_retrieval"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
DATA_DIR  = MSI_RAW_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR   = OUT_DIR / "figure5"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI        = 300
TOP_ORGANS = 10
PT_SIZE    = 4
PT_ALPHA   = 0.70
CLIP_PCT   = 1
MIN_ORGAN_N   = 10
MIN_DATASETS  = 2
TOP_RET_ORGANS = 15   # max organs to show in Panel B to avoid label overlap

ORGAN_PALETTE = [
    "#2166ac", "#d6604d", "#4dac26", "#fdae6b", "#9970ab",
    "#1b7837", "#e08214", "#74add1", "#a50026", "#35978f",
]
ORGANISM_PALETTE = {
    "Homo sapiens": "#2166ac",
    "Mus musculus": "#d6604d",
}
OTHER_COLOR = "#d0d0d0"

S2_COLOR  = "#2166ac"
S1_COLOR  = "#d6604d"

QUERIES      = [("Kidney", "Homo sapiens"), ("Brain", None), ("Lung", "Homo sapiens")]
QUERY_COLORS = ["#9970ab", "#2166ac", "#4dac26"]

TITLE_B = "B   Sample Embedding Space\n(UMAP, colored by organ, n=5,600)"
TITLE_C = "C   Sample Embedding Space\n(UMAP, colored by organism)"
TITLE_D = "D   Cross-study Organ Retrieval (R@1)\n(leave-one-study-out, 5,600 independent MSI datasets)"


# â"€â"€ ION IMAGE LOADING â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€



# â"€â"€ PANEL E: retrieval image examples â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

TITLE_E = "E   Intra- vs Inter-organ Cosine Distance`n(50 k random sample pairs)"

def draw_panel_e(ax, sm, emb_normed):
    """Violin plot of cosine distances for within-organ vs across-organ pairs."""
    rng = np.random.default_rng(42)
    N_PAIRS = 50_000
    n = len(sm)
    organs = sm["organ"].values

    i_a = rng.integers(0, n, N_PAIRS * 4)
    i_b = rng.integers(0, n, N_PAIRS * 4)
    valid = i_a != i_b
    i_a, i_b = i_a[valid], i_b[valid]

    sims  = np.einsum("ij,ij->i", emb_normed[i_a], emb_normed[i_b])
    dists = 1.0 - sims
    same  = organs[i_a] == organs[i_b]

    intra = dists[same][:N_PAIRS]
    inter = dists[~same][:N_PAIRS]

    vp = ax.violinplot([intra, inter], positions=[0, 1],
                       showmedians=True, showextrema=False, widths=0.6)
    vp["bodies"][0].set_facecolor(S2_COLOR); vp["bodies"][0].set_alpha(0.55)
    vp["bodies"][1].set_facecolor(S1_COLOR); vp["bodies"][1].set_alpha(0.55)
    vp["cmedians"].set_color("#111"); vp["cmedians"].set_linewidth(1.8)

    for data, pos, col in [(intra, 0, S2_COLOR), (inter, 1, S1_COLOR)]:
        q1, q3 = np.percentile(data, [25, 75])
        med    = np.median(data)
        ax.plot([pos - 0.08, pos + 0.08], [q1, q1], color=col, lw=1.2)
        ax.plot([pos - 0.08, pos + 0.08], [q3, q3], color=col, lw=1.2)
        ax.plot([pos, pos], [q1, q3], color=col, lw=1.0)
        ax.text(pos, q3 + 0.015, f"med={med:.3f}", ha="center",
                fontsize=7.5, color=col, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Intra-organ\n(same tissue)", "Inter-organ\n(different tissue)"],
                       fontsize=9)
    ax.set_ylabel("Cosine distance  (1 - similarity)", fontsize=9)
    ax.set_title(TITLE_E, fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

FIG4_STEPS = [
    {"label": "MSI Stack\n(C x H x W)",    "sub": "C ion channels",             "kind": "data",   "icon": "msi",         "pos": (0, 0)},
    {"label": "Stage 1\nResNet-18 BT",     "sub": "per-channel patch encoder",  "kind": "model",  "icon": "resnet",      "pos": (0, 1)},
    {"label": "C x 512-d\nChannel Tokens", "sub": "one token per channel",      "kind": "output", "icon": "embedding",   "pos": (0, 2)},
    {"label": "ChannelAgg.\nTransformer",  "sub": "cross-channel attention",    "kind": "model",  "icon": "transformer", "pos": (0, 3)},
    {"label": "CLS Token\n512-dim",        "sub": "one vector per sample",      "kind": "output", "icon": "embedding",   "pos": (0, 4)},
    {"label": "UMAP &\nRetrieval",         "sub": "organ / organism clusters",  "kind": "eval",   "icon": "umap",        "pos": (0, 5)},
]


def save_fig(fig, stem):
    for ext in ("svg", "png"):
        fig.savefig(str(OUT_DIR / f"{stem}.{ext}"), dpi=DPI, bbox_inches="tight")
def save_panel(fig, stem):
    """Save individual panel as SVG without titles or padding."""
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    fig.savefig(str(PANEL_DIR / f"{stem}.svg"), bbox_inches="tight", pad_inches=0)
    print(f"  saved panel {stem}.svg")

    print(f"  saved {stem}")


def _clip_umap(ax, df, pct=CLIP_PCT):
    x, y = df["umap_x"].values, df["umap_y"].values
    ax.set_xlim(np.percentile(x, pct), np.percentile(x, 100 - pct))
    ax.set_ylim(np.percentile(y, pct), np.percentile(y, 100 - pct))


def normalize_organ(s):
    return {"Kideny": "Kidney", "colon": "Colon",
            "gratric cancer tissue": "gastric cancer tissue"}.get(str(s), str(s))


# â"€â"€ LOAD DATA â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def load_umap_meta():
    coords = np.load(str(UMAP_DIR / "umap2d_stage2.npy"))
    ch     = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                         usecols=["sample_path", "Organism_Part", "organism"])
    samp   = ch.drop_duplicates("sample_path").reset_index(drop=True)
    sm     = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv").merge(
                 samp, on="sample_path", how="left")
    assert len(sm) == len(coords)
    sm["umap_x"] = coords[:, 0]
    sm["umap_y"] = coords[:, 1]
    sm["organ"]  = sm["Organism_Part"].apply(normalize_organ)
    sm["org"]    = sm["organism"].apply(
        lambda o: o if o in ORGANISM_PALETTE else "Other")
    return sm


def load_retrieval():
    """Load pre-computed per-organ R@1 results from probe script."""
    df = pd.read_csv(RET_DIR / "crossdataset_retrieval_pivot_r1.csv")

    # filter: min samples + min datasets
    df = df[
        (df["n_samples"] >= MIN_ORGAN_N) &
        (df["n_datasets"] >= MIN_DATASETS)
    ].copy()

    # keep top N by Stage 2 R@1 to avoid label overlap
    df = df.nlargest(TOP_RET_ORGANS, "Stage 2")
    df = df.sort_values("Stage 2", ascending=True).reset_index(drop=True)
    return df


# â"€â"€ PANEL A: UMAP by organ â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def draw_panel_a(ax, df):
    top_organs = df["organ"].value_counts().head(TOP_ORGANS).index.tolist()
    cmap       = {o: ORGAN_PALETTE[i] for i, o in enumerate(top_organs)}

    for organ in top_organs:
        mask = df["organ"] == organ
        ax.scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                   s=PT_SIZE, c=cmap[organ], alpha=PT_ALPHA,
                   linewidths=0, rasterized=True)

    _clip_umap(ax, df)
    ax.set_title(TITLE_B, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=cmap[o],
                              label=f"{o} ({(df['organ']==o).sum()})")
               for o in top_organs]
    ax.legend(handles=handles, fontsize=6.5, frameon=True, framealpha=0.9,
              edgecolor="#ccc", loc="upper left", handlelength=1.2)


# â"€â"€ PANEL B: retrieval bar chart â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def draw_panel_b(ax, ret):
    y  = np.arange(len(ret))
    bh = 0.28

    ax.barh(y + bh/2, ret["Stage 2"], height=bh, color=S2_COLOR,
            alpha=0.85, label="Stage 2 (MetaboFM)", zorder=3)
    ax.barh(y - bh/2, ret["Stage 1"], height=bh, color=S1_COLOR,
            alpha=0.85, label="Stage 1 (ResNet)", zorder=3)

    # random baseline tick marks
    for i, row in ret.iterrows():
        ax.plot([row["random@1"], row["random@1"]],
                [i - bh - 0.1, i + bh + 0.1],
                color="#444", lw=1.2, zorder=4)

    # y-axis on the RIGHT so labels don't bleed into Panel A
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks(y)
    ax.set_yticklabels(ret["organ"].tolist(), fontsize=8.5)

    ax.set_xlabel("Recall@1", fontsize=10)
    ax.set_title(TITLE_C, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    ax.axvline(0, color="#aaa", lw=0.5)

    # legend inside axes, upper left (bars are shorter there)
    handles = [
        mpatches.Patch(color=S2_COLOR, alpha=0.85, label="Stage 2 (MetaboFM)"),
        mpatches.Patch(color=S1_COLOR, alpha=0.85, label="Stage 1 (ResNet)"),
        Line2D([0], [0], color="#444", lw=1.5, label="Random"),
    ]
    ax.legend(handles=handles, fontsize=7.5, frameon=True, framealpha=0.85,
              edgecolor="#ccc", loc="upper left")

    # macro R@1 annotation at bottom right
    macro_s2 = ret["Stage 2"].mean()
    macro_s1 = ret["Stage 1"].mean()
    ax.text(0.98, 0.02,
            f"Macro R@1\nStage 2={macro_s2:.3f}\nStage 1={macro_s1:.3f}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.85))


# â"€â"€ PANEL C: UMAP by organism â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def draw_panel_c(ax, df):
    for org in reversed(list(ORGANISM_PALETTE)):
        mask = df["org"] == org
        ax.scatter(df.loc[mask, "umap_x"], df.loc[mask, "umap_y"],
                   s=PT_SIZE, c=ORGANISM_PALETTE[org], alpha=PT_ALPHA,
                   linewidths=0, rasterized=True)

    _clip_umap(ax, df)
    ax.set_title(TITLE_D, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=ORGANISM_PALETTE[org],
                              label=f"{org} ({(df['org']==org).sum()})")
               for org in ORGANISM_PALETTE]
    ax.legend(handles=handles, fontsize=7.5, frameon=True, framealpha=0.9,
              edgecolor="#ccc", loc="upper left", handlelength=1.2)



# -- PANEL F: random spatial sample gallery ----------------------------------

N_ORGANS_F = 3
N_COLS_F   = 4

# â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    print("[LOAD] â€¦")
    df  = load_umap_meta()
    ret = load_retrieval()
    print(f"  {len(df)} samples | {len(ret)} organs in retrieval")
    print(f"  Macro R@1 - Stage 2: {ret['Stage 2'].mean():.3f}, "
          f"Stage 1: {ret['Stage 1'].mean():.3f}, "
          f"Random: {ret['random@1'].mean():.3f}")

    # -- load embeddings for Panel E (intra/inter-organ distance violin)
    from sklearn.preprocessing import normalize as _norm
    emb_s2    = np.load(str(EMB_DIR / "stage2_sample_cls.npy")).astype(np.float32)
    s2_normed = _norm(emb_s2, norm="l2")
    sm_emb    = df.reset_index(drop=True)
    # â"€â"€ individual panels â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    fig_a, ax_a = plt.subplots(figsize=(7.0, 5.5))
    draw_panel_a(ax_a, df)
    save_panel(fig_a, "figure5_panelB_umap_organ")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7.0, 5.5))
    draw_panel_c(ax_b, df)
    save_panel(fig_b, "figure5_panelC_umap_organism")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(5.5, 7.0))
    draw_panel_b(ax_c, ret)
    save_panel(fig_c, "figure5_panelD_retrieval")
    plt.close(fig_c)

    fig_e, ax_e = plt.subplots(figsize=(5.0, 5.5))
    draw_panel_e(ax_e, sm_emb, s2_normed)
    save_panel(fig_e, "figure5_panelE_intra_inter_dist")
    plt.close(fig_e)

    top_organs_f = df["organ"].value_counts().head(N_ORGANS_F).index.tolist()
    rng_f = np.random.default_rng(42)
    for organ in top_organs_f:
        pool = df.loc[df["organ"] == organ, "sample_path"].values
        pool = np.array([sp for sp in pool if not _excluded(sp)])
        chosen = rng_f.choice(pool, size=min(N_COLS_F, len(pool)), replace=False)
        _slug = organ.lower().replace(" ", "_")
        for col_i, sample_path in enumerate(chosen):
            img = load_best_channel(sample_path)
            if img is None:
                print(f"  [SKIP] panel F organ={organ} col={col_i + 1}: no image")
                continue
            _stem = f"figure5_panelF_{_slug}_{col_i + 1}"
            _fi, _ai = plt.subplots(figsize=(3, 3))
            _ai.imshow(img, cmap="viridis", aspect="equal", interpolation="antialiased")
            _ai.axis("off")
            add_scale_bar(_ai, sample_path)
            _fi.savefig(str(PANEL_DIR / f"{_stem}.svg"), bbox_inches="tight", pad_inches=0)
            plt.close(_fi)
            print(f"  saved panel {_stem}.svg")

    print("[DONE] outputs ->", PANEL_DIR)


if __name__ == "__main__":
    main()























