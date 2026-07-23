"""
plot_figS14.py
--------------
Supplementary Figure S14: Extended Sample Retrieval Gallery.

Extends Figure 7 retrieval examples to more organs and includes failure cases.
For each organ: query image + top-2 nearest neighbours from different acquisitions.
Randomly sampled queries (seed=42) to avoid cherry-picking.

Panels:
  A  Successful retrieval examples + hard cases combined in one figure

Usage:
  conda run -n torch_gpu python plot_figS14.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from plot_utils import set_nature_style, load_best_channel
set_nature_style()

# -- CONFIG -------------------------------------------------------------------
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
RET_DIR   = METABOFM_ROOT / "outputs/crossdataset_retrieval"
DATA_DIR  = MSI_RAW_DIR
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS14_retrieval_gallery_extended"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI       = 300
N_GALLERY = 3    # query + 2 NNs per organ
RNG_SEED  = 42


CAPTION = """\
Supplementary Figure 14 | Extended sample retrieval gallery across organs.

Extended cross-acquisition retrieval gallery across a broader set of organs, including both high-performance and challenging cases, using the same retrieval pipeline as Fig. 7a. Queries are drawn uniformly at random (seed = 42); nearest neighbours are retrieved from different acquisitions to demonstrate cross-acquisition generalisation. Each row shows a query ion image (purple border) and its two nearest neighbours from different acquisitions. The upper section shows the five organs with the highest leave-one-acquisition-out Recall@1; the lower section (below the divider) shows the three organs with the lowest Recall@1. Organs with lower performance tend to have fewer training examples or greater intra-organ metabolic heterogeneity, leading to less consistent nearest-neighbour retrieval.
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


def load_data():
    emb = np.load(str(EMB_DIR / "stage2_sample_cls.npy")).astype(np.float32)
    emb = normalize(emb, norm="l2")

    ch  = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                      usecols=["sample_path", "Organism_Part", "dataset_id"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)

    sm = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv").merge(
             samp, on="sample_path", how="left")
    assert len(sm) == len(emb)

    def _norm(s):
        return {"Kideny": "Kidney", "colon": "Colon",
                "gratric cancer tissue": "gastric cancer tissue"}.get(str(s).strip(), str(s).strip())
    sm["organ"] = sm["Organism_Part"].apply(_norm)
    return sm, emb


def draw_gallery(fig, axes_grid, sm, emb, organs, rng, title_prefix=""):
    """
    For each organ in `organs`, pick a random query from one dataset,
    retrieve top-2 NNs from different datasets, display images in a row.
    axes_grid: shape (len(organs), N_GALLERY)
    """
    for row_i, organ in enumerate(organs):
        mask  = sm["organ"] == organ
        pool  = np.where(mask)[0]
        if len(pool) < N_GALLERY:
            for col_i in range(N_GALLERY):
                axes_grid[row_i, col_i].axis("off")
                axes_grid[row_i, col_i].set_facecolor("#eeeeee")
            axes_grid[row_i, 0].set_ylabel(organ, fontsize=8, fontweight="bold",
                                            rotation=0, ha="right", va="center",
                                            labelpad=6)
            continue

        # pick random query
        q_idx = int(rng.choice(pool))
        q_ds  = sm.loc[q_idx, "dataset_id"]
        q_emb = emb[q_idx]

        # sims to all other samples of the same organ from different datasets
        sims  = emb[pool] @ q_emb
        order = np.argsort(sims)[::-1]
        # exclude same dataset as query, and the query itself
        nn_indices = []
        for pos in order:
            global_idx = pool[pos]
            if global_idx == q_idx:
                continue
            if sm.loc[global_idx, "dataset_id"] != q_ds:
                nn_indices.append(global_idx)
            if len(nn_indices) >= N_GALLERY - 1:
                break

        chosen = [q_idx] + nn_indices

        for col_i in range(N_GALLERY):
            ax = axes_grid[row_i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            if col_i < len(chosen):
                sample_path = sm.loc[chosen[col_i], "sample_path"]
                full_path   = DATA_DIR / sample_path if not Path(sample_path).is_absolute() else Path(sample_path)
                img = load_best_channel(str(full_path))
                if img is not None:
                    ax.imshow(img, cmap="viridis", aspect="equal",
                              interpolation="antialiased")
                else:
                    ax.set_facecolor("#e8e8e8")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="#888")

                if row_i == 0:
                    label = "Query" if col_i == 0 else f"NN {col_i}"
                    ax.set_title(label, fontsize=8, pad=2)

                # Mark query with border
                if col_i == 0:
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_color("#9970ab")
                        sp.set_linewidth(2)
            else:
                ax.set_visible(False)

        axes_grid[row_i, 0].set_ylabel(organ, fontsize=8, fontweight="bold",
                                        rotation=0, ha="right", va="center",
                                        labelpad=6)


def main():
    import matplotlib.gridspec as gridspec

    sm, emb = load_data()
    rng = np.random.default_rng(RNG_SEED)

    per_organ = pd.read_csv(RET_DIR / "crossdataset_retrieval_per_organ.csv")
    r1_col = [c for c in per_organ.columns if "recall@1" in c.lower() or "recall_1" in c.lower()]
    r1_col = r1_col[0] if r1_col else per_organ.columns[1]
    per_organ = per_organ.sort_values(r1_col, ascending=False)

    avail = set(sm["organ"].unique())
    top_organs  = [o for o in per_organ["organ"].head(10).tolist() if o in avail][:5]
    hard_organs = [o for o in per_organ["organ"].tail(10).tolist() if o in avail][:3]

    # Combined panel: top organs + divider + hard cases in one figure
    _cell  = 2.2
    n_top  = len(top_organs)
    n_hard = len(hard_organs)
    n_rows = n_top + 1 + n_hard   # +1 thin divider row

    fig = plt.figure(figsize=(N_GALLERY * _cell, n_rows * _cell))
    heights = [1.0] * n_top + [0.18] + [1.0] * n_hard
    gs = gridspec.GridSpec(n_rows, N_GALLERY, figure=fig,
                           height_ratios=heights, hspace=0.05, wspace=0.02)

    axes_top = np.array([[fig.add_subplot(gs[r, c]) for c in range(N_GALLERY)]
                         for r in range(n_top)])
    draw_gallery(fig, axes_top, sm, emb, top_organs, rng)

    ax_div = fig.add_subplot(gs[n_top, :])
    ax_div.axis("off")
    ax_div.axhline(0.5, color="#aaaaaa", lw=0.8)
    ax_div.text(0.5, 0.5, "Challenging cases (lowest R@1)",
                ha="center", va="center", fontsize=9,
                color="#555555", transform=ax_div.transAxes)

    axes_hard = np.array([[fig.add_subplot(gs[n_top + 1 + r, c]) for c in range(N_GALLERY)]
                          for r in range(n_hard)])
    draw_gallery(fig, axes_hard, sm, emb, hard_organs, rng)

    save_panel(fig, "figS14_panelA_retrieval_gallery_combined")
    plt.close(fig)

    for old in ["figS14_panelA_retrieval_gallery_top",
                "figS14_panelB_retrieval_gallery_hard"]:
        p = PANEL_DIR / f"{old}.svg"
        if p.exists():
            p.unlink()

    write_caption()
    print("FigS11 done.")


if __name__ == "__main__":
    main()

