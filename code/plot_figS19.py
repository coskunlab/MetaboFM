"""
plot_figS19.py
--------------
Supplementary Figure S19: Extended Sample Retrieval Gallery.

Extends Figure 7e with additional retrieval examples for the same three
organs (Kidney, Brain, Lung), each shown as a query ion image plus its
top-2 nearest neighbours from different acquisitions. Saved as a single
combined composite figure. Randomly sampled queries (seed=42) to avoid
cherry-picking; distinct queries per organ (no repeats).

Panels:
  A  Combined grid: N_EXAMPLES_PER_ORGAN rows per organ x 3 organs,
     query + top-2 nearest-neighbour columns.

Usage:
  conda run -n torch_gpu python plot_figS19.py
"""

from __future__ import annotations
import re
from pathlib import Path
from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import normalize
from plot_utils import set_nature_style, load_best_channel, add_scale_bar, _excluded
set_nature_style()

# -- CONFIG -------------------------------------------------------------------
EMB_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
DATA_DIR  = MSI_RAW_DIR
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS19_retrieval_gallery_extended"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI  = 300
N_GALLERY = 3    # query + 2 NNs per row
ORGANS = ["Kidney", "Brain", "Lung"]   # same organs as Fig. 7e
N_EXAMPLES_PER_ORGAN = 3               # distinct query examples per organ
RNG_SEED  = 42


CAPTION = """\
Supplementary Figure 19 | Extended sample retrieval gallery for Kidney, Brain, and Lung.

Additional cross-acquisition retrieval examples for the same three organs shown in Fig. 7e (Kidney, Brain, Lung), using the same retrieval pipeline as Fig. 7a. For each organ, three distinct queries are drawn uniformly at random (seed = 42, no repeats); nearest neighbours are retrieved from different acquisitions to demonstrate cross-acquisition generalisation. Each row shows a query ion image (purple border) and its two nearest neighbours from different acquisitions.
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
                      usecols=["sample_path", "Organism_Part"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)

    sm = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv").merge(
             samp, on="sample_path", how="left")
    assert len(sm) == len(emb)

    def _norm(s):
        return {"Kideny": "Kidney", "colon": "Colon",
                "gratric cancer tissue": "gastric cancer tissue"}.get(str(s).strip(), str(s).strip())
    sm["organ"] = sm["Organism_Part"].apply(_norm)
    return sm, emb


def _pick_query_and_nns(sm, emb, pool, q_idx, rng):
    q_ds  = sm.loc[q_idx, "dataset_id"]
    q_emb = emb[q_idx]
    sims  = emb[pool] @ q_emb
    order = np.argsort(sims)[::-1]
    nn_indices = []
    for pos in order:
        global_idx = pool[pos]
        if global_idx == q_idx:
            continue
        if sm.loc[global_idx, "dataset_id"] != q_ds:
            nn_indices.append(global_idx)
        if len(nn_indices) >= N_GALLERY - 1:
            break
    return [q_idx] + nn_indices


def draw_gallery(fig, axes_grid, sm, emb, organs, rng, n_examples):
    """
    axes_grid: shape (len(organs) * n_examples, N_GALLERY)
    For each organ, draws n_examples rows, each with a distinct random
    query (no repeats within an organ) plus its top-2 nearest neighbours
    from different acquisitions.
    """
    for organ_i, organ in enumerate(organs):
        mask = (sm["organ"] == organ) & (~sm["sample_path"].apply(_excluded))
        pool = np.where(mask)[0]

        if len(pool) < N_GALLERY:
            for ex_i in range(n_examples):
                row_i = organ_i * n_examples + ex_i
                for col_i in range(N_GALLERY):
                    axes_grid[row_i, col_i].axis("off")
                    axes_grid[row_i, col_i].set_facecolor("#eeeeee")
            continue

        n_queries = min(n_examples, len(pool))
        query_idxs = rng.choice(pool, size=n_queries, replace=False)

        for ex_i in range(n_examples):
            row_i = organ_i * n_examples + ex_i
            if ex_i >= n_queries:
                for col_i in range(N_GALLERY):
                    axes_grid[row_i, col_i].axis("off")
                    axes_grid[row_i, col_i].set_facecolor("#eeeeee")
                continue

            q_idx = int(query_idxs[ex_i])
            chosen = _pick_query_and_nns(sm, emb, pool, q_idx, rng)

            for col_i in range(N_GALLERY):
                ax = axes_grid[row_i, col_i]
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)

                if col_i < len(chosen):
                    sample_path = sm.loc[chosen[col_i], "sample_path"]
                    full_path = DATA_DIR / sample_path if not Path(sample_path).is_absolute() else Path(sample_path)
                    img = load_best_channel(str(full_path))
                    if img is not None:
                        ax.imshow(img, cmap="viridis", aspect="equal", interpolation="antialiased")
                        add_scale_bar(ax, sample_path, fontsize=5)
                    else:
                        ax.set_facecolor("#e8e8e8")
                        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax.transAxes, fontsize=7, color="#888")

                    if organ_i == 0 and ex_i == 0:
                        label = "Query" if col_i == 0 else f"NN {col_i}"
                        ax.set_title(label, fontsize=8, pad=2)

                    if col_i == 0:
                        for sp in ax.spines.values():
                            sp.set_visible(True)
                            sp.set_color("#9970ab")
                            sp.set_linewidth(2)
                else:
                    ax.set_visible(False)

            if ex_i == n_examples // 2:
                axes_grid[row_i, 0].set_ylabel(organ, fontsize=9, fontweight="bold",
                                                rotation=0, ha="right", va="center",
                                                labelpad=8)


def main():
    sm, emb = load_data()
    rng = np.random.default_rng(RNG_SEED)

    n_rows = len(ORGANS) * N_EXAMPLES_PER_ORGAN
    _cell = 2.2
    fig = plt.figure(figsize=(N_GALLERY * _cell, n_rows * _cell))
    gs = gridspec.GridSpec(n_rows, N_GALLERY, figure=fig, hspace=0.08, wspace=0.03)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(N_GALLERY)]
                     for r in range(n_rows)])

    draw_gallery(fig, axes, sm, emb, ORGANS, rng, N_EXAMPLES_PER_ORGAN)

    save_panel(fig, "figS19_panelA_retrieval_gallery_combined")
    plt.close(fig)

    # Remove any stale per-organ/per-role panels from the previous individual-
    # panel version of this script.
    for p in PANEL_DIR.glob("figS19_panelA_top_*.svg"):
        p.unlink()
    for p in PANEL_DIR.glob("figS19_panelA_hard_*.svg"):
        p.unlink()

    write_caption()
    print("FigS19 done.")


if __name__ == "__main__":
    main()
