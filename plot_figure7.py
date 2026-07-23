"""
plot_figure7.py
---------------
Figure 7: Cross-study Sample Retrieval Examples.

Demonstrates that Stage 2 embeddings enable biologically meaningful
retrieval across 5,600 independent MSI studies.

Panels:
  A  Sample embedding space (UMAP) with three query samples highlighted
     and their top-10 retrieved neighbours circled.
  B  Ranked neighbour list for each query (organ, organism, cosine sim).
  C  Neighbour organ purity (top-10) â€” Stage 2 vs Stage 1 â€” for
     the same three queries.

Usage:
  conda run -n torch_gpu python code_v2/plot_figure7.py
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
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from plot_utils import set_nature_style, load_best_channel, draw_pipeline_diagram
set_nature_style()

# â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
UMAP_DIR = METABOFM_ROOT / "outputs/sample_umap"
CD_DIR   = METABOFM_ROOT / "outputs/crossdataset_retrieval"
DATA_DIR = MSI_RAW_DIR
OUT_DIR  = METABOFM_ROOT / "outputs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR   = OUT_DIR / "figure7"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI      = 300
TOP_K    = 25   # neighbours for purity computation
TABLE_K  = 7    # rows shown in Panel B table (fits axes without overflow)

# Three representative queries: (organ, organism) with good coverage
# Picked from top organs by Stage 2 R@1 (Kidney=0.92, Brain=0.79, Lung=0.66)
QUERIES = [
    ("Kidney", "Homo sapiens"),
    ("Brain",  None),           # all organisms pooled â€” more samples, better retrieval
    ("Lung",   "Homo sapiens"),
]

# Colour per query for all panels
QUERY_COLORS = ["#2166ac", "#d6604d", "#4dac26"]
S1_COLOR     = "#d6604d"
S2_COLOR     = "#2166ac"

# Background UMAP: color by top organs, desaturated so queries stand out
BG_TOP_N  = 10
BG_PALETTE = [
    "#a6c8e8", "#f4a99a", "#a8d98a", "#fdd9a6", "#cdb8de",
    "#8ec4a0", "#f5c97a", "#b8d9ed", "#e8848a", "#8dcac4",
]
BG_OTHER  = "#d8d8d8"


# â”€â”€ ION IMAGE HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€



def draw_panel_e_retrieval(fig, outer_gs, sm, query_idxs, nn_s2_list, sims_s2_list):
    """1 title row + 3 query rows x 3 cols (query + top-2 retrieved)."""
    N_SHOW = 3
    nq = len(QUERIES)
    gs = outer_gs.subgridspec(nq + 1, N_SHOW,
                               height_ratios=[0.18] + [1.0] * nq,
                               hspace=0.25, wspace=0.06)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.set_title(
        "E   Sample-level Retrieval Examples\n"
        "(query left; green border = correct tissue match, red = mismatch)",
        fontsize=12, fontweight="bold", pad=6, color="#111")

    for qi_pos, ((organ, organism), qcol, qi, nns, sims) in enumerate(
            zip(QUERIES, QUERY_COLORS, query_idxs, nn_s2_list, sims_s2_list)):
        show_idxs  = [qi] + list(nns[:N_SHOW - 1])
        is_q_flags = [True] + [False] * (N_SHOW - 1)
        q_slug = organ.lower().replace(" ", "_")
        for col_i, (idx, is_q) in enumerate(zip(show_idxs, is_q_flags)):
            ax  = fig.add_subplot(gs[qi_pos + 1, col_i])
            img = load_best_channel(sm.loc[idx, "sample_path"])
            if img is not None:
                ax.imshow(img, cmap="viridis", aspect="equal", interpolation="antialiased")
                # save individual sample image
                role = "query" if is_q else f"nn{col_i}"
                idx_organ = sm.loc[idx, "organ"]
                _f, _a = plt.subplots(figsize=(3, 3 * img.shape[0]/img.shape[1]))
                _a.imshow(img, cmap="viridis", aspect="equal", interpolation="antialiased")
                _a.axis("off")
                _stem = f"figure7_panelE_{q_slug}_{role}_{idx_organ.lower().replace(' ','_')}"
                _f.savefig(str(PANEL_DIR / f"{_stem}.svg"), bbox_inches="tight", pad_inches=0)
                plt.close(_f)
                print(f"  saved panel {_stem}.svg")
            else:
                ax.set_facecolor("#e8e8e8")
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="#aaa")
            ax.set_xticks([]); ax.set_yticks([])
            idx_organ = sm.loc[idx, "organ"]
            idx_org   = sm.loc[idx, "organism"]
            org_s = ("H.sap." if idx_org == "Homo sapiens" else
                     "M.mus." if idx_org == "Mus musculus" else str(idx_org)[:7])
            match = (idx_organ == organ)
            if is_q:
                bcol = qcol
                ttl  = f"Query: {organ}\n({org_s})"
            else:
                bcol = "#2ca02c" if match else "#d62728"
                sim  = float(sims[col_i - 1])
                lbl  = "match" if match else "X"
                ttl  = f"[{lbl}] {idx_organ}\n({org_s})  {sim:.2f}"
            for side in ("top", "bottom", "left", "right"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(bcol)
                ax.spines[side].set_linewidth(2.5)
            ax.set_title(ttl, fontsize=7.5, color=bcol, pad=3, fontweight="bold")

def load_data():
    emb_s2 = np.load(str(EMB_DIR / "stage2_sample_cls.npy")).astype(np.float32)
    coords = np.load(str(UMAP_DIR / "umap2d_stage2.npy")).astype(np.float32)

    ch = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                     usecols=["sample_path", "Organism_Part", "organism", "dataset_id"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)

    sm = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv")
    sm = sm.merge(samp, on="sample_path", how="left")
    assert len(sm) == len(emb_s2)

    sm["organ"] = sm["Organism_Part"].replace({"Kideny": "Kidney", "colon": "Colon"})
    sm["umap_x"] = coords[:, 0]
    sm["umap_y"] = coords[:, 1]
    return sm, emb_s2


def load_stage1(sm):
    """Mean-pool ResNet CLS channel embeddings to sample level."""
    ch_emb  = np.load(str(EMB_DIR / "resnet_cls_embeddings.npy"),
                      mmap_mode="r").astype(np.float32)
    ch_meta = pd.read_csv(EMB_DIR / "resnet_cls_meta.csv", usecols=["sample_path"])
    sp_to_si = {sp: i for i, sp in enumerate(sm["sample_path"])}
    emb_s1 = np.zeros((len(sm), ch_emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(sm), dtype=np.int32)
    for row, sp in enumerate(ch_meta["sample_path"]):
        si = sp_to_si.get(sp)
        if si is not None:
            emb_s1[si] += ch_emb[row]
            counts[si] += 1
    valid = counts > 0
    emb_s1[valid] /= counts[valid, None]
    return emb_s1


def pick_query(sm, emb_normed, organ, organism=None):
    """
    Pick the sample closest to the organ centroid in embedding space.
    This is the most representative (median) sample â€” neither cherry-picked
    for high purity nor randomly placed at the periphery of the organ cluster.
    """
    if organism is not None:
        mask = (sm["organ"] == organ) & (sm["organism"] == organism)
    else:
        mask = sm["organ"] == organ
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No samples for {organ} / {organism}")
    centroid = emb_normed[idx].mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sims_to_ctr = emb_normed[idx] @ centroid
    chosen = idx[np.argmax(sims_to_ctr)]
    print(f"    {organ}: idx={chosen} (centroid-closest, n={len(idx)})")
    return int(chosen)


def retrieve(q_idx, emb_normed, sm, k):
    """Return indices of top-k neighbours, excluding same dataset."""
    q_ds  = sm.loc[q_idx, "dataset_id"]
    sims  = emb_normed[q_idx] @ emb_normed.T
    sims[sm["dataset_id"] == q_ds] = -np.inf
    sims[q_idx] = -np.inf
    nn    = np.argsort(-sims)[:k]
    return nn, sims[nn]


# â”€â”€ PANEL A: UMAP with queries + neighbours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_panel_rk(ax):
    """Panel C: Cross-dataset retrieval R@k (Stage 2 vs Stage 1 vs random)."""
    df = pd.read_csv(CD_DIR / "crossdataset_retrieval_overall.csv")
    ks = [1, 5, 10, 20]
    variants = [
        ("Stage 2",  S2_COLOR, "stage2_ch_refined"),
        ("Stage 1",  S1_COLOR, "stage1"),
    ]

    df_s2 = df[df["variant"] == "Stage 2"].iloc[0]
    df_s1 = df[df["variant"] == "Stage 1"].iloc[0]
    random_val = df_s2["macro_random@1"]  # random baseline is same across k

    x = np.arange(len(ks))
    bw = 0.30
    for i, (label, color, _) in enumerate(variants):
        row = df_s2 if label == "Stage 2" else df_s1
        vals = [float(row[f"weighted_recall@{k}"]) for k in ks]
        ax.bar(x + (i - 0.5) * bw, vals, width=bw, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.4, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 0.5) * bw, v + 0.008, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=color)

    # random reference line
    ax.axhline(random_val, color="#aaa", lw=1.2, ls="--",
               label=f"Random (R@1={random_val:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"R@{k}" for k in ks], fontsize=10)
    ax.set_ylabel("Weighted Recall@k\n(cross-dataset, 5,600 samples)", fontsize=11)
    ax.set_title("C   Cross-study Embedding Transfer\n"
                 "(leave-one-study-out; organ used as ground-truth proxy, 31 classes)",
                 fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower right")


def draw_panel_a(ax, sm, query_idxs, nn_idxs_list):
    pct = 1
    x, y = sm["umap_x"].values, sm["umap_y"].values

    # build organ colour map (desaturated) for background structure
    top_organs = sm["organ"].value_counts().head(BG_TOP_N).index.tolist()
    bg_cmap    = {o: BG_PALETTE[i] for i, o in enumerate(top_organs)}

    # which points are highlighted (query + neighbours)
    highlighted = set()
    for qi, nns in zip(query_idxs, nn_idxs_list):
        highlighted.add(qi)
        highlighted.update(nns.tolist())
    bg_mask = np.ones(len(sm), dtype=bool)
    bg_mask[list(highlighted)] = False

    # draw background organ by organ so structure is visible
    bg_idx = np.where(bg_mask)[0]
    # "other" organs first (bottom layer)
    other_mask = bg_mask & ~sm["organ"].isin(top_organs).values
    if other_mask.any():
        ax.scatter(x[other_mask], y[other_mask], s=6, c=BG_OTHER,
                   alpha=0.60, linewidths=0, rasterized=True)
    for organ in reversed(top_organs):
        omask = bg_mask & (sm["organ"] == organ).values
        if omask.any():
            ax.scatter(x[omask], y[omask], s=8, c=bg_cmap[organ],
                       alpha=0.80, linewidths=0, rasterized=True)

    # neighbours (bold, full-saturation colour)
    for (organ, organism), color, nns in zip(QUERIES, QUERY_COLORS, nn_idxs_list):
        ax.scatter(x[nns], y[nns], s=45, c=color, alpha=0.90,
                   linewidths=0, rasterized=True, zorder=4)

    # query points (on top)
    for (organ, organism), color, qi in zip(QUERIES, QUERY_COLORS, query_idxs):
        ax.scatter(x[qi], y[qi], s=180, c=color, marker="*",
                   linewidths=0.8, edgecolors="white", zorder=5)

    ax.set_xlim(np.percentile(x, pct), np.percentile(x, 100 - pct))
    ax.set_ylim(np.percentile(y, pct), np.percentile(y, 100 - pct))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_title(f"B   Embedding Space with Query Samples\n"
                 f"and Their Top-{TOP_K} Retrieved Neighbours",
                 fontsize=10, fontweight="bold", pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend: query colours + star marker
    handles = []
    for (organ, organism), color in zip(QUERIES, QUERY_COLORS):
        org_short = ("H. sapiens" if organism == "Homo sapiens" else
                     "M. musculus" if organism == "Mus musculus" else
                     "all organisms" if organism is None else organism)
        handles.append(mpatches.Patch(color=color, label=f"{organ} ({org_short})"))
    handles.append(Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="#555", markersize=9,
                          label="Query sample"))
    ax.legend(handles=handles, fontsize=7, frameon=True, framealpha=0.9,
              edgecolor="#ccc", loc="upper left", handlelength=1.2)


# â”€â”€ PANEL B: ranked neighbour table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_panel_b(ax, sm, query_idxs, nn_idxs_list, sims_list):
    ax.axis("off")
    ax.set_title(f"C   Top-{TABLE_K} Retrieved Neighbours per Query\n"
                 "(cross-study; each row is an independent MSI dataset)",
                 fontsize=10, fontweight="bold", pad=5, loc="left")

    # columns: # | Organ | Organism | Sim.
    col_headers = ["#", "Organ", "Organism", "Sim."]
    col_w       = [0.08, 0.32, 0.36, 0.18]   # fractions of panel_w, sum â‰¤ 1

    n_queries = len(QUERIES)
    panel_w   = 1.0 / n_queries
    header_h  = 0.14
    row_h     = (0.82 - header_h) / TABLE_K
    y_top     = 0.96

    for qi_pos, ((organ, organism), color, qi, nns, sims) in enumerate(
            zip(QUERIES, QUERY_COLORS, query_idxs, nn_idxs_list, sims_list)):
        x0 = qi_pos * panel_w + 0.01

        # sub-title (query label)
        org_short = ("H. sapiens" if organism == "Homo sapiens" else
                     "M. musculus" if organism == "Mus musculus" else
                     "all organisms" if organism is None else organism)
        ax.text(x0 + (panel_w - 0.02) / 2, y_top,
                f"Query: {organ}  ({org_short})",
                transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                color=color, ha="center", va="top")

        # column headers with divider line
        y_hdr = y_top - 0.07
        cx = x0
        for hdr, cw in zip(col_headers, col_w):
            ax.text(cx, y_hdr, hdr,
                    transform=ax.transAxes, fontsize=6.8, fontweight="bold",
                    color="#333", ha="left", va="top")
            cx += cw * panel_w
        # underline
        ax.plot([x0, x0 + panel_w - 0.02], [y_hdr - 0.04, y_hdr - 0.04],
                transform=ax.transAxes, color="#aaa", lw=0.8)

        # rows
        for rank, (ni, sim) in enumerate(zip(nns[:TABLE_K], sims[:TABLE_K])):
            y_row = y_top - 0.07 - header_h - rank * row_h
            nn_organ   = sm.loc[ni, "organ"]
            nn_org     = sm.loc[ni, "organism"]
            nn_org_s   = ("H. sap."  if nn_org == "Homo sapiens"  else
                          "M. mus."  if nn_org == "Mus musculus"   else nn_org[:10])
            match      = (nn_organ == organ)
            bg_clr     = "#eef5fb" if (rank % 2 == 0) else "white"
            ax.add_patch(plt.Rectangle(
                (x0, y_row - row_h * 0.80), panel_w - 0.02, row_h * 0.85,
                transform=ax.transAxes, facecolor=bg_clr, linewidth=0))

            row_vals = [str(rank + 1), nn_organ[:18], nn_org_s, f"{sim:.3f}"]
            cx = x0
            for vi, (val, cw) in enumerate(zip(row_vals, col_w)):
                fc = color if (match and vi == 1) else "#222"
                fw = "bold" if (match and vi == 1) else "normal"
                ax.text(cx, y_row - 0.01, val,
                        transform=ax.transAxes, fontsize=6.5,
                        color=fc, fontweight=fw, ha="left", va="top")
                cx += cw * panel_w

        # vertical separator between query columns
        if qi_pos < n_queries - 1:
            ax.plot([(qi_pos + 1) * panel_w] * 2, [0.05, 0.97],
                    transform=ax.transAxes, color="#ddd", lw=0.8)


# â”€â”€ PANEL C: organ purity Stage 1 vs Stage 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_panel_c(ax, sm, query_idxs, nn_s2_list, nn_s1_list):
    labels = []
    purity_s2 = []
    purity_s1 = []

    for (organ, organism), qi, nns2, nns1 in zip(
            QUERIES, query_idxs, nn_s2_list, nn_s1_list):
        org_short = ("H. sap." if organism == "Homo sapiens" else
                     "M. mus." if organism == "Mus musculus" else
                     "all org." if organism is None else organism[:8])
        labels.append(f"{organ}\n({org_short})")
        purity_s2.append((sm.loc[nns2, "organ"] == organ).mean())
        purity_s1.append((sm.loc[nns1, "organ"] == organ).mean())

    x   = np.arange(len(labels))
    bw  = 0.32
    ax.bar(x - bw / 2, purity_s2, width=bw, color=S2_COLOR,
           alpha=0.85, label="Stage 2 (MetaboFM)", zorder=3)
    ax.bar(x + bw / 2, purity_s1, width=bw, color=S1_COLOR,
           alpha=0.85, label="Stage 1 (ResNet)", zorder=3)

    for xi, (p2, p1) in enumerate(zip(purity_s2, purity_s1)):
        ax.text(xi - bw / 2, p2 + 0.02, f"{p2:.2f}", ha="center",
                va="bottom", fontsize=8, color=S2_COLOR, fontweight="bold")
        ax.text(xi + bw / 2, p1 + 0.02, f"{p1:.2f}", ha="center",
                va="bottom", fontsize=8, color=S1_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, ha="center")
    ax.set_ylabel(f"Organ purity (top-{TOP_K})", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"D   Neighbour Organ Purity (Top-{TOP_K})\nStage 2 vs Stage 1",
                 fontsize=10, fontweight="bold", pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(1.0, color="#aaa", lw=0.8, ls="--")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9,
              edgecolor="#ccc", loc="lower right")


# â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_fig(fig, stem):
    for ext in ("svg", "png"):
        fig.savefig(str(OUT_DIR / f"{stem}.{ext}"), dpi=DPI, bbox_inches="tight")
def save_panel(fig, stem):
    """Save individual panel as SVG without titles or padding."""
    for ax in fig.get_axes():
        ax.set_title("", loc="center")
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")
    fig.suptitle("")
    fig.savefig(str(PANEL_DIR / f"{stem}.svg"), bbox_inches="tight", pad_inches=0)
    print(f"  saved panel {stem}.svg")

    print(f"  saved {stem}")


def main():
    print("[LOAD] embeddings â€¦")
    sm, emb_s2 = load_data()
    emb_s1     = load_stage1(sm)
    print(f"  {len(sm)} samples")

    s2_normed = normalize(emb_s2, norm="l2")
    s1_normed = normalize(emb_s1, norm="l2")

    print("[QUERY] picking centroid-closest representative sample per organ â€¦")
    query_idxs = [pick_query(sm, s2_normed, organ, organism)
                  for organ, organism in QUERIES]

    print("[RETRIEVE] top-10 neighbours â€¦")
    nn_s2_list, sims_s2_list = [], []
    nn_s1_list, _            = [], []
    for qi in query_idxs:
        nns2, sim2 = retrieve(qi, s2_normed, sm, TOP_K)
        nns1, _    = retrieve(qi, s1_normed, sm, TOP_K)
        nn_s2_list.append(nns2)
        sims_s2_list.append(sim2)
        nn_s1_list.append(nns1)

    # â”€â”€ Panel E: retrieval image examples (saves its own per-query sub-panel
    # SVGs as a side effect) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig_e = plt.figure(figsize=(9, 12))
    gs_e  = fig_e.add_gridspec(1, 1)
    draw_panel_e_retrieval(fig_e, gs_e[0, 0], sm, query_idxs, nn_s2_list, sims_s2_list)
    save_panel(fig_e, "figure7_panelE_retrieval_gallery")
    plt.close(fig_e)

    # â”€â”€ individual panels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig_a, ax_a = plt.subplots(figsize=(7, 6))
    draw_panel_a(ax_a, sm, query_idxs, nn_s2_list)
    save_panel(fig_a, "figure7_panelB_umap")
    plt.close(fig_a)

    fig_rk, ax_rk = plt.subplots(figsize=(7, 4))
    draw_panel_rk(ax_rk)
    save_panel(fig_rk, "figure7_panelC_recall_k")
    plt.close(fig_rk)

    fig_b, ax_b = plt.subplots(figsize=(10, 7))
    draw_panel_b(ax_b, sm, query_idxs, nn_s2_list, sims_s2_list)
    save_panel(fig_b, "figure7_panelD_neighbours")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(7, 4))
    draw_panel_c(ax_c, sm, query_idxs, nn_s2_list, nn_s1_list)
    save_panel(fig_c, "figure7_panelD_purity")
    plt.close(fig_c)

    print("[DONE] →", PANEL_DIR)


if __name__ == "__main__":
    main()



