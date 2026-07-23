"""
plot_figure3.py
---------------
Figure 3: Molecule Identity Preservation in Embedding Space.

Panels:
  A  Within- vs between-molecule cosine similarity per variant
     (horizontal grouped bars; delta Î" labelled)
  B  Distribution of per-m/z within-molecule similarities
     for Stage 2 vs Stage 1 (violin; shows result is systematic
     across 1,661 m/z groups, not cherry-picked)
  C  Scatter: within-molecule similarity vs number of observations
     per m/z group for Stage 2 (coloured by organ diversity)

Usage:

  python plot_figure3.py
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
from plot_utils import set_nature_style, load_specific_channel, find_channel_for_mz, draw_pipeline_diagram, _load_npz, _is_clean
set_nature_style()

# â"€â"€ CONFIG â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
VAR_DIR  = METABOFM_ROOT / "outputs/molecule_variance"
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/figures"
DATA_DIR = MSI_RAW_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR   = OUT_DIR / "figure3"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

VARIANT_ORDER = [
    "MetaboFM Stage 2",
    "Stage 1 (ResNet)",
    "ResNet + SMILES",
    "SMILES only",
]
COLORS = {
    "MetaboFM Stage 2": "#2166ac",
    "Stage 1 (ResNet)": "#74add1",
    "ResNet + SMILES":  "#4dac26",
    "SMILES only":      "#f4a582",
}
# The strings above are the raw `variant` values in the underlying data files
# and must stay as-is for filtering. Displayed labels use the paper's naming
# convention elsewhere (plain "Stage 1"/"Stage 2", "ResNet+SMILES" with no
# spaces, hyphenated "SMILES-only") rather than these informal analysis-script
# names ("MetaboFM Stage 2", "(ResNet)", "ResNet + SMILES", "SMILES only").
DISPLAY_LABELS = {
    "MetaboFM Stage 2": "Stage 2",
    "Stage 1 (ResNet)": "Stage 1",
    "ResNet + SMILES":  "ResNet+SMILES",
    "SMILES only":      "SMILES-only",
}
WITHIN_ALPHA  = 1.0
BETWEEN_ALPHA = 0.45

TITLE_B = "B   Within- vs Between-molecule Similarity\n(cosine similarity, grouped by m/z)"
TITLE_C = "C   Per-m/z Within-molecule Similarity\n(1,661 m/z groups, >=10 samples each)"

# â"€â"€ HELPERS â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

# â"€â"€ ION IMAGE LOADING â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def load_channel_image(sample_path, channel_idx):
    """Load one channel from .npz, normalise 0â€"1. Returns (H, W) float32 or None."""
    fname = Path(sample_path).name
    p = DATA_DIR / fname
    if not p.exists():
        return None
    d = np.load(str(p))
    img = d["patch"][int(channel_idx)].astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def _clean_rows(rows, n_per):
    """
    Given candidate rows (sample_path, channel_idx) for one organ at one m/z,
    load each channel image and keep only those passing the quality filter
    (non-empty, non-flat, and signal concentrated within the tissue interior
    rather than a boundary-only rim). Returns up to n_per rows centered on the
    MEDIAN spatial std among clean candidates (not the highest-variance ones),
    consistent with the median-sample selection used for representative ion
    images elsewhere in the paper -- this avoids selecting the most visually
    striking clean examples, which would reintroduce a cherry-picking risk
    even though the initial quality filter itself is outcome-blind.
    """
    scored = []
    for _, row in rows.iterrows():
        patch = _load_npz(row["sample_path"])
        if patch is None:
            continue
        ci = int(row["channel_idx"])
        if ci >= patch.shape[0]:
            continue
        img = patch[ci]
        if not _is_clean(img):
            continue
        scored.append((float(img.std()), row))
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    if n <= n_per:
        return scored
    start = (n - n_per) // 2
    return scored[start:start + n_per]


def find_mz_examples(ch_meta, organs, n_per=2):
    """
    Find a shared m/z (rounded to 3 d.p.) with at least n_per quality-filtered,
    non-boundary-only channel images in every organ in `organs`.
    Returns (mz_val, [rows_organ1, rows_organ2, ...]).
    """
    cm = ch_meta.copy()
    cm["mz_r"] = cm["mz"].round(3)
    per_organ = {o: cm[cm["Organism_Part"] == o] for o in organs}
    common = set(per_organ[organs[0]]["mz_r"])
    for o in organs[1:]:
        common &= set(per_organ[o]["mz_r"])
    common = sorted(common)
    # prefer higher m/z (lipids) and those with enough clean samples in ALL organs
    for mz_r in reversed(common):
        candidate_rows = []
        ok = True
        for o in organs:
            r = per_organ[o][per_organ[o]["mz_r"] == mz_r]
            r = r[[(DATA_DIR / Path(sp).name).exists() for sp in r["sample_path"]]]
            if len(r) < n_per:
                ok = False
                break
            clean = _clean_rows(r, n_per)
            if len(clean) < n_per:
                ok = False
                break
            candidate_rows.append(pd.DataFrame([row for _, row in clean]))
        if ok:
            return float(mz_r), candidate_rows
    return None, None


# â"€â"€ PANEL D: same m/z ion images across two tissue types â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

ORGANS_D = ["Liver", "Brain", "Kidney"]
COLORS_D = ["#e08214", "#2166ac", "#4dac26"]


N_PER_ORGAN_D = 3

def draw_panel_d(ax_parent, ch_meta):
    """
    Title row + N-organ grid: N_PER_ORGAN_D samples per organ, same m/z.
    Illustrates high within-organ / low between-organ similarity.
    """
    mz_r, rows_per_organ = find_mz_examples(ch_meta, ORGANS_D, n_per=N_PER_ORGAN_D)
    print(f"[Panel D] selected m/z = {mz_r}")

    ax_parent.axis("off")
    if mz_r is None:
        ax_parent.text(0.5, 0.5, "No common m/z found", transform=ax_parent.transAxes,
                       ha="center", va="center", fontsize=10, color="#aaa")
        return

    fig = ax_parent.figure
    n_organs = len(ORGANS_D)
    inner = ax_parent.get_subplotspec().subgridspec(
        n_organs + 1, N_PER_ORGAN_D,
        height_ratios=[0.20] + [1.0] * n_organs, hspace=0.06, wspace=0.06)

    ax_title = fig.add_subplot(inner[0, :])
    ax_title.axis("off")
    organ_list_str = ", ".join(ORGANS_D)
    ax_title.set_title(
        f"D   Same m/z Ion Images Across Tissue Types\n"
        f"(m/z = {mz_r:.3f}  |  top to bottom: {organ_list_str})",
        fontsize=12, fontweight="bold", pad=6, color="#111")

    for row_i, (rows, organ, col) in enumerate(zip(rows_per_organ, ORGANS_D, COLORS_D)):
        for col_i, (_, row) in enumerate(rows.iterrows()):
            ax_in = fig.add_subplot(inner[row_i + 1, col_i])
            img = load_specific_channel(row["sample_path"], row["channel_idx"])
            if img is not None:
                ax_in.imshow(img, cmap="viridis", aspect="equal",
                             interpolation="antialiased")
                _slug = organ.lower().replace(" ", "_")
                _stem = f"figure3_panelD_{_slug}_{col_i + 1}"
                _f, _a = plt.subplots(figsize=(3, 3))
                _a.imshow(img, cmap="viridis", aspect="equal", interpolation="antialiased")
                _a.axis("off")
                _f.savefig(str(PANEL_DIR / f"{_stem}.svg"), bbox_inches="tight", pad_inches=0)
                plt.close(_f)
                print(f"  saved panel {_stem}.svg")
            ax_in.set_xticks([]); ax_in.set_yticks([])
            for side in ("top", "bottom", "left", "right"):
                ax_in.spines[side].set_color(col)
                ax_in.spines[side].set_linewidth(2.0)
            if col_i == 0:
                ax_in.set_ylabel(organ, fontsize=9, fontweight="bold",
                                 color=col, labelpad=3)


FIG5_STEPS = [
    {"label": "Ion Image\nChannel (HxW)", "sub": "per m/z value",            "kind": "data",   "icon": "msi",       "pos": (0, 0)},
    {"label": "Stage 1\nResNet-18 BT",   "sub": "patch-level encoder",      "kind": "model",  "icon": "resnet",    "pos": (0, 1)},
    {"label": "512-d Channel\nEmbedding","sub": "one vector per channel",   "kind": "output", "icon": "embedding", "pos": (0, 2)},
    {"label": "Pairwise\nCosine Sim.",   "sub": "within vs between m/z",   "kind": "eval",   "icon": "scoring",   "pos": (0, 3)},
    {"label": "Identity\nPreservation",  "sub": "within - between metric",  "kind": "output", "icon": "scoring",   "pos": (0, 4)},
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


# â"€â"€ PANEL A: within vs between bars â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def draw_panel_a(ax, summary):
    y     = np.arange(len(VARIANT_ORDER))
    h     = 0.28

    for i, var in enumerate(VARIANT_ORDER):
        row = summary[summary["variant"] == var]
        if row.empty:
            continue
        row = row.iloc[0]
        w = float(row["mean_within"])
        b = float(row["mean_between"])
        d = float(row["delta"])
        c = COLORS[var]

        # within bar (solid)
        ax.barh(i + h / 2, w, height=h, color=c, alpha=WITHIN_ALPHA,
                edgecolor="white", linewidth=0.4)
        # between bar (lighter)
        ax.barh(i - h / 2, b, height=h, color=c, alpha=BETWEEN_ALPHA,
                edgecolor="white", linewidth=0.4)

        # delta bracket
        x_max = max(w, b)
        ax.annotate(
            "", xy=(x_max + 0.004, i + h / 2),
            xytext=(x_max + 0.004, i - h / 2),
            arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0),
        )
        sign = "+" if d >= 0 else ""
        ax.text(x_max + 0.012, i, f"d={sign}{d:.3f}",
                va="center", ha="left", fontsize=8.5, color="#222", fontweight="bold")

        # value labels
        ax.text(w + 0.002, i + h / 2, f"{w:.3f}",
                va="center", ha="left", fontsize=8, color="#222")
        ax.text(b + 0.002, i - h / 2, f"{b:.3f}",
                va="center", ha="left", fontsize=8, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_LABELS[v] for v in VARIANT_ORDER], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cosine similarity", fontsize=11)
    ax.set_title(TITLE_B, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    legend_handles = [
        mpatches.Patch(color="#666", alpha=WITHIN_ALPHA,  label="Within-molecule"),
        mpatches.Patch(color="#666", alpha=BETWEEN_ALPHA, label="Between-molecule"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, frameon=False, loc="lower right")


# â"€â"€ PANEL B: per-mz distribution violin â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def draw_panel_b(ax, per_mz):
    variants_b = ["MetaboFM Stage 2", "Stage 1 (ResNet)", "ResNet + SMILES", "SMILES only"]
    data   = [per_mz[per_mz["variant"] == v]["within_sim"].values for v in variants_b]
    colors = [COLORS[v] for v in variants_b]
    labels = ["Stage 2", "Stage 1", "ResNet\n+SMILES", "SMILES-only"]

    parts = ax.violinplot(data, positions=range(len(variants_b)),
                          showmedians=True, showextrema=False, widths=0.65)

    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.6)
        body.set_edgecolor(c)
        body.set_linewidth(0.8)
    parts["cmedians"].set_color("#222")
    parts["cmedians"].set_linewidth(2)

    # annotate medians
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 0.38, med, f"{med:.3f}", va="center", fontsize=8, color="#222")

    ax.set_xticks(range(len(variants_b)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Within-molecule cosine similarity\n(per m/z group)", fontsize=11)
    ax.set_title(TITLE_C, fontsize=12, fontweight="bold", pad=6)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    # random-baseline reference
    ax.axhline(0, color="#bbb", lw=0.8, ls="--")
    ax.text(len(variants_b) - 0.5, 0.02, "random baseline",
            fontsize=7.5, color="#aaa", ha="right")


# â"€â"€ MAIN â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def main():
    summary  = pd.read_csv(VAR_DIR / "molecule_variance_summary.csv")
    per_mz   = pd.read_csv(VAR_DIR / "molecule_variance_per_mz.csv")
    ch_meta  = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                           usecols=["sample_path", "channel_idx", "mz", "Organism_Part"])

    print("[DATA]")
    print(summary.to_string(index=False))

    # â"€â"€ individual panels â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    fig_a, ax_a = plt.subplots(figsize=(7.5, 4.5))
    draw_panel_a(ax_a, summary)
    save_panel(fig_a, "figure3_panelB_within_between")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(6.0, 4.5))
    draw_panel_b(ax_b, per_mz)
    save_panel(fig_b, "figure3_panelC_per_mz_violin")
    plt.close(fig_b)

    # â”€â”€ Panel D: actual ion images, same m/z in Liver vs Brain (saves its own
    # per-organ sub-panel SVGs as a side effect) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig_d = plt.figure(figsize=(7.5, 8.0))
    ax_d  = fig_d.add_subplot(1, 1, 1)
    draw_panel_d(ax_d, ch_meta)
    save_panel(fig_d, "figure3_panelD_same_mz_organs")
    plt.close(fig_d)

    print("[DONE] outputs →", PANEL_DIR)


if __name__ == "__main__":
    main()



