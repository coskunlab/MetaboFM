"""
plot_figure6.py
---------------
Figure 6: Molecule Embedding Space and Drug-associated Organization.
(Merged with the former Figure 7 drug-likeness analysis. The per-class/
per-drug representative ion-image panels previously shown here and in the
old Figure 7 were removed: they were purely illustrative and did not carry
quantitative weight, and repeated attempts to select clean, non-boundary-only
representative channels for rare/specific metabolites were not reliably
successful.)

Panels:
  b  UMAP of per-m/z centroids, coloured by HMDB super_class
     (top 8 classes; remainder greyed as "Other/Unknown")
  c  Per-class MAP@10 horizontal bar chart
     (only classes with >=10 centroid groups shown)
  d  PCA scatter of molecule centroids coloured by drug_similarity;
     known drug-matched molecules highlighted in red with name labels.
  e  Boxplot of drug_similarity by HMDB super_class (top classes).
  f  Horizontal bar chart of top non-drug candidate molecules ranked by
     drug_similarity.

Usage:

  python plot_figure6.py
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from plot_utils import set_nature_style, draw_pipeline_diagram
set_nature_style()

# ── CONFIG ────────────────────────────────────────────────────────────────
CENT_DIR   = METABOFM_ROOT / "outputs/molecule_centroids"
EMB_DIR    = METABOFM_ROOT / "outputs/embeddings_v2"
DATA_DIR   = MSI_RAW_DIR
DRUG_DIR   = METABOFM_ROOT / "outputs/figure7_data"
OUT_DIR    = METABOFM_ROOT / "outputs/figures"
BENCH_DIR  = METABOFM_ROOT / "outputs/benchmarks_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR   = OUT_DIR / "figure6"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI       = 300
TOP_N_CLASSES = 8          # how many classes get distinct colour
MIN_GROUPS    = 10         # minimum groups for Panel C bar chart

# How many known drugs to annotate in Panel D
N_DRUG_LABELS   = 6
# Minimum group count for Panel E boxplot
MIN_GROUPS_E    = 10
# Top N non-drug for Panel F
TOP_NONDRUG     = 15

CMAP_SCORE = "RdYlBu_r"   # low=blue, high=red

# Colour palette — distinct, colourblind-friendly
CLASS_PALETTE = [
    "#2166ac",   # Lipids
    "#d6604d",   # Organoheterocyclic
    "#4dac26",   # Organic acids
    "#fdae6b",   # Organic oxygen
    "#9970ab",   # Benzenoids
    "#1b7837",   # Phenylpropanoids
    "#e08214",   # Nucleosides
    "#74add1",   # Org. nitrogen
]
OTHER_COLOR  = "#c8c8c8"

TITLE_B = "B   Molecule Embedding Space\n(UMAP of per-m/z centroids, n=1,671)"
TITLE_C = "C   Per-class Retrieval Performance\n(MAP@10, nearest-neighbour on centroids)"
TITLE_D = "D   Drug-likeness Landscape\n(PCA of molecule embedding centroids, n=1,671)"
TITLE_E = "E   Drug-likeness by Chemical Class\n(HMDB super_class)"
TITLE_F = "F   Top Non-drug Candidates\n(ranked by proximity to drug-matched molecules)"

CAPTION = """\
Figure 6 | MetaboFM organises chemical space according to metabolite class and identifies drug-associated regions.

a, MetaboFM molecule embedding pipeline. Per-channel ion images are encoded by Stage 1 (ResNet-18 spatial encoder) and aggregated by Stage 2 (cross-channel Transformer) to produce 512-dim channel embeddings. These are averaged over all samples per m/z to produce a single centroid embedding per metabolite. UMAP and PCA project these centroids into 2D chemical maps for the panels below.

b, UMAP of metabolite embedding centroids coloured by HMDB super-class. The 1,671 m/z group centroids form class-coherent clusters, with lipids, organic acids, and nucleosides occupying distinct regions of chemical space.

c, Per-class metabolite retrieval MAP@10. Horizontal bar chart showing mean average precision at rank 10 for nearest-neighbour retrieval within each HMDB super-class. MetaboFM achieves high MAP across diverse chemical classes. The dashed line shows SMILES-only performance as a structure-informed reference (uses MolFormer embeddings of HMDB candidate SMILES; not a component of MetaboFM).

d, PCA projection of molecule centroids coloured by drug-likeness score (n = 1,671). Drug-matched metabolites (crimson circles, numbered) form a distinct region in PCA space; non-drug metabolites in the same region (warm colours) receive high drug-likeness scores.

e, Drug-likeness by HMDB chemical class. Horizontal boxplot of drug-likeness scores stratified by HMDB super-class, sorted by median score. Nucleosides and nucleotides and benzenoids show the highest median drug-likeness, consistent with the known chemical properties of approved drugs.

f, Top non-drug candidate metabolites ranked by drug-likeness score. Horizontal bar chart of the 15 highest-scoring metabolites not matched to any known drug, excluding those with unknown HMDB class. Bar colour indicates HMDB super-class.
"""


def write_caption():
    (PANEL_DIR / "captions.txt").write_text(CAPTION, encoding="utf-8")
    print("  saved captions.txt")


# ── HELPERS ──────────────────────────────────────────────────────────────

FIG6_STEPS = [
    {"label": "Ion Image\nChannel",        "sub": "spatial input",               "kind": "data",   "icon": "msi",         "pos": (0, 0)},
    {"label": "Stage 1\nResNet-18",        "sub": "spatial image encoder",       "kind": "model",  "icon": "resnet",      "pos": (0, 1)},
    {"label": "Stage 2\nTransformer",      "sub": "cross-channel aggregation",   "kind": "model",  "icon": "transformer", "pos": (0, 2)},
    {"label": "Per-m/z\nCentroid",         "sub": "mean across all samples",     "kind": "output", "icon": "embedding",   "pos": (0, 3)},
    {"label": "UMAP &\nRetrieval",         "sub": "1,671 m/z groups",            "kind": "eval",   "icon": "umap",        "pos": (0, 4)},
]

FIG6_CONNECTIONS = [
    (0, 1),  # image -> stage1
    (1, 2),  # stage1 -> stage2
    (2, 3),  # stage2 -> centroid
    (3, 4),  # centroid -> umap
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


def shorten(name: str) -> str:
    """Shorter display names for long HMDB class strings."""
    mapping = {
        "Lipids and lipid-like molecules":            "Lipids",
        "Organoheterocyclic compounds":               "Organoheterocyclics",
        "Organic acids and derivatives":              "Organic acids",
        "Organic oxygen compounds":                   "Org. oxygen cpds.",
        "Benzenoids":                                 "Benzenoids",
        "Phenylpropanoids and polyketides":           "Phenylpropanoids",
        "Nucleosides, nucleotides, and analogues":    "Nucleosides/nts.",
        "Organic nitrogen compounds":                 "Org. nitrogen cpds.",
        "unknown":                                    "Unknown",
    }
    return mapping.get(name, name)


# ── PANEL B (UMAP) ─────────────────────────────────────────────────────────

def draw_panel_b(ax, df, top_classes, color_map):
    # plot "Other/Unknown" first (grey background)
    mask_other = ~df["hmdb_super_class"].isin(top_classes)
    ax.scatter(
        df.loc[mask_other, "umap_x"],
        df.loc[mask_other, "umap_y"],
        s=8, c=OTHER_COLOR, alpha=0.5, linewidths=0, rasterized=True,
    )

    # plot named classes on top
    for cls, col in zip(top_classes, CLASS_PALETTE):
        mask = df["hmdb_super_class"] == cls
        ax.scatter(
            df.loc[mask, "umap_x"],
            df.loc[mask, "umap_y"],
            s=12, c=col, alpha=0.80, linewidths=0, rasterized=True,
            label=shorten(cls),
        )

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(TITLE_B, fontsize=12, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    # legend
    handles = [mpatches.Patch(color=c, label=shorten(cls))
               for cls, c in zip(top_classes, CLASS_PALETTE)]
    handles.append(mpatches.Patch(color=OTHER_COLOR, label="Other / Unknown"))
    ax.legend(handles=handles, fontsize=7.5, frameon=False,
              loc="lower left", handlelength=1.2, borderaxespad=0.5)


# ── PANEL C (per-class MAP@10) ──────────────────────────────────────────────

def draw_panel_c(ax, df_pc, top_classes, color_map):
    # filter to classes with MIN_GROUPS, exclude unknown, sort by MAP@10
    df_c = df_pc[
        (df_pc["n_groups"] >= MIN_GROUPS) &
        (df_pc["hmdb_super_class"] != "unknown")
    ].copy()
    df_c = df_c.sort_values("map_at_10", ascending=True)

    y = np.arange(len(df_c))
    colors = [color_map.get(cls, OTHER_COLOR) for cls in df_c["hmdb_super_class"]]

    ax.barh(y, df_c["map_at_10"], height=0.65,
            color=colors, edgecolor="white", linewidth=0.4)

    # value + n labels — both on the right side of bar to avoid y-axis overlap
    for i, (_, row) in enumerate(df_c.iterrows()):
        ax.text(row["map_at_10"] + 0.008, i + 0.15, f"{row['map_at_10']:.3f}",
                va="center", ha="left", fontsize=8.5, color="#222")
        ax.text(row["map_at_10"] + 0.008, i - 0.20, f"n={int(row['n_groups'])}",
                va="center", ha="left", fontsize=7, color="#888")

    ax.set_yticks(y)
    ax.set_yticklabels([shorten(c) for c in df_c["hmdb_super_class"]], fontsize=9)
    ax.set_xlabel("MAP@10", fontsize=11)
    ax.set_title(TITLE_C, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0, 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.axvline(1.0, color="#bbb", lw=0.8, ls="--")
    # SMILES-only global MAP@10 reference line (RC6: circularity control)
    _bm = pd.read_csv(BENCH_DIR / "retrieval" / "results_all_variants.csv")
    SMILES_MAP10 = round(float(_bm[(_bm["variant"] == "smiles_only[all]") & (_bm["field"] == "super_class")]["map_at_k"].mean()), 3)
    ax.axvline(SMILES_MAP10, color="#d6604d", lw=1.4, ls="--", zorder=5)
    ax.text(SMILES_MAP10 + 0.01, 0.12, f"SMILES-only\n(MAP@10={SMILES_MAP10})",
            fontsize=7.5, color="#d6604d", va="bottom", ha="left",
            transform=ax.get_xaxis_transform())


# ── PANEL D (drug-likeness PCA landscape) ──────────────────────────────────

def draw_panel_d(ax, fig, df):
    """Panel D — returns numbered_drugs list for the side legend."""
    non_drug = df[~df["has_drug"]]
    drug     = df[df["has_drug"]]

    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(CMAP_SCORE)

    # non-drug: coloured by drug-likeness score
    sc = ax.scatter(
        non_drug["PC1"], non_drug["PC2"],
        c=non_drug["drug_similarity"], cmap=CMAP_SCORE,
        norm=norm, s=10, alpha=0.65, linewidths=0, rasterized=True,
    )
    # drug-matched: crimson
    ax.scatter(
        drug["PC1"], drug["PC2"],
        c="crimson", s=22, alpha=0.85, linewidths=0.3,
        edgecolors="white", zorder=3, rasterized=True,
    )

    # pick top-scoring drug-matched molecules, then greedily diversify within that set
    drug2   = drug.dropna(subset=["drug_name"]).nlargest(N_DRUG_LABELS * 4, "drug_similarity")
    # greedy farthest-point selection for spatial spread
    chosen  = [drug2.iloc[0]]
    rest    = drug2.iloc[1:].copy()
    while len(chosen) < N_DRUG_LABELS and len(rest) > 0:
        sel_xy = np.array([[r["PC1"], r["PC2"]] for r in chosen])
        rem_xy = rest[["PC1", "PC2"]].to_numpy()
        dists  = np.min(np.sqrt(((rem_xy[:, None] - sel_xy[None])**2).sum(-1)), axis=1)
        best   = rest.iloc[dists.argmax()]
        chosen.append(best)
        rest   = rest[rest.index != best.name]
    selected = pd.DataFrame(chosen).reset_index(drop=True)

    for i, row in selected.iterrows():
        num = str(i + 1)
        ax.scatter(row["PC1"], row["PC2"], s=60, c="white",
                   edgecolors="#333", linewidths=0.8, zorder=5)
        ax.text(row["PC1"], row["PC2"], num,
                ha="center", va="center", fontsize=5.5,
                fontweight="bold", color="#333", zorder=6)

    cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Drug-likeness score", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title(TITLE_D, fontsize=12, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    handles = [
        mpatches.Patch(color="crimson", label="Drug-matched"),
        mpatches.Patch(color=cmap(0.8),  label="High drug-likeness"),
        mpatches.Patch(color=cmap(0.2),  label="Low drug-likeness"),
    ]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper right")

    return selected


# ── PANEL E (drug-likeness boxplot) ─────────────────────────────────────────

def draw_panel_e(ax, df):
    # filter to classes with enough data, exclude unknown
    counts = df["hmdb_super_class"].value_counts()
    keep   = counts[counts >= MIN_GROUPS_E].index.difference(["unknown"])
    sub    = df[df["hmdb_super_class"].isin(keep)].copy()
    sub["class_short"] = sub["hmdb_super_class"].map(shorten)

    # order by median drug_similarity
    order = (
        sub.groupby("class_short")["drug_similarity"]
        .median().sort_values(ascending=True).index.tolist()
    )

    data   = [sub[sub["class_short"] == cls]["drug_similarity"].values for cls in order]
    labels = order

    bp = ax.boxplot(
        data, vert=False, patch_artist=True,
        medianprops=dict(color="#222", linewidth=2),
        whiskerprops=dict(color="#888", linewidth=0.8),
        capprops=dict(color="#888", linewidth=0.8),
        flierprops=dict(marker="o", markersize=2, alpha=0.4, markerfacecolor="#888",
                        linestyle="none"),
        boxprops=dict(linewidth=0.8),
    )

    cmap = plt.get_cmap(CMAP_SCORE)
    medians = [np.median(d) for d in data]
    for patch, med in zip(bp["boxes"], medians):
        patch.set_facecolor(cmap(med))
        patch.set_alpha(0.75)

    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Drug-likeness score", fontsize=11)
    ax.set_title(TITLE_E, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(-0.05, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.axvline(0.5, color="#ccc", lw=0.8, ls="--")


# ── PANEL F (top non-drug candidates) ───────────────────────────────────────

CLASS_COLORS_F = {
    "Lipids and lipid-like molecules":         "#2166ac",
    "Organoheterocyclic compounds":            "#d6604d",
    "Organic acids and derivatives":           "#4dac26",
    "Organic oxygen compounds":                "#fdae6b",
    "Benzenoids":                              "#9970ab",
    "Phenylpropanoids and polyketides":        "#1b7837",
    "Nucleosides, nucleotides, and analogues": "#e08214",
    "Organic nitrogen compounds":              "#74add1",
    "unknown":                                 "#c8c8c8",
}


def draw_panel_f(ax, df, df_top):
    """Horizontal bar chart of top non-drug candidates by drug-likeness score,
    coloured by HMDB super_class. Excludes molecules with unknown HMDB class."""
    # filter unknowns from the full non-drug pool and re-rank
    non_drug_known = df[
        ~df["has_drug"] &
        df["hmdb_super_class"].notna() &
        (df["hmdb_super_class"] != "unknown")
    ].copy()
    df_top = non_drug_known.nlargest(TOP_NONDRUG, "drug_similarity").reset_index(drop=True)
    df_top["label"] = df_top["mol_name"].fillna("Unknown").apply(
        lambda n: (str(n)[:32] + "...") if len(str(n)) > 34 else str(n)
    )
    # sort ascending so highest score is at top
    df_top = df_top.sort_values("drug_similarity", ascending=True).reset_index(drop=True)

    y      = np.arange(len(df_top))
    colors = [CLASS_COLORS_F.get(str(r), "#c8c8c8")
              for r in df_top["hmdb_super_class"].fillna("unknown")]

    ax.barh(y, df_top["drug_similarity"], height=0.7,
            color=colors, edgecolor="white", linewidth=0.3)

    # score labels
    for i, (_, row) in enumerate(df_top.iterrows()):
        ax.text(row["drug_similarity"] + 0.002, i, f"{row['drug_similarity']:.3f}",
                va="center", ha="left", fontsize=8, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels(df_top["label"].tolist(), fontsize=8.5)
    # colour y-axis labels by class
    for tick, (_, row) in zip(ax.get_yticklabels(), df_top.iterrows()):
        cls = str(row.get("hmdb_super_class", "unknown"))
        tick.set_color(CLASS_COLORS_F.get(cls, "#333333"))

    ax.set_xlabel("Drug-likeness score", fontsize=11)
    ax.set_title(TITLE_F, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlim(0.97, 1.005)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    # class colour legend
    seen = {}
    for cls in df_top["hmdb_super_class"].fillna("unknown"):
        cls = str(cls)
        if cls not in seen:
            seen[cls] = CLASS_COLORS_F.get(cls, "#c8c8c8")
    handles = [mpatches.Patch(color=col, label=shorten(cls))
               for cls, col in seen.items()]
    ax.legend(handles=handles, fontsize=7, frameon=False,
              loc="lower right", title="Class", title_fontsize=7)


# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    df      = pd.read_csv(CENT_DIR / "molecule_centroids.csv")
    df_pc   = pd.read_csv(CENT_DIR / "perclass_map10.csv")

    print(f"[DATA] {len(df):,} centroid groups")
    print(df_pc.to_string(index=False))

    # top classes by group count (exclude "unknown")
    df_pc_known = df_pc[df_pc["hmdb_super_class"] != "unknown"].copy()
    top_classes = df_pc_known.nlargest(TOP_N_CLASSES, "n_groups")["hmdb_super_class"].tolist()
    color_map   = {cls: col for cls, col in zip(top_classes, CLASS_PALETTE)}

    # ── panel b ──────────────────────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(6.5, 6.0))
    draw_panel_b(ax_b, df, top_classes, color_map)
    save_panel(fig_b, "figure6_panelB_umap")
    plt.close(fig_b)

    # ── panel c ──────────────────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(figsize=(6.5, 4.5))
    draw_panel_c(ax_c, df_pc, top_classes, color_map)
    save_panel(fig_c, "figure6_panelC_perclass_map")
    plt.close(fig_c)

    # ── panels d-f (drug-likeness landscape) ────────────────────────────
    df_drug = pd.read_csv(DRUG_DIR / "figure7_mol_df.csv")
    df_drug_top = pd.read_csv(DRUG_DIR / "figure7_top_nondrug.csv")
    print(f"[DATA] {len(df_drug)} centroids, {df_drug['has_drug'].sum()} drug-matched")

    fig_d, ax_d = plt.subplots(figsize=(7.0, 6.0))
    selected = draw_panel_d(ax_d, fig_d, df_drug)
    save_panel(fig_d, "figure6_panelD_pca_drug")
    plt.close(fig_d)

    # save numbered drug label legend matching the combined figure style
    lines = ["Annotated drugs:"] + [f"  {i+1}. {str(row['drug_name'])}"
                                     for i, (_, row) in enumerate(selected.iterrows())]
    text = "\n".join(lines)
    fig_leg, ax_leg = plt.subplots(figsize=(3.5, 0.22 * len(lines) + 0.2))
    fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax_leg.axis("off")
    ax_leg.text(0.0, 1.0, text,
                transform=ax_leg.transAxes, fontsize=7, va="top", ha="left",
                family="monospace", color="#333", linespacing=1.5)
    fig_leg.savefig(str(PANEL_DIR / "figure6_panelD_drug_labels.svg"),
                    bbox_inches="tight", pad_inches=0)
    plt.close(fig_leg)
    print("  saved panel figure6_panelD_drug_labels.svg")

    fig_e, ax_e = plt.subplots(figsize=(6.5, 5.5))
    draw_panel_e(ax_e, df_drug)
    save_panel(fig_e, "figure6_panelE_boxplot")
    plt.close(fig_e)

    fig_f, ax_f = plt.subplots(figsize=(6.5, 6.5))
    draw_panel_f(ax_f, df_drug, df_drug_top)
    save_panel(fig_f, "figure6_panelF_bars")
    plt.close(fig_f)

    write_caption()
    print("[DONE] outputs ->", PANEL_DIR)


if __name__ == "__main__":
    main()
