"""
probe_crossplatform_retrieval.py
---------------------------------
Cross-platform molecular channel retrieval.

For each channel embedding from a held-out platform (query), retrieve
the top-k most similar channel embeddings from all other platforms
(gallery). A retrieval is "correct" if the retrieved channel shares
the same HMDB super_class as the query.

Only unambiguous channels (n_cand_molformer == 1) with known HMDB
labels (not 'unknown') are used — both as queries and gallery entries.
This ensures labels are reliable and addresses reviewer comment 7.

This experiment tests whether MetaboFM learns chemically meaningful
representations that transfer across MSI acquisition platforms — a
capability not achievable by simple m/z lookup.

Variants compared
-----------------
  stage2_ch_refined : contextual channel embedding (Stage 2, 512-dim)
  stage1_cls        : per-channel ResNet CLS (Stage 1, 256-dim)
  mz_soft           : soft Gaussian m/z similarity baseline
  random            : shuffled embeddings

Metrics (per held-out platform, then macro-averaged)
------------------------------------------------------
  Recall@1, Recall@5, Recall@10, MRR (mean reciprocal rank)

Leave-out groups: analyzerType

Part 2 — Ambiguous-m/z class disambiguation
--------------------------------------------
Restricts to nominal-mass bins (±0.5 Da) that contain ≥2 HMDB
super_classes. Within each bin the m/z baseline is blind (all items
have the same nominal mass → random ordering). Only the spatial image
content encoded by the model can distinguish chemical classes.
Query: held-out platform channel in an ambiguous bin.
Gallery: all other-platform channels in the SAME nominal-mass bin.
Correct: same HMDB super_class.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
BM_DIR   = METABOFM_ROOT / "outputs/benchmarks_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/crossplatform_retrieval"

CH_REFINED_NPY = EMB_DIR / "stage2_channel_refined.npy"
CLS_NPY        = EMB_DIR / "resnet_cls_embeddings.npy"
IMAGENET_NPY   = EMB_DIR / "imagenet_cls_embeddings.npy"
META_CSV       = EMB_DIR / "v2_channels_with_n_cand.csv"
LABEL_CSV      = BM_DIR  / "_hmdb_cache" / "hmdb__super_class.csv"

# m/z grid for soft baseline
MZ_MIN, MZ_MAX = 50.0, 1200.0
MZ_BINS        = 2000
MZ_SIGMA       = 2.0   # Da — Gaussian width for soft m/z fingerprint

TOP_K     = [1, 5, 10]
MIN_QUERY = 20          # skip platform if fewer unambiguous query channels
SEED      = 42

# Platforms to evaluate (need enough unambiguous, labeled channels)
MIN_PLATFORM_LABELED = 200


# ── HELPERS ───────────────────────────────────────────────────────────────────

def l2_norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def mz_soft_embedding(mz_values: np.ndarray,
                       bins: int = MZ_BINS,
                       mz_min: float = MZ_MIN,
                       mz_max: float = MZ_MAX,
                       sigma: float = MZ_SIGMA) -> np.ndarray:
    """Gaussian-smoothed m/z fingerprint: (N,) → (N, bins) float32."""
    centers = np.linspace(mz_min, mz_max, bins, dtype=np.float32)   # (bins,)
    diff    = mz_values[:, None] - centers[None, :]                  # (N, bins)
    emb     = np.exp(-(diff ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return l2_norm(emb)


def retrieval_metrics(scores: np.ndarray, y_query: np.ndarray,
                      y_gallery: np.ndarray, ks: list[int]) -> dict:
    """
    scores: (N_query, N_gallery) cosine similarities
    y_query / y_gallery: integer class labels
    Returns mean Recall@k and MRR.
    """
    # Sort descending
    order = np.argsort(-scores, axis=1)   # (N_query, N_gallery)
    results = {f"R@{k}": [] for k in ks}
    results["MRR"] = []

    for qi in range(len(y_query)):
        ranked_labels = y_gallery[order[qi]]
        correct       = ranked_labels == y_query[qi]
        for k in ks:
            results[f"R@{k}"].append(int(correct[:k].any()))
        first_hit = np.where(correct)[0]
        results["MRR"].append(1.0 / (first_hit[0] + 1) if len(first_hit) else 0.0)

    return {k: float(np.mean(v)) for k, v in results.items()}


def run_retrieval_faiss(X_query: np.ndarray, X_gallery: np.ndarray,
                        y_query: np.ndarray, y_gallery: np.ndarray,
                        ks: list[int], max_k: int) -> dict:
    """Run retrieval using faiss if available, else numpy fallback."""
    try:
        import faiss
        dim = X_gallery.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X_gallery.astype(np.float32))
        _, I = index.search(X_query.astype(np.float32), max_k)
        scores = np.zeros((len(X_query), max_k), dtype=np.float32)
        for qi in range(len(X_query)):
            scores[qi] = (y_gallery[I[qi]] == y_query[qi]).astype(np.float32)
        # Build binary correct array for metrics
        results = {f"R@{k}": [] for k in ks}
        results["MRR"] = []
        for qi in range(len(y_query)):
            correct = y_gallery[I[qi]] == y_query[qi]
            for k in ks:
                results[f"R@{k}"].append(int(correct[:k].any()))
            first_hit = np.where(correct)[0]
            results["MRR"].append(1.0 / (first_hit[0] + 1) if len(first_hit) else 0.0)
        return {k: float(np.mean(v)) for k, v in results.items()}
    except ImportError:
        # numpy fallback (slower)
        scores = X_query @ X_gallery.T   # (N_q, N_g)
        return retrieval_metrics(scores, y_query, y_gallery, ks)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("[LOAD] Stage 2 channel_refined ...")
    Z2  = np.load(str(CH_REFINED_NPY), mmap_mode="r")   # (N, 512)
    print(f"  shape={Z2.shape}")

    print("[LOAD] Stage 1 CLS ...")
    Z1  = np.load(str(CLS_NPY), mmap_mode="r")           # (N, 256)
    print(f"  shape={Z1.shape}")

    print("[LOAD] ImageNet ResNet-18 (untrained-domain baseline) ...")
    Zin = np.load(str(IMAGENET_NPY), mmap_mode="r")      # (N, 512)
    print(f"  shape={Zin.shape}")

    print("[LOAD] Metadata + labels ...")
    meta   = pd.read_csv(META_CSV)
    labels = pd.read_csv(LABEL_CSV)                       # row_id, label

    # Merge label into meta by row position
    assert len(meta) == len(labels), "metadata / label length mismatch"
    meta["super_class"] = labels["label"].values

    # ── Filter to unambiguous, labeled channels ───────────────────────────────
    mask = (
        (meta["n_cand_molformer"] == 1) &
        (meta["super_class"] != "unknown") &
        meta["analyzerType"].notna()
    )
    meta_f = meta[mask].copy().reset_index(drop=True)
    orig_idx = np.where(mask)[0]   # indices into full Z2, Z1 arrays

    print(f"  Unambiguous labeled channels: {len(meta_f):,} / {len(meta):,}")

    # Encode class labels as integers
    classes   = sorted(meta_f["super_class"].unique())
    cls2int   = {c: i for i, c in enumerate(classes)}
    y_all     = meta_f["super_class"].map(cls2int).values.astype(np.int32)

    # ── Build embedding matrices for filtered set ─────────────────────────────
    print("[BUILD] Embedding matrices for filtered channels ...")
    X2  = l2_norm(np.asarray(Z2[orig_idx], dtype=np.float32))
    X1  = l2_norm(np.asarray(Z1[orig_idx], dtype=np.float32))
    Xin = l2_norm(np.asarray(Zin[orig_idx], dtype=np.float32))
    Xmz = mz_soft_embedding(meta_f["mz"].values)

    rng   = np.random.default_rng(SEED)
    Xrand = l2_norm(rng.standard_normal((len(meta_f), 512)).astype(np.float32))

    platforms = (meta_f.groupby("analyzerType")
                       .size()
                       .where(lambda s: s >= MIN_PLATFORM_LABELED)
                       .dropna()
                       .index.tolist())
    print(f"  Platforms with ≥{MIN_PLATFORM_LABELED} labeled channels: {platforms}")

    max_k = max(TOP_K)

    VARIANTS = [
        ("stage2_ch_refined", X2),
        ("stage1_cls",        X1),
        ("imagenet",          Xin),
        ("mz_soft",           Xmz),
        ("random",            Xrand),
    ]

    all_rows = []

    for plat in platforms:
        m_test  = (meta_f["analyzerType"] == plat).values
        m_train = ~m_test

        if m_test.sum() < MIN_QUERY:
            print(f"[SKIP] {plat}: only {m_test.sum()} query channels")
            continue

        y_q = y_all[m_test]
        y_g = y_all[m_train]

        # Ensure queries have at least one correct gallery item for each class
        q_classes = set(y_q)
        g_classes = set(y_g)
        valid_q   = np.array([i for i, yq in enumerate(y_q) if yq in g_classes])
        if len(valid_q) < MIN_QUERY:
            print(f"[SKIP] {plat}: insufficient valid queries ({len(valid_q)})")
            continue

        y_q_valid = y_q[valid_q]
        n_q       = len(valid_q)
        n_g       = m_train.sum()
        print(f"\n[{plat}] n_query={n_q:,}  n_gallery={n_g:,}  "
              f"n_classes={len(q_classes & g_classes)}")

        for vname, X in VARIANTS:
            Xq = X[m_test][valid_q]
            Xg = X[m_train]
            metrics = run_retrieval_faiss(Xq, Xg, y_q_valid, y_g, TOP_K, max_k)
            row = {"platform": plat, "variant": vname,
                   "n_query": n_q, "n_gallery": n_g}
            row.update(metrics)
            all_rows.append(row)
            print(f"  {vname:<22s}  " +
                  "  ".join(f"R@{k}={metrics[f'R@{k}']:.3f}" for k in TOP_K) +
                  f"  MRR={metrics['MRR']:.3f}")

    if not all_rows:
        print("[ERROR] No valid platform folds found.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "crossplatform_retrieval_results.csv", index=False)

    # ── Summary table: macro-average across platforms ─────────────────────────
    metric_cols = [f"R@{k}" for k in TOP_K] + ["MRR"]
    summ = (df.groupby("variant")[metric_cols]
              .mean()
              .round(3)
              .sort_values("R@10", ascending=False))
    summ.to_csv(OUT_DIR / "crossplatform_retrieval_summary.csv")
    print("\n=== Macro-averaged across platforms ===")
    print(summ.to_string())

    # ── Plot: R@10 per platform, grouped by variant ───────────────────────────
    plat_order = (df[df["variant"] == "stage2_ch_refined"]
                  .sort_values("R@10", ascending=False)["platform"].tolist())
    n_plat = len(plat_order)
    n_var  = len(VARIANTS)
    bar_w  = 0.8 / n_var
    x      = np.arange(n_plat)

    colors = {"stage2_ch_refined": "steelblue",
              "stage1_cls":        "seagreen",
              "imagenet":          "mediumpurple",
              "mz_soft":           "darkorange",
              "random":            "lightgray"}

    fig, axes = plt.subplots(1, len(TOP_K), figsize=(5 * len(TOP_K), 5), sharey=False)
    for ai, k in enumerate(TOP_K):
        ax = axes[ai]
        for vi, (vname, _) in enumerate(VARIANTS):
            sub = df[df["variant"] == vname].set_index("platform")
            vals = [sub.loc[p, f"R@{k}"] if p in sub.index else 0.0
                    for p in plat_order]
            offset = (vi - n_var / 2 + 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w, label=vname,
                   color=colors.get(vname, "gray"), alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(plat_order, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Recall@{k}")
        ax.set_title(f"Cross-platform retrieval — R@{k}")
        if ai == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        "Cross-platform molecular channel retrieval\n"
        "(leave-analyzerType-out, unambiguous HMDB super_class labels, n_cand==1)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "crossplatform_retrieval_Rk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # MRR bar chart
    fig2, ax2 = plt.subplots(figsize=(max(6, n_plat * 1.8), 4))
    for vi, (vname, _) in enumerate(VARIANTS):
        sub  = df[df["variant"] == vname].set_index("platform")
        mrrs = [sub.loc[p, "MRR"] if p in sub.index else 0.0 for p in plat_order]
        offset = (vi - n_var / 2 + 0.5) * bar_w
        ax2.bar(x + offset, mrrs, bar_w, label=vname,
                color=colors.get(vname, "gray"), alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(plat_order, rotation=25, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("MRR")
    ax2.set_title("Cross-platform retrieval — MRR\n"
                  "(leave-analyzerType-out, unambiguous channels only)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "crossplatform_retrieval_MRR.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"\n[DONE] Part 1 results saved to: {OUT_DIR}")

    # ── Part 2: ambiguous-m/z class disambiguation ────────────────────────────
    run_ambiguous_mz_disambiguation(meta_f, y_all, X2, X1, Xmz, Xrand,
                                    VARIANTS, TOP_K, OUT_DIR)


def run_ambiguous_mz_disambiguation(
    meta_f: pd.DataFrame,
    y_all: np.ndarray,
    X2: np.ndarray,
    X1: np.ndarray,
    Xmz: np.ndarray,
    Xrand: np.ndarray,
    VARIANTS: list,
    TOP_K: list[int],
    out_dir: Path,
    min_bin_gallery: int = 10,    # min gallery items per nominal-mass bin
    min_bin_classes: int = 2,     # min HMDB classes per bin for it to be ambiguous
    min_queries_per_platform: int = 50,
):
    """
    Within ambiguous nominal-mass bins, retrieve by embedding similarity.
    Gallery = same-bin, other-platform channels only.
    m/z baseline is random within each bin (all items share same nominal mass).
    """
    print("\n" + "="*60)
    print("[Part 2] Ambiguous-m/z class disambiguation retrieval")
    print("="*60)

    meta_f = meta_f.copy()
    meta_f["nominal_mz"] = meta_f["mz"].round(0).astype(int)

    # Find nominal-mass bins that contain ≥2 HMDB super_classes
    bin_class_counts = (meta_f.groupby("nominal_mz")["super_class"]
                               .nunique())
    ambiguous_bins   = set(bin_class_counts[bin_class_counts >= min_bin_classes].index)
    meta_amb         = meta_f[meta_f["nominal_mz"].isin(ambiguous_bins)].copy()
    print(f"  Ambiguous nominal-mass bins : {len(ambiguous_bins):,}")
    print(f"  Channels in ambiguous bins  : {len(meta_amb):,} / {len(meta_f):,}")

    if len(meta_amb) < 100:
        print("[SKIP] Not enough ambiguous-bin channels.")
        return

    # Reassign local position in filtered set
    amb_mask  = meta_f["nominal_mz"].isin(ambiguous_bins).values
    amb_idx   = np.where(amb_mask)[0]    # positions in meta_f / X2, X1, Xmz, Xrand

    # Embeddings restricted to ambiguous subset
    aX2   = X2[amb_idx]
    aX1   = X1[amb_idx]
    aXmz  = Xmz[amb_idx]
    aXrand= Xrand[amb_idx]
    ay    = y_all[amb_idx]
    ameta = meta_amb.reset_index(drop=True)

    AVARIANTS = [
        ("stage2_ch_refined", aX2),
        ("stage1_cls",        aX1),
        ("mz_soft",           aXmz),   # should be ~random within bin
        ("random",            aXrand),
    ]

    colors = {"stage2_ch_refined": "steelblue",
              "stage1_cls":        "seagreen",
              "imagenet":          "mediumpurple",
              "mz_soft":           "darkorange",
              "random":            "lightgray"}

    platforms = (ameta.groupby("analyzerType")
                      .size()
                      .where(lambda s: s >= min_queries_per_platform)
                      .dropna()
                      .index.tolist())
    print(f"  Platforms (≥{min_queries_per_platform} ambiguous channels): {platforms}")

    all_rows = []
    max_k    = max(TOP_K)

    for plat in platforms:
        m_test  = (ameta["analyzerType"] == plat).values
        m_train = ~m_test

        # Group queries by nominal_mz; gallery = same bin, other platform
        q_meta = ameta[m_test].copy().reset_index(drop=True)
        g_meta = ameta[m_train].copy().reset_index(drop=True)

        if len(q_meta) < min_queries_per_platform:
            continue

        # Build per-query retrieval within its nominal-mass bin
        q_local_idx = np.where(m_test)[0]   # positions in ameta / aX2 etc.
        g_local_idx = np.where(m_train)[0]

        g_by_bin: dict[int, np.ndarray] = {}
        for bin_val, grp in g_meta.groupby("nominal_mz"):
            g_by_bin[bin_val] = grp.index.values   # local positions in g_meta

        results_per_variant: dict[str, list[dict]] = {v: [] for v, _ in AVARIANTS}

        n_valid_q = 0
        for qi, q_row in q_meta.iterrows():
            bin_val = int(q_row["nominal_mz"])
            if bin_val not in g_by_bin:
                continue
            g_pos = g_by_bin[bin_val]   # local positions in g_meta
            if len(g_pos) < min_bin_gallery:
                continue

            yq   = ay[q_local_idx[qi]]
            yg   = ay[g_local_idx[g_pos]]

            # Skip if no gallery item shares the query class
            if not (yg == yq).any():
                continue

            n_valid_q += 1
            k_eff = min(max_k, len(g_pos))

            for vname, aX in AVARIANTS:
                xq = aX[q_local_idx[qi]]           # (dim,)
                xg = aX[g_local_idx[g_pos]]        # (n_gallery, dim)
                sims = xg @ xq                      # cosine (already L2-normed)
                order = np.argsort(-sims)[:k_eff]
                ranked_labels = yg[order]
                correct = ranked_labels == yq
                row = {}
                for k in TOP_K:
                    row[f"R@{k}"] = int(correct[:k].any())
                first_hit = np.where(correct)[0]
                row["MRR"] = 1.0 / (first_hit[0] + 1) if len(first_hit) else 0.0
                results_per_variant[vname].append(row)

        if n_valid_q < min_queries_per_platform:
            print(f"[SKIP] {plat}: only {n_valid_q} valid queries after bin filtering")
            continue

        print(f"\n[{plat}] n_valid_queries={n_valid_q:,}")
        for vname, _ in AVARIANTS:
            recs = results_per_variant[vname]
            if not recs:
                continue
            rdf = pd.DataFrame(recs)
            metric_means = rdf.mean()
            row = {"platform": plat, "variant": vname, "n_query": n_valid_q}
            for k in TOP_K:
                row[f"R@{k}"] = round(float(metric_means[f"R@{k}"]), 3)
            row["MRR"] = round(float(metric_means["MRR"]), 3)
            all_rows.append(row)
            print(f"  {vname:<22s}  " +
                  "  ".join(f"R@{k}={row[f'R@{k}']:.3f}" for k in TOP_K) +
                  f"  MRR={row['MRR']:.3f}")

    if not all_rows:
        print("[Part 2] No valid results.")
        return

    df2 = pd.DataFrame(all_rows)
    df2.to_csv(out_dir / "ambiguous_mz_disambiguation_results.csv", index=False)

    metric_cols = [f"R@{k}" for k in TOP_K] + ["MRR"]
    summ2 = (df2.groupby("variant")[metric_cols]
               .mean().round(3)
               .sort_values("R@10", ascending=False))
    summ2.to_csv(out_dir / "ambiguous_mz_disambiguation_summary.csv")
    print("\n=== Part 2 — Macro-averaged (ambiguous m/z bins) ===")
    print(summ2.to_string())

    # Plot
    plat_order = (df2[df2["variant"] == "stage2_ch_refined"]
                  .sort_values("R@10", ascending=False)["platform"].tolist())
    if not plat_order:
        plat_order = df2["platform"].unique().tolist()
    n_plat = len(plat_order)
    n_var  = len(AVARIANTS)
    bar_w  = 0.8 / n_var
    x      = np.arange(n_plat)

    fig, axes = plt.subplots(1, len(TOP_K), figsize=(5 * len(TOP_K), 5), sharey=False)
    for ai, k in enumerate(TOP_K):
        ax = axes[ai] if len(TOP_K) > 1 else axes
        for vi, (vname, _) in enumerate(AVARIANTS):
            sub  = df2[df2["variant"] == vname].set_index("platform")
            vals = [sub.loc[p, f"R@{k}"] if p in sub.index else 0.0
                    for p in plat_order]
            offset = (vi - n_var / 2 + 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w, label=vname,
                   color=colors.get(vname, "gray"), alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(plat_order, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Recall@{k}")
        ax.set_title(f"Ambiguous-m/z disambiguation — R@{k}")
        if ai == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        "Within-nominal-mass-bin class disambiguation\n"
        "(m/z baseline is random; only image features distinguish classes)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ambiguous_mz_disambiguation_Rk.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Part 2 results saved to: {out_dir}")


if __name__ == "__main__":
    main()
