"""
probe_molecule_variance.py
--------------------------
Systematic within- vs between-molecule embedding similarity analysis.

Molecule identity is defined by m/z value rounded to 4 decimal places
(instrument precision grouping). For each m/z group with ≥ MIN_SAMPLES
observations across different MSI samples, we compute:
  - within_sim : mean pairwise cosine similarity between embeddings of
                 the SAME molecule from DIFFERENT samples
  - between_sim: mean cosine similarity between embeddings of DIFFERENT
                 molecules (randomly sampled pairs, matched count)

Run across all available embedding variants to show which representations
best preserve molecule identity.

Outputs (in OUT_DIR)
--------------------
  molecule_variance_summary.csv   one row per variant: mean within/between/delta
  molecule_variance_per_mz.csv    one row per (variant, mz_group): within_sim

Usage
-----
  python probe_molecule_variance.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── CONFIG ─────────────────────────────────────────────────────────────────────
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR  = METABOFM_ROOT / "outputs/molecule_variance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_CSV = EMB_DIR / "stage2_channel_meta.csv"

MIN_SAMPLES    = 10    # min observations per m/z group
MAX_PER_GROUP  = 50    # cap group size to keep pairwise tractable
N_BETWEEN      = 5000  # total between-molecule pairs to sample
MZ_DECIMALS    = 4
SEED           = 42

VARIANTS = {
    "MetaboFM Stage 2":  {"emb": "stage2_channel_refined.npy",  "row_ids": None},
    "Stage 1 (ResNet)":  {"emb": "resnet_cls_embeddings.npy",   "row_ids": None},
    "ResNet + SMILES":   {"emb": "resnet+smiles.npy",           "row_ids": "row_ids__resnet+smiles.npy"},
    "SMILES only":       {"emb": "smiles_only.npy",             "row_ids": "row_ids__smiles_only.npy"},
}
# ImageNet embeddings file is all-zero (extraction not run) — excluded.

# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_variant(cfg: dict, n_total: int) -> np.ndarray | None:
    """Load embeddings aligned to the full 158k channel index.
    Variants with row_ids are subsets; others are already aligned."""
    emb_path = EMB_DIR / cfg["emb"]
    if not emb_path.exists():
        print(f"  [SKIP] {cfg['emb']} not found")
        return None
    emb = np.load(str(emb_path), mmap_mode="r").astype(np.float32)

    if cfg["row_ids"] is not None:
        ids_path = EMB_DIR / cfg["row_ids"]
        row_ids  = np.load(str(ids_path)).astype(int)
        # build full-length array with nan sentinel rows
        full = np.full((n_total, emb.shape[1]), np.nan, dtype=np.float32)
        full[row_ids] = emb
        return full
    return emb


def l2_norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def pairwise_cosine(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    a = l2_norm(a)
    if b is None:
        return a @ a.T
    return a @ l2_norm(b).T


def within_group_mean(vecs: np.ndarray) -> float:
    """Mean of upper-triangle pairwise cosine sims."""
    if len(vecs) < 2:
        return np.nan
    S = pairwise_cosine(vecs)
    n = len(vecs)
    idx = np.triu_indices(n, k=1)
    return float(S[idx].mean())


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("[LOAD] meta CSV ...")
    meta = pd.read_csv(META_CSV)
    meta["mz_r"] = meta["mz"].round(MZ_DECIMALS)
    n_total = len(meta)
    print(f"  {n_total:,} channels, {meta['mz_r'].nunique():,} unique m/z groups")

    # ── build mz groups ───────────────────────────────────────────────────────
    groups = {}
    for mz_r, grp in meta.groupby("mz_r"):
        # require observations from at least MIN_SAMPLES different MSI samples
        n_samples = grp["sample_path"].nunique()
        if n_samples < MIN_SAMPLES:
            continue
        idxs = grp.index.tolist()
        if len(idxs) > MAX_PER_GROUP:
            idxs = rng.choice(idxs, MAX_PER_GROUP, replace=False).tolist()
        groups[mz_r] = idxs

    print(f"  {len(groups):,} m/z groups with ≥{MIN_SAMPLES} samples "
          f"(capped at {MAX_PER_GROUP} obs each)")

    group_list = list(groups.items())
    all_mz     = list(groups.keys())

    # ── per-variant analysis ──────────────────────────────────────────────────
    summary_rows  = []
    per_mz_rows   = []

    for var_name, cfg in VARIANTS.items():
        print(f"\n[VARIANT] {var_name}")
        emb = load_variant(cfg, n_total)
        if emb is None:
            continue

        within_means = []
        valid_mz     = []

        for mz_r, idxs in group_list:
            vecs = emb[idxs]
            # drop nan rows (subset variants)
            mask = ~np.isnan(vecs).any(axis=1)
            vecs = vecs[mask]
            if len(vecs) < 2:
                continue
            w = within_group_mean(vecs)
            if not np.isnan(w):
                within_means.append(w)
                valid_mz.append(mz_r)
                per_mz_rows.append({
                    "variant": var_name,
                    "mz_r":    mz_r,
                    "within_sim": w,
                    "n_obs":   len(vecs),
                })

        # between-molecule: sample random pairs from different groups
        # build a pool of valid (non-nan) indices per mz group for fast sampling
        valid_pool = {}
        for mz_r in valid_mz:
            idxs = np.array(groups[mz_r])
            mask = ~np.isnan(emb[idxs]).any(axis=1)
            if mask.sum() >= 1:
                valid_pool[mz_r] = idxs[mask]
        valid_mz_pool = [m for m in valid_mz if m in valid_pool]

        n_valid_groups = len(valid_mz_pool)
        between_sims   = []
        attempts = 0
        while len(between_sims) < N_BETWEEN and attempts < N_BETWEEN * 10:
            attempts += 1
            gi, gj = rng.choice(n_valid_groups, 2, replace=False)
            mz_i, mz_j = valid_mz_pool[gi], valid_mz_pool[gj]
            ai = int(rng.choice(valid_pool[mz_i]))
            bj = int(rng.choice(valid_pool[mz_j]))
            vi, vj = emb[ai], emb[bj]
            sim = float(l2_norm(vi[None]) @ l2_norm(vj[None]).T)
            between_sims.append(sim)

        mean_within  = float(np.mean(within_means))
        mean_between = float(np.mean(between_sims)) if between_sims else np.nan
        delta        = mean_within - mean_between

        print(f"  within={mean_within:.4f}  between={mean_between:.4f}  "
              f"delta={delta:+.4f}  (n_mz={len(valid_mz)})")

        summary_rows.append({
            "variant":      var_name,
            "mean_within":  mean_within,
            "mean_between": mean_between,
            "delta":        delta,
            "n_mz_groups":  len(valid_mz),
            "n_between":    len(between_sims),
        })

    # ── save ──────────────────────────────────────────────────────────────────
    df_sum = pd.DataFrame(summary_rows)
    df_mz  = pd.DataFrame(per_mz_rows)

    df_sum.to_csv(OUT_DIR / "molecule_variance_summary.csv", index=False)
    df_mz.to_csv(OUT_DIR / "molecule_variance_per_mz.csv",  index=False)

    print("\n[RESULTS]")
    print(df_sum.to_string(index=False))
    print(f"\n[DONE] outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
