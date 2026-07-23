"""
benchmarks.py
-------------
HMDB taxonomy benchmarks for MetaboFM (ResNet-18 + Barlow Twins).

Variants benchmarked (from fuse_embeddings.py):
  resnet_only       l2(z_cls)                       256-dim
  smiles_only       l2(mean_pool(candidates))       768-dim
  resnet+smiles     concat(l2(z_cls), l2(z_smi))   1024-dim

HMDB labels: majority vote over candidate CIDs per channel row.
Splits: dataset-grouped 70/10/20, repeated 3 times.

Prerequisites:
  python extract_stage1_embeddings.py --checkpoint ...
  python align_embeddings.py
  python fuse_embeddings.py

Outputs saved to <OUT_ROOT>/:
  linear_probe/results_all_variants.csv + summary + leaderboard
  retrieval/results_all_variants.csv    + summary + leaderboard
  constant_splits_report.csv

Usage
-----
  python benchmarks.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

from utils import normalize_cid, parse_cids_field, l2_normalize_rows

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")

# â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DUMP    = METABOFM_ROOT / "metaspace_images_dump"
OUT_DIR = METABOFM_ROOT / "outputs/embeddings_v2"

FILTERED_CSV = METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv"
CAND_PQ      = DUMP / "channels_with_candidates.parquet"
HMDB_PQ      = DUMP / "molformer_pubchem_index_enriched_semantic.parquet"

OUT_ROOT   = METABOFM_ROOT / "outputs/benchmarks_v2"
OUT_LP     = OUT_ROOT / "linear_probe"
OUT_RET    = OUT_ROOT / "retrieval"
HMDB_CACHE = OUT_ROOT / "_hmdb_cache"

VARIANTS_DEF = [
    # ImageNet-pretrained ResNet-18 (zero-shot, no MSI-specific training) — ablation baseline
    ("imagenet_resnet",      OUT_DIR / "imagenet_cls_embeddings.npy", OUT_DIR / "row_ids__imagenet.npy"),
    # Stage 1: MSI-pretrained ResNet-18 CLS token (256-dim)
    ("resnet_only",          OUT_DIR / "resnet_only.npy",             OUT_DIR / "row_ids__resnet_only.npy"),
    # SMILES-only MolFormer embedding (chemistry baseline, no image)
    ("smiles_only",          OUT_DIR / "smiles_only.npy",             OUT_DIR / "row_ids__smiles_only.npy"),
    # Stage 1 + post-hoc MolFormer fusion (unambiguous channels only)
    ("resnet+smiles",        OUT_DIR / "resnet+smiles.npy",           OUT_DIR / "row_ids__resnet+smiles.npy"),
    # Stage 2 channel_refined (cross-channel Transformer, 512-dim)
    ("stage2_ch_refined",    OUT_DIR / "stage2_channel_refined.npy",  OUT_DIR / "row_ids__stage2_ch_refined.npy"),
    # --- Unambiguous subset (n_cand==1): apples-to-apples comparison across all variants ---
    ("imagenet__unambig",         OUT_DIR / "imagenet__unambig.npy",              OUT_DIR / "row_ids__imagenet__unambig.npy"),
    ("resnet_only__unambig",      OUT_DIR / "resnet_only__unambig.npy",           OUT_DIR / "row_ids__resnet_only__unambig.npy"),
    ("smiles_only__unambig",      OUT_DIR / "smiles_only__unambig.npy",           OUT_DIR / "row_ids__smiles_only__unambig.npy"),
    ("resnet+smiles__unambig",    OUT_DIR / "resnet+smiles__unambig.npy",         OUT_DIR / "row_ids__resnet+smiles__unambig.npy"),
    ("stage2_ch_refined__unambig",OUT_DIR / "stage2_ch_refined__unambig.npy",     OUT_DIR / "row_ids__stage2_ch_refined__unambig.npy"),
]

SEED          = 6740
TRAIN_FRAC    = 0.70;  VAL_FRAC = 0.10;  TEST_FRAC = 0.20
SPLIT_REPEATS = 3
SPLIT_SEEDS   = [SEED + 1000 * r for r in range(SPLIT_REPEATS)]

C_FIXED    = 1.0;  MAX_ITER_BIN = 200;  MAX_ITER_MULTI = 250;  LP_TOL = 1e-4
RETR_KS    = [10]
HMDB_FIELDS = ["super_class", "class"]
UNKNOWN    = "unknown"
MIN_DS_PER_CLASS_GLOBAL = 2
MIN_ROWS_IN_TRAIN       = 50
DATASET_COL             = "dataset_id"


# â”€â”€ HMDB LABELS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _hmdb_tag(tax_str, key: str, default: str = UNKNOWN) -> str:
    if tax_str is None:
        return default
    try:
        if pd.isna(tax_str):
            return default
    except Exception:
        pass
    for part in str(tax_str).split(";"):
        part = part.strip()
        if part.lower().startswith(key.lower() + ":"):
            v = part.split(":", 1)[1].strip()
            return v if v else default
    return default


def build_cid_to_taxonomy(pq_path: Path) -> dict[str, str]:
    df = pd.read_parquet(pq_path, columns=["cid", "hmdb_taxonomy"])
    df["cid_n"] = df["cid"].astype(str).map(normalize_cid)
    return {cid: tax for cid, tax in zip(df["cid_n"], df["hmdb_taxonomy"]) if cid}


def label_row(cand_cids_str, field: str, cid_to_tax: dict) -> str:
    cids = parse_cids_field(cand_cids_str)
    tags = [_hmdb_tag(cid_to_tax.get(c), field) for c in cids if c in cid_to_tax]
    tags = [t for t in tags if t != UNKNOWN]
    if not tags:
        return UNKNOWN
    return str(pd.Series(tags).value_counts().index[0])


def compute_hmdb_labels_cached(ch_df: pd.DataFrame, cid_to_tax: dict,
                                cache_dir: Path, field: str) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv = cache_dir / f"hmdb__{field}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        m  = dict(zip(df["row_id"].astype(np.int64), df["label"].astype(str)))
        return np.array([m.get(i, UNKNOWN) for i in range(len(ch_df))], dtype=object)
    print(f"[HMDB] computing {field} for {len(ch_df):,} rows ...")
    labels = np.array(
        [label_row(v, field, cid_to_tax)
         for v in tqdm(ch_df["cand_pubchem_cids"].values, leave=False)],
        dtype=object,
    )
    pd.DataFrame({"row_id": np.arange(len(ch_df), dtype=np.int64),
                  "label": labels.astype(str)}).to_csv(csv, index=False)
    return labels


# â”€â”€ SPLITS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def stratified_ds_split(ds_ids: np.ndarray, y_ds: np.ndarray, seed: int):
    idx  = np.arange(len(ds_ids))
    sss1 = StratifiedShuffleSplit(1, test_size=TEST_FRAC, random_state=seed)
    trva, te = next(sss1.split(idx, y_ds))
    val_frac2 = VAL_FRAC / max(TRAIN_FRAC + VAL_FRAC, 1e-12)
    sss2 = StratifiedShuffleSplit(1, test_size=val_frac2, random_state=seed + 1)
    tr_r, va_r = next(sss2.split(np.arange(len(trva)), y_ds[trva]))
    return ds_ids[trva[tr_r]], ds_ids[trva[va_r]], ds_ids[te]


def build_splits_for_field(field: str, ch_dataset_vec: np.ndarray,
                            hmdb_labels: np.ndarray, split_seeds: list[int]):
    ds = pd.Series(ch_dataset_vec.astype(str))
    y  = pd.Series(hmdb_labels.astype(str))
    m_known = (y != UNKNOWN) & (y != "") & (y != "nan")
    dfk = pd.DataFrame({"dataset_id": ds[m_known].values, "label": y[m_known].values})
    ds_mode = (dfk.groupby("dataset_id")["label"]
               .agg(lambda s: str(pd.Series(s).value_counts().index[0]))
               .reset_index())
    vc_ds = ds_mode["label"].value_counts()
    ok    = vc_ds[vc_ds >= MIN_DS_PER_CLASS_GLOBAL].index.astype(str).tolist()
    ds_mode = ds_mode[ds_mode["label"].isin(ok)].reset_index(drop=True)
    if ds_mode["label"].nunique() < 2:
        raise RuntimeError(f"{field}: <2 dataset-level classes after filtering.")

    ds_ids = ds_mode["dataset_id"].astype(str).values
    y_ds   = pd.Categorical(ds_mode["label"].astype(str)).codes.astype(np.int64)
    ds2label = dict(zip(ds_mode["dataset_id"].astype(str), ds_mode["label"].astype(str)))

    splits_by_seed = {}
    report_rows    = []
    for seed in split_seeds:
        tr, va, te = stratified_ds_split(ds_ids, y_ds, seed=int(seed) + 100 + (hash(field) % 997))
        splits_by_seed[int(seed)] = {
            "train": set(map(str, tr)), "val": set(map(str, va)), "test": set(map(str, te))
        }
        report_rows.append({"field": field, "split_seed": int(seed),
                             "n_ds_total": len(ds_ids), "n_ds_train": len(tr),
                             "n_ds_val": len(va), "n_ds_test": len(te),
                             "n_classes_ds": int(len(np.unique(y_ds)))})
    return splits_by_seed, ds2label, pd.DataFrame(report_rows)


# â”€â”€ PROBES & RETRIEVAL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fit_logreg(Xtr, ytr, Xte, yte, seed: int, max_iter: int, n_cls: int) -> dict:
    solver = "saga" if n_cls <= 2 else "lbfgs"
    clf = LogisticRegression(C=C_FIXED, max_iter=max_iter, class_weight="balanced",
                              random_state=seed, solver=solver, tol=LP_TOL, n_jobs=1)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    return {"acc": float(accuracy_score(yte, yhat)),
            "f1_macro": float(f1_score(yte, yhat, average="macro"))}


def faiss_knn_ip(Xtr: np.ndarray, Xte: np.ndarray, K: int) -> np.ndarray:
    try:
        import faiss
        idx = faiss.IndexFlatIP(int(Xtr.shape[1]))
        idx.add(Xtr.astype(np.float32))
        _, I = idx.search(Xte.astype(np.float32), min(K, len(Xtr)))
        return I
    except ImportError:
        raise RuntimeError("faiss not installed â€” pip install faiss-cpu")


def recall_k(ytr, yte, I, K):
    K = min(K, I.shape[1])
    return float((ytr[I[:, :K]] == yte[:, None]).any(axis=1).mean())


def map_k(ytr, yte, I, K):
    K    = min(K, I.shape[1])
    rel  = (ytr[I[:, :K]] == yte[:, None]).astype(np.float32)
    nr   = rel.sum(axis=1)
    prec = np.cumsum(rel, axis=1) / (np.arange(K) + 1.0)[None, :]
    ap   = (prec * rel).sum(axis=1) / np.maximum(nr, 1)
    ap[nr == 0] = 0.0
    return float(ap.mean())


def purity_k(ytr, yte, I, K):
    K = min(K, I.shape[1])
    return float((ytr[I[:, :K]] == yte[:, None]).mean(axis=1).mean())


# â”€â”€ EVALUATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def evaluate_variant(variant_name: str, X: np.ndarray, row_ids: np.ndarray,
                     field: str, hmdb_labels_by_field: dict,
                     ch_dataset_vec: np.ndarray,
                     splits_by_field: dict, ds2label_by_field: dict,
                     row_mask: np.ndarray | None = None):
    y_all    = hmdb_labels_by_field[field]
    ds2label = ds2label_by_field[field]
    splits   = splits_by_field[field]

    # Apply optional global row mask (e.g. unambiguous channels only)
    if row_mask is not None:
        row_ids = row_ids[row_mask[row_ids]]

    ds_rows = ch_dataset_vec[row_ids].astype(str)
    y_rows  = y_all[row_ids].astype(str)
    m_ok    = (y_rows != UNKNOWN) & (y_rows != "") & (y_rows != "nan")
    m_ds_ok = np.array([d in ds2label for d in ds_rows[m_ok]], dtype=bool)

    row_ids_k = row_ids[m_ok][m_ds_ok]
    ds_rows_k = ds_rows[m_ok][m_ds_ok]
    y_rows_k  = y_rows[m_ok][m_ds_ok]

    if not len(row_ids_k):
        return [], []

    vc      = pd.Series(y_rows_k).value_counts()
    cats    = sorted(vc.index.astype(str).tolist(), key=lambda c: (-int(vc.get(c, 0)), c))
    y_codes = pd.Categorical(pd.Series(y_rows_k).astype(str),
                              categories=cats, ordered=True).codes.astype(np.int64)

    pos  = {int(r): i for i, r in enumerate(row_ids)}
    take = np.array([pos[int(r)] for r in row_ids_k], dtype=np.int64)
    Xk   = np.asarray(X, dtype=np.float32)[take]

    lp_rows, ret_rows = [], []
    Kmax = max(RETR_KS)

    for split_seed in SPLIT_SEEDS:
        sp   = splits[int(split_seed)]
        m_tr = np.array([d in sp["train"] for d in ds_rows_k], dtype=bool)
        m_va = np.array([d in sp["val"]   for d in ds_rows_k], dtype=bool)
        m_te = np.array([d in sp["test"]  for d in ds_rows_k], dtype=bool)
        tr, va, te = np.where(m_tr)[0], np.where(m_va)[0], np.where(m_te)[0]

        if len(tr) < 10 or len(te) < 10:
            continue

        vc_tr  = pd.Series(y_codes[tr]).value_counts()
        ok_cls = vc_tr[vc_tr >= MIN_ROWS_IN_TRAIN].index.astype(int).tolist()
        if len(ok_cls) < 2:
            continue

        m_okc = np.isin(y_codes, ok_cls)
        tr2, va2, te2 = tr[m_okc[tr]], va[m_okc[va]], te[m_okc[te]]
        if len(tr2) < 10 or len(te2) < 10:
            continue

        remap = {c: i for i, c in enumerate(sorted(ok_cls))}
        y2    = np.array([remap.get(int(c), -1) for c in y_codes], dtype=np.int64)
        ytr2, yte2 = y2[tr2], y2[te2]
        if (ytr2 < 0).any() or (yte2 < 0).any():
            continue
        if len(np.unique(ytr2)) < 2 or len(np.unique(yte2)) < 2:
            continue

        Xtr, Xte = Xk[tr2], Xk[te2]
        n_cls    = int(len(np.unique(y2[y2 >= 0])))
        max_it   = MAX_ITER_BIN if n_cls == 2 else MAX_ITER_MULTI
        seed_h   = int(split_seed) + 77 + (hash(field + variant_name) % 997)

        res = fit_logreg(Xtr, ytr2, Xte, yte2, seed=seed_h, max_iter=max_it, n_cls=n_cls)
        lp_rows.append({
            "variant": variant_name, "field": field, "split_seed": int(split_seed),
            "n_train": len(ytr2), "n_val": len(va2), "n_test": len(yte2),
            "n_classes": n_cls, "test_acc": res["acc"], "test_f1_macro": res["f1_macro"],
        })

        I = faiss_knn_ip(l2_normalize_rows(Xtr), l2_normalize_rows(Xte), K=Kmax)
        for K in RETR_KS:
            K_eff = min(K, len(ytr2))
            if K_eff < 1:
                continue
            ret_rows.append({
                "variant": variant_name, "field": field, "split_seed": int(split_seed),
                "K": K_eff, "n_train": len(ytr2), "n_test": len(yte2), "n_classes": n_cls,
                "recall_at_k": recall_k(ytr2, yte2, I, K_eff),
                "map_at_k":    map_k(ytr2,    yte2, I, K_eff),
                "purity_at_k": purity_k(ytr2, yte2, I, K_eff),
            })

    return lp_rows, ret_rows


# â”€â”€ SUMMARIES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def summarize_lp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df.groupby(["field", "variant"], as_index=False)
              .agg(mean_f1=("test_f1_macro", "mean"), std_f1=("test_f1_macro", "std"),
                   mean_acc=("test_acc", "mean"), n=("test_f1_macro", "count"))
              .sort_values(["field", "mean_f1"], ascending=[True, False])
              .reset_index(drop=True))


def summarize_ret(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df.groupby(["field", "variant", "K"], as_index=False)
              .agg(recall_mean=("recall_at_k", "mean"), recall_std=("recall_at_k", "std"),
                   map_mean=("map_at_k", "mean"),       map_std=("map_at_k", "std"),
                   purity_mean=("purity_at_k", "mean"), purity_std=("purity_at_k", "std"),
                   n=("recall_at_k", "count"))
              .sort_values(["field", "K", "recall_mean"], ascending=[True, True, False])
              .reset_index(drop=True))


# â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    for p in (FILTERED_CSV, CAND_PQ, HMDB_PQ):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
    for name, emb_p, rid_p in VARIANTS_DEF:
        if not emb_p.exists():
            print(f"  [SKIP] {name}: embedding not found ({emb_p.name}) — run extract script first")
            continue
        if not rid_p.exists():
            # Auto-create row_ids for variants that cover all channels
            _N = np.load(str(emb_p), mmap_mode="r").shape[0]
            np.save(str(rid_p), np.arange(_N, dtype=np.int64))
            print(f"[INFO] Created row_ids for {name}: arange({_N})")

    for p in (OUT_LP, OUT_RET, HMDB_CACHE):
        p.mkdir(parents=True, exist_ok=True)

    # Load channel CSV and join candidate CIDs via manifest_row
    print("[LOAD] Channel CSV ...")
    df_flt = pd.read_csv(FILTERED_CSV)
    N = len(df_flt)
    print(f"  {N:,} channels")

    print("[LOAD] Candidate CIDs ...")
    df_cand = pd.read_parquet(CAND_PQ, columns=["manifest_row", "cand_pubchem_cids", "n_cand_molformer"])
    df_cand = df_cand.drop_duplicates("manifest_row").set_index("manifest_row")
    df_flt["cand_pubchem_cids"]  = df_flt["manifest_row"].map(df_cand["cand_pubchem_cids"])
    df_flt["n_cand_molformer"]   = df_flt["manifest_row"].map(df_cand["n_cand_molformer"]).fillna(0).astype(int)
    unambiguous_mask = (df_flt["n_cand_molformer"] == 1).to_numpy()
    print(f"  unambiguous channels (n_cand==1): {unambiguous_mask.sum():,} / {N:,}")

    ch_dataset_vec = df_flt[DATASET_COL].astype(str).to_numpy()

    print("[LOAD] CID â†’ HMDB taxonomy ...")
    cid_to_tax = build_cid_to_taxonomy(HMDB_PQ)
    print(f"  {len(cid_to_tax):,} CID entries")

    hmdb_labels_by_field = {f: compute_hmdb_labels_cached(df_flt, cid_to_tax, HMDB_CACHE, f)
                             for f in HMDB_FIELDS}

    splits_by_field, ds2label_by_field, split_reports = {}, {}, []
    for field in HMDB_FIELDS:
        sbs, ds2l, rep = build_splits_for_field(
            field, ch_dataset_vec, hmdb_labels_by_field[field], SPLIT_SEEDS
        )
        splits_by_field[field]   = sbs
        ds2label_by_field[field] = ds2l
        split_reports.append(rep)
    pd.concat(split_reports, ignore_index=True).to_csv(OUT_ROOT / "constant_splits_report.csv", index=False)

    print("[LOAD] Embedding variants ...")
    variants = []
    for name, emb_p, rid_p in VARIANTS_DEF:
        X    = np.load(str(emb_p), mmap_mode="r")
        rids = np.load(str(rid_p)).astype(np.int64)
        variants.append({"variant": name, "X": X, "row_ids": rids})
        print(f"  {name:<18s}  X={X.shape}  n_rows={len(rids):,}")

    # Metadata-only baseline: one-hot encode technical metadata columns
    # Tests whether technical metadata alone (platform, organism, polarity) solves the tasks.
    print("  Building metadata_only baseline ...")
    meta_cols = ["ionisationSource", "organism", "polarity", "analyzerType"]
    available = [c for c in meta_cols if c in df_flt.columns]
    X_meta = pd.get_dummies(
        df_flt[available].fillna("unknown"), prefix_sep="="
    ).values.astype(np.float32)
    norms = np.linalg.norm(X_meta, axis=1, keepdims=True)
    X_meta = X_meta / np.where(norms > 0, norms, 1.0)
    variants.append({
        "variant": "metadata_only",
        "X": X_meta,
        "row_ids": np.arange(N, dtype=np.int64),
    })
    print(f"  {'metadata_only':<18s}  X={X_meta.shape}  n_rows={N:,}")

    # m/z-only baseline: single feature — the nominal m/z value of each channel.
    # HMDB super_class is largely deterministic from m/z (lipids ~700-1000 Da,
    # amino acids ~100-300 Da). If this baseline is competitive with stage2,
    # the HMDB benchmark is measuring m/z range, not image content.
    print("  Building mz_only baseline ...")
    mz_col = None
    for col in ["mz", "m/z", "moverz", "mz_value", "mass"]:
        if col in df_flt.columns:
            mz_col = col
            break
    if mz_col is not None:
        X_mz = df_flt[mz_col].fillna(0.0).values.astype(np.float32).reshape(-1, 1)
        X_mz = (X_mz - X_mz.mean()) / (X_mz.std() + 1e-8)
        variants.append({
            "variant": "mz_only",
            "X": X_mz,
            "row_ids": np.arange(N, dtype=np.int64),
        })
        print(f"  {'mz_only':<18s}  X={X_mz.shape}  n_rows={N:,}  col='{mz_col}'")
    else:
        print(f"  [SKIP] mz_only: no m/z column found in CSV (tried: mz, m/z, moverz, mz_value, mass)")

    subsets = [
        ("all",          None),
        ("unambiguous",  unambiguous_mask),
    ]

    all_lp, all_ret = [], []
    for v in tqdm(variants, desc="variants"):
        for subset_name, mask in subsets:
            vname = f"{v['variant']}[{subset_name}]"
            for field in HMDB_FIELDS:
                lp_rows, ret_rows = evaluate_variant(
                    vname, v["X"], v["row_ids"], field,
                    hmdb_labels_by_field, ch_dataset_vec,
                    splits_by_field, ds2label_by_field,
                    row_mask=mask,
                )
                all_lp.extend(lp_rows)
                all_ret.extend(ret_rows)

    df_lp  = pd.DataFrame(all_lp)
    df_ret = pd.DataFrame(all_ret)
    df_lp.to_csv(OUT_LP  / "results_all_variants.csv", index=False)
    df_ret.to_csv(OUT_RET / "results_all_variants.csv", index=False)

    lp_summ = summarize_lp(df_lp)
    if not lp_summ.empty:
        lp_summ.to_csv(OUT_LP / "summary.csv", index=False)
        print("\n=== Linear Probe ===")
        print(lp_summ.to_string(index=False))

    ret_summ = summarize_ret(df_ret)
    if not ret_summ.empty:
        ret_summ.to_csv(OUT_RET / "summary.csv", index=False)
        print("\n=== Retrieval ===")
        print(ret_summ.to_string(index=False))

    print(f"\n[DONE] Outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()

