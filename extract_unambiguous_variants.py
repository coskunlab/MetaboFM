"""
extract_unambiguous_variants.py
--------------------------------
For every benchmark variant, creates a parallel embedding file restricted to
the unambiguous channels (n_cand == 1) — the same subset used by smiles_only.

This enables a fully apples-to-apples comparison across all variants on the
same 35k channels, directly addressing the circularity concern: SMILES-only
uses chemical structure information while all other variants use ion images
only, but all see identical rows.

Outputs (EMB_DIR)
-----------------
  <variant>__unambig.npy          subset embeddings
  row_ids__<variant>__unambig.npy corresponding channel CSV row indices
"""

from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np

EMB_DIR = METABOFM_ROOT / "outputs/embeddings_v2"

# smiles_only row_ids define the unambiguous set (n_cand == 1)
smi_rids = np.load(str(EMB_DIR / "row_ids__smiles_only.npy"))
smi_set  = set(smi_rids.tolist())
print(f"Unambiguous channels (n_cand==1): {len(smi_set):,}")

VARIANTS = [
    ("imagenet_cls_embeddings",  "row_ids__imagenet",          "imagenet__unambig"),
    ("resnet_only",              "row_ids__resnet_only",        "resnet_only__unambig"),
    ("smiles_only",              "row_ids__smiles_only",        "smiles_only__unambig"),
    ("resnet+smiles",            "row_ids__resnet+smiles",      "resnet+smiles__unambig"),
    ("stage2_channel_refined",   "row_ids__stage2_ch_refined",  "stage2_ch_refined__unambig"),
    ("stage2_unambig",           "row_ids__stage2_unambig",     None),  # already done, skip
]

for emb_stem, rid_stem, out_stem in VARIANTS:
    if out_stem is None:
        continue

    emb_path = EMB_DIR / f"{emb_stem}.npy"
    rid_path = EMB_DIR / f"{rid_stem}.npy"

    if not emb_path.exists():
        print(f"  [SKIP] {emb_stem}.npy not found")
        continue
    if not rid_path.exists():
        # auto row_ids: arange over full embedding
        N = np.load(str(emb_path), mmap_mode="r").shape[0]
        rids = np.arange(N, dtype=np.int64)
    else:
        rids = np.load(str(rid_path))

    emb = np.load(str(emb_path), mmap_mode="r")

    mask = np.array([int(r) in smi_set for r in rids], dtype=bool)
    emb_sub  = np.asarray(emb[mask], dtype=np.float32)
    rids_sub = rids[mask]

    np.save(str(EMB_DIR / f"{out_stem}.npy"),          emb_sub)
    np.save(str(EMB_DIR / f"row_ids__{out_stem}.npy"), rids_sub)
    print(f"  {emb_stem}: {len(rids):,} -> {len(rids_sub):,}  saved {out_stem}.npy")

print("Done.")
