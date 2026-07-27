"""
fuse_embeddings.py
------------------
Fuse ResNet-18 CLS embeddings (256-dim) with MolFormer SMILES embeddings
(768-dim) using channel-level CSR arrays from align_embeddings_v2.py.

For each channel, candidate molecule embeddings are mean-pooled and
l2-normalized to produce a single SMILES vector.

Variants
--------
  resnet_only       l2(z_cls)                                   256-dim
  smiles_only       l2(z_smi)  [unambiguous n_cand==1 only]     768-dim
  resnet+smiles     concat(l2(z_cls), l2(z_smi))               1024-dim

Note: SMILES embeddings are restricted to channels with exactly one candidate
molecule (n_cand==1). Channels with multiple structural isomers are excluded
because no candidate can be preferred from the MSI spectrum alone (all share
the same nominal mass). This directly addresses annotation ambiguity.

Output (in <out_dir>/)
-----------------------
  resnet_only.npy              (N, 256)   float32
  smiles_only.npy              (N_smi, 768)  float32
  resnet+smiles.npy            (N_smi, 1024) float32
  row_ids__<variant>.npy       int64 indices into channel CSV
  fusion_summary.csv

Usage
-----
  python fuse_embeddings_v2.py
"""

from __future__ import annotations

from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import l2_normalize, mean_pool

# â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DUMP    = METABOFM_ROOT / "metaspace_images_dump"
OUT_DIR = METABOFM_ROOT / "outputs/embeddings_v2"

CLS_NPY   = OUT_DIR / "resnet_cls_embeddings.npy"
SMI_NPY   = DUMP / "molformer_pubchem_embeddings.npy"
CHAN_FLAT  = OUT_DIR / "v2_channel_cand_rows_flat.npy"
CHAN_OFFS  = OUT_DIR / "v2_channel_cand_rows_offsets.npy"

OVERWRITE = True


# â”€â”€ HELPERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_smiles_vec(i: int, flat: np.ndarray, offs: np.ndarray,
                   Z_smi: np.ndarray, unambiguous_only: bool = True) -> np.ndarray | None:
    """Return L2-normalised SMILES embedding for channel i.

    If unambiguous_only=True (default), returns None for channels with >1 candidate
    molecule â€” addressing the annotation ambiguity concern (reviewer comment 7).
    Multiple structural isomers share the same nominal mass so no candidate can be
    preferred over another from the MSI spectrum alone.
    """
    a, b = int(offs[i]), int(offs[i + 1])
    n_cand = b - a
    if n_cand == 0:
        return None
    if unambiguous_only and n_cand > 1:
        return None
    rows = flat[a:b].astype(np.int64)
    rows = rows[(rows >= 0) & (rows < len(Z_smi))]
    if len(rows) == 0:
        return None
    # Single unambiguous candidate â€” no pooling needed
    return l2_normalize(np.asarray(Z_smi[rows[0]], dtype=np.float32))


# â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    for p in (CLS_NPY, SMI_NPY, CHAN_FLAT, CHAN_OFFS):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Run extract_resnet_embeddings.py and align_embeddings_v2.py first."
            )

    print("[LOAD] ResNet CLS embeddings ...")
    Z_cls = np.load(str(CLS_NPY), mmap_mode="r")
    N = len(Z_cls)
    print(f"  shape={Z_cls.shape}")

    print("[LOAD] MolFormer SMILES embeddings ...")
    Z_smi = np.load(str(SMI_NPY), mmap_mode="r")
    print(f"  shape={Z_smi.shape}")

    print("[LOAD] Channel CSR arrays ...")
    chan_flat = np.load(str(CHAN_FLAT), mmap_mode="r")
    chan_offs = np.load(str(CHAN_OFFS), mmap_mode="r")

    # Allocate buffers
    buf_resnet = np.zeros((N, 256),  dtype=np.float32)
    buf_smiles = np.zeros((N, 768),  dtype=np.float32)
    buf_fused  = np.zeros((N, 1024), dtype=np.float32)
    has_smi    = np.zeros(N, dtype=bool)

    print(f"[FUSE] Processing {N:,} channels ...")
    for i in tqdm(range(N)):
        z_cls = l2_normalize(np.asarray(Z_cls[i], dtype=np.float32))
        buf_resnet[i] = z_cls

        z_smi = get_smiles_vec(i, chan_flat, chan_offs, Z_smi, unambiguous_only=True)
        if z_smi is not None:
            has_smi[i]    = True
            buf_smiles[i] = z_smi
            buf_fused[i]  = l2_normalize(np.concatenate([z_cls, z_smi]))

    row_ids_all = np.arange(N, dtype=np.int64)
    row_ids_smi = np.where(has_smi)[0].astype(np.int64)

    save_plan = {
        "resnet_only":   (buf_resnet,              row_ids_all),
        "smiles_only":   (buf_smiles[row_ids_smi], row_ids_smi),
        "resnet+smiles": (buf_fused[row_ids_smi],  row_ids_smi),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, (X, rids) in save_plan.items():
        emb_p = OUT_DIR / f"{name}.npy"
        rid_p = OUT_DIR / f"row_ids__{name}.npy"
        if not OVERWRITE and emb_p.exists():
            print(f"[SKIP] {name} already exists")
            continue
        np.save(str(emb_p), X.astype(np.float32))
        np.save(str(rid_p), rids)
        summary_rows.append({"variant": name, "n_rows": len(rids), "dim": X.shape[1]})
        print(f"[OK] {name:<18s}  shape={X.shape}  n_rows={len(rids):,}")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "fusion_summary.csv", index=False)
    print(f"\n[DONE] Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

