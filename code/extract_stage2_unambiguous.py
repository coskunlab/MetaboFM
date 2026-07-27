"""
extract_stage2_unambiguous.py
-----------------------------
Creates a Stage 2 embedding subset restricted to the same unambiguous channels
(n_cand == 1) used by the smiles_only / resnet+smiles baselines.

This allows a direct apples-to-apples comparison between Stage 2 and SMILES-only
on identical rows, without inflating the comparison by evaluating SMILES-only on
an easier (unambiguous) subset while Stage 2 sees all channels.

Outputs
-------
  benchmarks_v2/fused/stage2_unambig.npy          (N_unambig, 512)
  benchmarks_v2/fused/row_ids__stage2_unambig.npy (N_unambig,)
"""

from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np

FUSED_DIR = METABOFM_ROOT / "outputs/embeddings_v2"

s2_emb  = np.load(str(FUSED_DIR / "stage2_channel_refined.npy"), mmap_mode="r")
s2_rids = np.load(str(FUSED_DIR / "row_ids__stage2_ch_refined.npy"))
smi_rids = np.load(str(FUSED_DIR / "row_ids__smiles_only.npy"))

print(f"Stage 2 total channels : {len(s2_rids):,}")
print(f"SMILES-only (n_cand==1): {len(smi_rids):,}")

smi_set = set(smi_rids.tolist())
mask = np.array([int(r) in smi_set for r in s2_rids], dtype=bool)

s2_unambig_emb  = np.asarray(s2_emb[mask], dtype=np.float32)
s2_unambig_rids = s2_rids[mask]

print(f"Stage 2 unambiguous    : {len(s2_unambig_rids):,}")

np.save(str(FUSED_DIR / "stage2_unambig.npy"),           s2_unambig_emb)
np.save(str(FUSED_DIR / "row_ids__stage2_unambig.npy"), s2_unambig_rids)
print("Saved stage2_unambig.npy and row_ids__stage2_unambig.npy")
