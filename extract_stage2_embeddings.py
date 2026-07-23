"""
extract_stage2_embeddings.py
-----------------------------
Extract Stage 2 sample-level embeddings using the trained ChannelAggregator.

For each MSI sample:
  1. Load pre-extracted Stage 1 CLS tokens (from resnet_cls_embeddings.npy)
  2. Run Stage 2 forward pass → sample_cls (512-dim) + channel_refined (C, 512)
  3. Save sample_cls per sample and mean-pooled channel_refined per channel

Outputs (in <RUN_DIR>/)
-----------------------
  stage2_sample_cls.npy      (N_samples, 512)  float32  — one per MSI sample
  stage2_sample_meta.csv     N_samples rows: sample_path, n_channels
  stage2_channel_refined.npy (N_channels, 512) float32  — one per channel row
  stage2_channel_meta.csv    N_channels rows (same order as resnet_cls_meta.csv)

Usage
-----
  python extract_stage2_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.channel_aggregator import (
    load_channel_aggregator,
    STAGE1_DIM,
)
from pretrain_stage2 import SampleEmbeddingDataset, MAX_CHANNELS, USE_MZ_EMBEDDING

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
CLS_NPY  = EMB_DIR / "resnet_cls_embeddings.npy"
META_CSV = EMB_DIR / "resnet_cls_meta.csv"

RUN_DIR  = METABOFM_ROOT / "metabofm_v2/stage2_resnet/run_20260711_130252"
CKPT     = RUN_DIR / "stage2_aggregator_best.pt"

OUT_DIR  = EMB_DIR  # save alongside Stage 1 embeddings

BATCH_SIZE = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    for p in (CLS_NPY, META_CSV, CKPT):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print(f"[LOAD] CLS embeddings: {CLS_NPY}")
    cls_array = np.load(str(CLS_NPY), mmap_mode="r")
    meta_df   = pd.read_csv(META_CSV)
    print(f"  shape={cls_array.shape}  samples={meta_df['sample_path'].nunique():,}")

    print(f"[LOAD] Stage 2 checkpoint: {CKPT}")
    model = load_channel_aggregator(str(CKPT))
    model = model.to(DEVICE).eval()
    agg_dim = model.agg_dim
    print(f"  agg_dim={agg_dim}")

    dataset = SampleEmbeddingDataset(meta_df, cls_array, max_channels=MAX_CHANNELS)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=(DEVICE == "cuda"))

    # Collect sample-level outputs
    all_sample_cls: list[np.ndarray] = []
    sample_meta_rows: list[dict]     = []

    # Collect channel-level refined embeddings (in original CSV row order)
    n_channels = len(meta_df)
    stage2_channel_refined = np.zeros((n_channels, agg_dim), dtype=np.float32)

    # We need to map dataset index → original CSV row ids
    # dataset.samples[i] = (row_ids, mz_vals)
    sample_to_rowids = [row_ids for row_ids, _ in dataset.samples]
    sample_paths     = [
        meta_df.iloc[row_ids[0]]["sample_path"] for row_ids in sample_to_rowids
    ]

    print(f"[EXTRACT] {len(dataset):,} samples, batch={BATCH_SIZE} ...")
    sample_cursor = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            cls_tokens = batch["cls_tokens"].to(DEVICE)   # (B, C, 256)
            mz         = batch["mz"].to(DEVICE)           # (B, C)
            cmask      = batch["channel_mask"].to(DEVICE) # (B, C) bool
            B = cls_tokens.shape[0]

            out = model(
                cls_tokens=cls_tokens,
                mz=mz if USE_MZ_EMBEDDING else None,
                channel_mask=cmask,
            )

            sample_cls_np      = out["sample_cls"].cpu().numpy()       # (B, 512)
            channel_refined_np = out["channel_refined"].cpu().numpy()  # (B, C, 512)
            cmask_np           = cmask.cpu().numpy()                   # (B, C)

            for b in range(B):
                si = sample_cursor + b
                row_ids = sample_to_rowids[si]
                C_real  = min(len(row_ids), MAX_CHANNELS)
                n_real  = int(cmask_np[b].sum())

                all_sample_cls.append(sample_cls_np[b])
                sample_meta_rows.append({
                    "sample_path": sample_paths[si],
                    "n_channels":  len(row_ids),
                })

                # Write refined channel embeddings back to their original row positions
                stage2_channel_refined[row_ids[:C_real]] = channel_refined_np[b, :C_real]

            sample_cursor += B

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_cls_arr = np.vstack(all_sample_cls).astype(np.float32)  # (N_samples, 512)
    sample_meta_df = pd.DataFrame(sample_meta_rows)

    np.save(str(OUT_DIR / "stage2_sample_cls.npy"), sample_cls_arr)
    sample_meta_df.to_csv(OUT_DIR / "stage2_sample_meta.csv", index=False)
    print(f"[OK] stage2_sample_cls.npy        shape={sample_cls_arr.shape}")

    np.save(str(OUT_DIR / "stage2_channel_refined.npy"), stage2_channel_refined)
    meta_df.to_csv(OUT_DIR / "stage2_channel_meta.csv", index=False)
    print(f"[OK] stage2_channel_refined.npy   shape={stage2_channel_refined.shape}")

    print(f"\n[DONE] Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
