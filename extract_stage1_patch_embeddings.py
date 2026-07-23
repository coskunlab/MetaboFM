"""
extract_stage1_patch_embeddings.py
----------------------------------
Extract Stage 1 patch-level spatial embeddings from the ResNet-18 encoder.

For each MSI sample, all channel images are run through the encoder and the
resulting 28×28 patch tokens are mean-pooled across channels, giving one
spatially-resolved (28, 28, 256) metabolic map per tissue sample.

This produces pixel-level embeddings: each 8×8-pixel patch in the original
ion image gets a 256-dim vector encoding the local multi-channel metabolic
profile at that spatial location — enabling within-tissue spatial analysis.

Storage: 5600 samples × 28 × 28 × 256 × 4 bytes ≈ 4.5 GB

Outputs (in <out_dir>/)
-----------------------
  resnet_patch_embeddings.npy   (N_samples, 28, 28, 256)  float32
  resnet_patch_meta.csv         one row per sample: sample_path, img_h, img_w,
                                Organism_Part, Condition, analyzerType,
                                ionisationSource, n_channels

Usage
-----
  python extract_resnet_patch_embeddings.py --checkpoint <path/to/encoder_final.pt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.resnet_encoder import build_ion_encoder_for_inference

# ── CONFIG ────────────────────────────────────────────────────────────────────

META_CSV  = METABOFM_ROOT / "outputs/embeddings_v2/resnet_cls_meta.csv"
DATA_ROOT = str(MSI_RAW_DIR)
OUT_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"

IMG_SIZE   = 224
PATCH_GRID = 28        # 224 / 8 = 28 patches per side
PATCH_DIM  = 256
BATCH_SIZE = 64        # channels per batch within a sample
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def resolve_path(sample_path: str, data_root: str) -> Path:
    p = Path(sample_path)
    if p.is_absolute() and p.exists():
        return p
    c = Path(data_root) / p
    if c.exists():
        return c
    return p


def tile_max_normalize(img: np.ndarray) -> np.ndarray:
    mx = float(img.max())
    return img / mx if mx > 0 else img


def resize_to(img: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(img).resize((size, size), Image.NEAREST)
    return np.array(pil, dtype=np.float32)


@torch.no_grad()
def encode_batch_patches(model, imgs: np.ndarray) -> np.ndarray:
    """imgs: (B, H, W) float32 → patch_tokens: (B, 784, 256)"""
    x = torch.from_numpy(imgs[:, None, :, :]).to(DEVICE)  # (B, 1, H, W)
    _, patches = model(x)                                   # patches: (B, 784, 256)
    return patches.cpu().numpy()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--meta_csv",   default=str(META_CSV))
    ap.add_argument("--data_root",  default=DATA_ROOT)
    ap.add_argument("--out_dir",    default=str(OUT_DIR))
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / "resnet_patch_embeddings.npy"
    out_csv = out_dir / "resnet_patch_meta.csv"

    print(f"[LOAD] Checkpoint: {args.checkpoint}")
    model = build_ion_encoder_for_inference(args.checkpoint)
    model = model.to(DEVICE).eval()
    print(f"[INFO] Device: {DEVICE}")

    print(f"[LOAD] Meta CSV: {args.meta_csv}")
    meta_df = pd.read_csv(args.meta_csv)
    sample_groups = list(meta_df.groupby("sample_path", sort=False))
    N_samples = len(sample_groups)
    print(f"  {N_samples:,} samples")

    # Allocate output array
    out_arr = np.zeros((N_samples, PATCH_GRID, PATCH_GRID, PATCH_DIM), dtype=np.float32)
    meta_rows = []

    for si, (sample_path, grp) in enumerate(tqdm(sample_groups, desc="samples")):
        npz_path = resolve_path(sample_path, args.data_root)
        if not npz_path.exists():
            tqdm.write(f"[WARN] Missing: {npz_path}")
            meta_rows.append({"sample_path": sample_path, "n_channels": 0,
                               "img_h": 0, "img_w": 0})
            continue

        try:
            npz = np.load(str(npz_path), allow_pickle=False)
            imgs_raw = npz["patch"].astype(np.float32)   # (C, H, W)  uint16 → float32
        except Exception as e:
            tqdm.write(f"[WARN] {npz_path.name}: {e}")
            meta_rows.append({"sample_path": sample_path, "n_channels": 0,
                               "img_h": 0, "img_w": 0})
            continue

        # Already (C, H, W)
        img_h, img_w = imgs_raw.shape[1], imgs_raw.shape[2]

        # Select only channels present in filtered CSV for this sample
        ch_indices = grp["channel_idx"].tolist()
        n_ch = len(ch_indices)

        # Process in batches, accumulate mean patch tokens
        patch_acc = np.zeros((PATCH_GRID * PATCH_GRID, PATCH_DIM), dtype=np.float64)
        n_valid = 0

        for b_start in range(0, n_ch, args.batch_size):
            batch_idx = ch_indices[b_start: b_start + args.batch_size]
            batch_imgs = []
            for ci in batch_idx:
                if ci >= imgs_raw.shape[0]:
                    continue
                img = tile_max_normalize(imgs_raw[ci].astype(np.float32))
                img = resize_to(img, IMG_SIZE)
                batch_imgs.append(img)

            if not batch_imgs:
                continue

            batch_np = np.stack(batch_imgs, axis=0)   # (B, 224, 224)
            patches  = encode_batch_patches(model, batch_np)  # (B, 784, 256)
            patch_acc += patches.sum(axis=0)
            n_valid   += len(batch_imgs)

        if n_valid > 0:
            patch_mean = (patch_acc / n_valid).astype(np.float32)  # (784, 256)
            out_arr[si] = patch_mean.reshape(PATCH_GRID, PATCH_GRID, PATCH_DIM)

        row = grp.iloc[0]
        meta_rows.append({
            "sample_path":      sample_path,
            "n_channels":       n_valid,
            "img_h":            img_h,
            "img_w":            img_w,
            "Organism_Part":    row.get("Organism_Part", ""),
            "Condition":        row.get("Condition", ""),
            "analyzerType":     row.get("analyzerType", ""),
            "ionisationSource": row.get("ionisationSource", ""),
            "organism":         row.get("organism", ""),
            "polarity":         row.get("polarity", ""),
            "dataset_id":       row.get("dataset_id", ""),
        })

    np.save(str(out_npy), out_arr)
    pd.DataFrame(meta_rows).to_csv(out_csv, index=False)

    print(f"\n[OK] patch embeddings: {out_arr.shape}  →  {out_npy}")
    print(f"[OK] meta CSV          →  {out_csv}")
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
