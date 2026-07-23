"""
extract_stage1_embeddings.py
----------------------------
Extract per-channel CLS embeddings (256-dim) from the Stage 1 ResNet-18
encoder for every channel in channels_v2_filtered.csv.

Channels are grouped by NPZ file so each file is loaded only once.

Output
------
  <out_dir>/resnet_cls_embeddings.npy   (N_channels, 256) float32  — row-aligned with channel CSV
  <out_dir>/resnet_cls_meta.csv         copy of channel CSV

Usage
-----
  python extract_resnet_embeddings.py --checkpoint ..\metabofm_v2\stage1_resnet\run_20260702_185458\encoder_final.pt
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

CSV_PATH  = str(METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv")
DATA_ROOT = str(MSI_RAW_DIR)
OUT_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"

IMG_SIZE   = 224
BATCH_SIZE = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def resolve_path(sample_path: str, data_root: str) -> Path:
    p = Path(sample_path)
    if p.is_absolute() and p.exists():
        return p
    # strip leading "metaspace_images_dump\" or similar shared prefix if already in data_root
    c = Path(data_root) / p
    if c.exists():
        return c
    # try just the filename parts under data_root
    return p


def tile_max_normalize(img: np.ndarray) -> np.ndarray:
    mx = float(img.max())
    return img / mx if mx > 0 else img


def pad_and_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Pad to square (zero-pad shorter side, centered), then resize — preserves aspect ratio."""
    from PIL import Image
    H, W = img.shape
    S = max(H, W)
    if H != W:
        canvas = np.zeros((S, S), dtype=np.float32)
        top  = (S - H) // 2
        left = (S - W) // 2
        canvas[top:top + H, left:left + W] = img
        img = canvas
    pil = Image.fromarray(img).resize((size, size), Image.NEAREST)
    return np.array(pil, dtype=np.float32)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to encoder_final.pt or checkpoint_epochXXX.pt")
    ap.add_argument("--csv",       default=CSV_PATH)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--out_dir",   default=str(OUT_DIR))
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--img_size",   type=int, default=IMG_SIZE)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Checkpoint : {args.checkpoint}")
    print(f"[INFO] CSV        : {args.csv}")
    print(f"[INFO] Data root  : {args.data_root}")
    print(f"[INFO] Device     : {DEVICE}")
    print(f"[INFO] Batch size : {args.batch_size}")

    encoder = build_ion_encoder_for_inference(args.checkpoint).to(DEVICE)
    encoder.eval()

    df = pd.read_csv(args.csv, sep=None, engine="python")
    N  = len(df)
    D  = 256
    print(f"[INFO] Channels   : {N:,}")

    out_npy  = out_dir / "resnet_cls_embeddings.npy"
    out_meta = out_dir / "resnet_cls_meta.csv"

    Z = np.lib.format.open_memmap(str(out_npy), mode="w+", dtype=np.float32, shape=(N, D))

    # Group by NPZ path so each file is opened once
    groups: dict[str, list[int]] = {}
    for row_i, sp in enumerate(df["sample_path"].tolist()):
        groups.setdefault(str(sp), []).append(row_i)

    errors = 0
    npz_bar = tqdm(groups.items(), desc="NPZ files", total=len(groups))

    for sp, row_indices in npz_bar:
        npz_path = resolve_path(sp, args.data_root)
        try:
            with np.load(str(npz_path), allow_pickle=False) as npz:
                patch = npz["patch"]   # (C, H, W)
        except Exception as e:
            tqdm.write(f"[WARN] Cannot load {npz_path}: {e}")
            for ri in row_indices:
                Z[ri] = 0.0
            errors += len(row_indices)
            continue

        # Build batch from this NPZ
        channel_idxs = df["channel_idx"].iloc[row_indices].tolist()

        imgs, valid_ri = [], []
        for ri, ci in zip(row_indices, channel_idxs):
            try:
                img = patch[int(ci)].astype(np.float32)
                img = tile_max_normalize(img)
                img = pad_and_resize(img, args.img_size)
                imgs.append(img)
                valid_ri.append(ri)
            except Exception as e:
                tqdm.write(f"[WARN] row {ri} ch {ci}: {e}")
                Z[ri] = 0.0
                errors += 1

        # Run in sub-batches
        for start in range(0, len(imgs), args.batch_size):
            batch_imgs = imgs[start:start + args.batch_size]
            batch_ri   = valid_ri[start:start + args.batch_size]
            x = torch.from_numpy(np.stack(batch_imgs))[:, None].to(DEVICE)  # (B, 1, H, W)
            with torch.no_grad():
                cls, _ = encoder(x)
            Z[batch_ri] = cls.cpu().numpy().astype(np.float32)

        Z.flush()

    df.to_csv(str(out_meta), index=False)

    print(f"\n[OK] Embeddings : {out_npy}  shape={Z.shape}")
    print(f"[OK] Meta       : {out_meta}")
    if errors:
        print(f"[WARN] {errors:,} channels failed (zero-filled)")


if __name__ == "__main__":
    main()
