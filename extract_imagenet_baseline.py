"""
extract_imagenet_baseline.py
-----------------------------
Extract per-channel embeddings from an ImageNet-pretrained ResNet-18
applied ZERO-SHOT to MSI ion images.

This is the critical ablation: does MSI-specific pretraining (Stage 1) add
value over a general-purpose visual encoder trained on natural images?

Adaptation for single-channel MSI:
  The ImageNet ResNet-18 expects 3-channel (RGB) input. We average the first
  conv weights across the 3 input channels to obtain a 1-channel conv weight
  (equivalent to treating the ion image as a grayscale input to all 3 channels).
  The rest of the network is unchanged.

Output embedding: 512-dim global average pool of layer4 (before the ImageNet FC head).
We do NOT project to 256-dim — using the raw 512-dim is more honest for a baseline
because it avoids any learned adaptation.

Output
------
  <out_dir>/imagenet_cls_embeddings.npy   (N_channels, 512) float32
  <out_dir>/row_ids__imagenet.npy         (N_channels,) int64

Usage
-----
  python extract_imagenet_baseline.py
"""

from __future__ import annotations
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH  = str(METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv")
DATA_ROOT = str(MSI_RAW_DIR)
OUT_DIR   = METABOFM_ROOT / "outputs/embeddings_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalisation stats (mean/std for each channel, applied to single-channel)
IMGNET_MEAN = 0.449   # mean of [0.485, 0.456, 0.406]
IMGNET_STD  = 0.226   # mean of [0.229, 0.224, 0.225]

EMBED_DIM = 512       # raw GAP output of ResNet-18 layer4 (no projection)
# ── ────────────────────────────────────────────────────────────────────────────


def build_imagenet_encoder() -> nn.Module:
    """
    ImageNet ResNet-18 adapted for single-channel input.
    Averages first conv weights across RGB channels → grayscale conv.
    Returns model that outputs 512-dim GAP of layer4.
    """
    base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

    # Adapt first conv: average RGB weights → single-channel
    w = base.conv1.weight.data          # (64, 3, 7, 7)
    w_gray = w.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    new_conv.weight.data = w_gray
    base.conv1 = new_conv

    # Remove the classification head; keep up to global average pool
    encoder = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4,
        base.avgpool,                          # (B, 512, 1, 1)
        nn.Flatten(),                          # (B, 512)
    )
    return encoder


def resolve_path(sample_path: str) -> Path:
    p = Path(sample_path)
    if p.is_absolute() and p.exists():
        return p
    c = Path(DATA_ROOT) / p
    if c.exists():
        return c
    return p


def tile_max_normalize(img: np.ndarray) -> np.ndarray:
    mx = float(img.max())
    if mx > 0:
        img = img / mx
    return img.astype(np.float32)


def prepare_image(img: np.ndarray) -> torch.Tensor:
    """Resize to IMG_SIZE, normalize to ImageNet stats, return (1, H, W) tensor."""
    img = tile_max_normalize(img)
    t = torch.from_numpy(img).unsqueeze(0)          # (1, H, W)
    t = TF.resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
    t = (t - IMGNET_MEAN) / IMGNET_STD              # ImageNet normalisation
    return t


@torch.no_grad()
def extract_embeddings(encoder: nn.Module, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    encoder.to(DEVICE)

    all_embs = np.zeros((len(df), EMBED_DIM), dtype=np.float32)
    row_ids  = df.index.to_numpy(dtype=np.int64)

    # Group by NPZ file to load each file only once
    groups = df.groupby("sample_path", sort=False)
    row_cursor = {idx: i for i, idx in enumerate(df.index)}

    for npz_path, group in tqdm(groups, desc="Files", unit="npz"):
        full_path = resolve_path(npz_path)
        try:
            npz = np.load(str(full_path))
            imgs = npz["data"]                       # (H, W, C) or (C, H, W)
            if imgs.ndim == 3 and imgs.shape[2] > imgs.shape[0]:
                imgs = imgs.transpose(2, 0, 1)       # ensure (C, H, W)
        except Exception as e:
            print(f"  SKIP {npz_path}: {e}")
            continue

        batch_imgs, batch_indices = [], []
        for _, row in group.iterrows():
            ch_idx = int(row["channel_idx"])
            if ch_idx >= imgs.shape[0]:
                continue
            img = imgs[ch_idx]
            batch_imgs.append(prepare_image(img))
            batch_indices.append(row_cursor[row.name])

            if len(batch_imgs) >= BATCH_SIZE:
                batch = torch.stack(batch_imgs).to(DEVICE)
                embs  = encoder(batch).cpu().numpy()
                for j, slot in enumerate(batch_indices):
                    all_embs[slot] = embs[j]
                batch_imgs, batch_indices = [], []

        if batch_imgs:
            batch = torch.stack(batch_imgs).to(DEVICE)
            embs  = encoder(batch).cpu().numpy()
            for j, slot in enumerate(batch_indices):
                all_embs[slot] = embs[j]

    return all_embs, row_ids


def main():
    print(f"[DEVICE] {DEVICE}")
    print("[LOAD] Channel CSV ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} channels")

    print("[BUILD] ImageNet ResNet-18 encoder (weights averaged to 1-channel) ...")
    encoder = build_imagenet_encoder()
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {total_params:,}  |  Output dim: {EMBED_DIM}")

    print("[EXTRACT] Running inference ...")
    embs, row_ids = extract_embeddings(encoder, df)

    out_emb = OUT_DIR / "imagenet_cls_embeddings.npy"
    out_ids = OUT_DIR / "row_ids__imagenet.npy"
    np.save(str(out_emb), embs)
    np.save(str(out_ids), row_ids)
    print(f"[DONE] {out_emb}  shape={embs.shape}")
    print(f"       {out_ids}")


if __name__ == "__main__":
    main()
