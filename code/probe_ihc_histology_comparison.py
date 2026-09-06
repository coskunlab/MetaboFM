"""
probe_ihc_histology_comparison.py
------------------------------
H&E-comparison experiment (the manuscript's H&E-comparison analysis), extended to a
second, non-METASPACE modality: a MALDI-IHC (mass-tag antibody) protein-marker
dataset of mouse brain (Miralys MB Tri-Modal, alz + wt), registered to H&E by
the MAGIC pipeline before this script ever sees it.

Unlike the METASPACE organs, these channels are protein markers (GFAP, NeuN,
pTau, ...), not metabolites -- there is no m/z, so Stage 2 (which uses m/z as
positional context) does not apply. That's fine: this experiment only needs
Stage 1's single-channel patch tokens, exactly as for the METASPACE organs.

Registration here is a plain uniform downscale (MAGIC's own affine step
already resolved rotation/translation before export), not METASPACE's
rotated affine -- so alignment in embed_ihc_histology_comparison.py is a
simple coordinate scale, no optical_alignment.py machinery needed. The tissue
boundary is an existing hand-drawn polygon (tissue_border.csv, napari export,
full-resolution H&E pixel coordinates) -- no new annotation required.

Must run under the torch_gpu conda env (GPU inference). Only encodes and
saves raw tokens -- no matplotlib/sklearn/np.linalg calls, which crash this
machine's torch_gpu env; embed_ihc_histology_comparison.py (base env) does
the PCA/UMAP + plotting.

Usage
-----
  python probe_ihc_histology_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from metabofm_paths import METABOFM_ROOT, IHC_RAW_DIR

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dataset import _pad_to_square
from models.resnet_encoder import build_ion_encoder_for_inference

DATA_ROOT = IHC_RAW_DIR
OUT_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINT = METABOFM_ROOT / "checkpoints/stage1_encoder_final.pt"
CHECKPOINT = DEFAULT_CHECKPOINT  # overridden by --checkpoint in main()
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONDITIONS = ["alz", "wt"]

MALDI_IHC_LABELS = [
    "GLUT1", "Rab7", "GFAP", "LC3", "Nicastrin", "Cathepsin D", "pTau-(pS404)",
    "NeuN (C8)", "NF-L", "MBP Dual-Labeled", "AKT", "SYN-I", "PVALB",
    "GSK-3B", "a/b-Synuclein", "B3-Tubulin", "pTau (Thr205)", "Amyloid-B42",
    "Histone H2A.X Dual-Labeled",
]


def preprocess_channel(img_hw: np.ndarray) -> torch.Tensor:
    """Same inference preprocessing as probe_histology_comparison.py:
    float32 -> finite cleanup -> centered zero pad -> nearest-resize to 224
    -> tile-max normalize."""
    img = np.asarray(img_hw, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(img).unsqueeze(0)
    x = _pad_to_square(x, pad_value=0.0)
    if x.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze(0)
    vmax = x.max()
    if vmax > 0:
        x = x / vmax
    return x


@torch.no_grad()
def encode_channel(encoder, img_hw: np.ndarray) -> np.ndarray:
    x = preprocess_channel(img_hw).unsqueeze(0).to(DEVICE)
    _, patches = encoder(x)
    return patches[0].cpu().numpy()


def process_one(encoder, condition: str):
    print(f"\n=== Brain IHC ({condition}) ===")
    cond_dir = DATA_ROOT / condition
    stack = tifffile.imread(cond_dir / "maldi_ihc_resized_affine_downsampled.tif")  # (19, H, W)
    if stack.shape[0] != len(MALDI_IHC_LABELS):
        raise ValueError(f"{condition}: stack has {stack.shape[0]} channels, "
                          f"expected {len(MALDI_IHC_LABELS)} labels")
    n_ch, H, W = stack.shape

    with tifffile.TiffFile(cond_dir / "he_resized_affine.tif") as tf:
        he_shape = tf.series[0].shape  # (Hh, Wh, 3)
    Hh, Wh = he_shape[0], he_shape[1]

    border = pd.read_csv(cond_dir / "tissue_border.csv")
    # napari shapes export: axis-0 = row (y), axis-1 = col (x), in full-res H&E pixels
    tissue_border_he_yx = border[["axis-0", "axis-1"]].to_numpy(dtype=np.float64)

    print(f"[INFO] {n_ch} MALDI-IHC channels, grid ({H},{W}), H&E frame ({Hh},{Wh})")
    print(f"[INFO] encoding {n_ch} channels through Stage 1 ResNet-18 ({DEVICE}) ...")
    all_tokens = [encode_channel(encoder, stack[i].astype(np.float32)) for i in range(n_ch)]
    token_stack = np.stack(all_tokens, axis=0)  # (n_ch, 784, D)
    concat_tokens = np.transpose(token_stack, (1, 0, 2)).reshape(token_stack.shape[1], -1)
    print(f"[INFO] concatenated tokens shape={concat_tokens.shape}")

    summed_ion = np.sum(stack.astype(np.float32), axis=0)

    npz_path = OUT_DIR / f"BrainIHC_{condition}_tokens_data.npz"
    np.savez(
        npz_path,
        organ="BrainIHC",
        dataset_id=condition,
        modality="MALDI-IHC",
        n_channels_matched=n_ch,
        channel_names=np.array(MALDI_IHC_LABELS),
        preprocessing=np.array("float32->nan_to_num->center_pad_zero->nearest_224->tile_max"),
        checkpoint=np.array(str(CHECKPOINT)),
        he_path=np.array(str(cond_dir / "he_resized_affine.tif")),
        he_height_px=Hh, he_width_px=Wh,
        tissue_border_he_yx=tissue_border_he_yx.astype(np.float32),
        summed_ion=summed_ion,
        channel_images=stack.astype(np.float32),  # (n_ch, H, W), raw marker intensity per channel
        concat_tokens=concat_tokens.astype(np.float32),
        H=H, W=W,
    )
    print(f"[DONE] saved -> {npz_path}")


def main():
    global CHECKPOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="Path to a Stage 1 encoder_final.pt (see weights-v1 release)")
    args = ap.parse_args()
    CHECKPOINT = Path(args.checkpoint)

    encoder = build_ion_encoder_for_inference(str(CHECKPOINT)).to(DEVICE)
    encoder.eval()
    print(f"[INFO] loaded Stage 1 encoder from {CHECKPOINT}")

    for condition in CONDITIONS:
        try:
            process_one(encoder, condition)
        except Exception as e:
            print(f"[ERROR] {condition}: {type(e).__name__} {str(e)[:300]}")

    print("\n[NEXT] run embed_ihc_histology_comparison.py (base conda env) for PCA/UMAP + figures")


if __name__ == "__main__":
    main()
