"""
probe_patch_coherence.py
-------------------------
Cheap sanity check for whether Stage 1's patch embeddings encode spatial
structure.

Idea: extract all 784 patch tokens for a handful of ion images (no masking,
deterministic). For each patch, compare its cosine similarity to its
immediate spatial neighbors vs. to random/distant patches in the same image.
If the encoder has learned spatial structure, neighboring patches should be
reliably more similar than distant ones.

This does NOT require a fully trained model â€” run it against any checkpoint
to track whether spatial coherence improves over training.

Usage
-----
  python probe_patch_coherence.py --checkpoint ../metabofm_v2/stage1_resnet/run_XXXX/encoder_final.pt
  python probe_patch_coherence.py --checkpoint ../metabofm_v2/stage1_resnet/run_XXXX/checkpoint_epoch010.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from dataset import IonImageDataset, set_data_root
from models.resnet_encoder import build_ion_encoder_for_inference, encode_ions

CSV_PATH  = str(METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv")
DATA_ROOT = str(MSI_RAW_DIR)
IMG_SIZE  = 224
PATCH_SIZE = 8
N_GRID    = IMG_SIZE // PATCH_SIZE   # 28

N_SAMPLES   = 16     # ion images to probe
N_PAIRS     = 2000   # random distant-pair samples per image (for the "distant" baseline)
SEED        = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def patch_foreground_mask(pixel_image: torch.Tensor, frac_thresh: float = 0.05) -> torch.Tensor:
    """
    pixel_image: (H, W) tile_max-normalized image (max pixel == 1.0).
    Returns (N_GRID*N_GRID,) bool â€” True for patches with real tissue signal,
    so background-dominated patches don't dilute the coherence measurement
    (background patches are trivially similar to each other regardless of
    whether the model has learned anything about spatial structure).
    """
    grid = pixel_image.view(N_GRID, PATCH_SIZE, N_GRID, PATCH_SIZE)   # (28, 8, 28, 8)
    patch_mean = grid.mean(dim=(1, 3))                                  # (28, 28)
    return (patch_mean.flatten() > frac_thresh)


def neighbor_vs_distant_similarity(patch_tokens: torch.Tensor, fg_mask: torch.Tensor) -> tuple[float, float, int]:
    """
    patch_tokens: (N_GRID*N_GRID, D), in row-major (r*N_GRID+c) order.
    fg_mask: (N_GRID*N_GRID,) bool â€” restrict comparisons to foreground (tissue) patches
    so empty background patches don't trivially inflate both neighbor and distant
    similarity and mask the real signal.
    Returns (mean_neighbor_cos_sim, mean_distant_cos_sim, n_foreground_patches).
    """
    tokens = F.normalize(patch_tokens, dim=-1)   # (N, D)
    fg_grid = fg_mask.view(N_GRID, N_GRID)
    grid = tokens.view(N_GRID, N_GRID, -1)        # (28, 28, D)

    # Neighbor pairs: right and down adjacency, both patches must be foreground
    right_sim  = (grid[:, :-1, :] * grid[:, 1:, :]).sum(-1)        # (28, 27)
    right_keep = fg_grid[:, :-1] & fg_grid[:, 1:]
    down_sim   = (grid[:-1, :, :] * grid[1:, :, :]).sum(-1)        # (27, 28)
    down_keep  = fg_grid[:-1, :] & fg_grid[1:, :]
    neighbor_sim = torch.cat([right_sim[right_keep], down_sim[down_keep]])

    # Distant pairs: random index pairs among foreground patches only
    fg_idx = fg_mask.nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(SEED)
    if len(fg_idx) < 8:
        return float("nan"), float("nan"), int(len(fg_idx))
    idx_a = fg_idx[torch.randint(0, len(fg_idx), (N_PAIRS,), generator=g)]
    idx_b = fg_idx[torch.randint(0, len(fg_idx), (N_PAIRS,), generator=g)]
    ra, ca = idx_a // N_GRID, idx_a % N_GRID
    rb, cb = idx_b // N_GRID, idx_b % N_GRID
    far_enough = ((ra - rb).abs() + (ca - cb).abs()) >= 4
    idx_a, idx_b = idx_a[far_enough], idx_b[far_enough]
    distant_sim = (tokens[idx_a] * tokens[idx_b]).sum(-1)

    return neighbor_sim.mean().item(), distant_sim.mean().item(), int(len(fg_idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to a Stage 1 checkpoint dir (config.json + model.safetensors)")
    args = ap.parse_args()

    set_data_root(DATA_ROOT)
    model = build_ion_encoder_for_inference(args.checkpoint).to(DEVICE).eval()

    df_all = pd.read_csv(CSV_PATH, sep=None, engine="python")
    df = df_all.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)

    ds = IonImageDataset(df=df, base_dir=DATA_ROOT, img_size=IMG_SIZE, norm="tile_max", augment=False)

    neighbor_sims, distant_sims = [], []
    for i in range(len(ds)):
        item = ds[i]
        pv = item["pixel_values"].unsqueeze(0).to(DEVICE)
        out = encode_ions(model, pv)
        patch_tokens = out.patch_tokens[0]   # (784, D)
        if patch_tokens.shape[0] != N_GRID * N_GRID:
            print(f"  [WARN] sample {i}: expected {N_GRID*N_GRID} patches, got {patch_tokens.shape[0]} â€” skipping")
            continue
        fg_mask = patch_foreground_mask(item["pixel_values"][0])
        n_fg = int(fg_mask.sum())
        if n_fg < 8:
            print(f"  [WARN] sample {i}: only {n_fg} foreground patches â€” skipping")
            continue
        n_sim, d_sim, _ = neighbor_vs_distant_similarity(patch_tokens.cpu(), fg_mask)
        neighbor_sims.append(n_sim)
        distant_sims.append(d_sim)
        print(f"  sample {i:2d}  n_fg={n_fg:3d}  neighbor_sim={n_sim:.4f}  distant_sim={d_sim:.4f}  gap={n_sim - d_sim:+.4f}")

    neighbor_sims = np.array(neighbor_sims)
    distant_sims  = np.array(distant_sims)
    gap = neighbor_sims - distant_sims

    print("\n=== Summary ===")
    print(f"checkpoint:          {args.checkpoint}")
    print(f"mean neighbor sim:   {neighbor_sims.mean():.4f}")
    print(f"mean distant sim:    {distant_sims.mean():.4f}")
    print(f"mean gap (n - d):    {gap.mean():.4f}  (std {gap.std():.4f})")
    if gap.mean() > 0.05:
        print("-> Patch tokens show spatial coherence (neighbors reliably more similar than distant patches).")
    elif gap.mean() > 0.0:
        print("-> Weak spatial coherence â€” neighbors slightly more similar, but margin is small.")
    else:
        print("-> No spatial coherence detected â€” patch tokens are not encoding local structure yet.")


if __name__ == "__main__":
    main()

