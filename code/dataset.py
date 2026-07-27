"""
dataset.py
----------
PyTorch Dataset for MetaboFM.

Key difference from v1: each __getitem__ returns ONE ion channel image as a
(1, img_size, img_size) tensor.  The dataset is therefore indexed at the
channel level, not the sample level.  One row in the metadata CSV corresponds
to one (MSI sample, channel_index) pair.

This design:
  - Feeds Stage 1 (Ion Image Encoder) directly — no channel stacking needed.
  - Eliminates the need for channel shuffle or keep-one-channel augmentations.
  - Lets Stage 1 learn unambiguous spatial patch embeddings.
  - Lets Stage 2 (Channel Aggregator) receive the CLS tokens grouped by MSI sample.

CSV schema expected
-------------------
Required columns:
  sample_path   : relative or absolute path to the .npz MSI sample stack
  channel_idx   : integer index of this channel within the sample stack
  mz            : float, m/z value (used as positional signal in Stage 2)

Optional columns:
  split         : "train" | "val" | "test"
  dataset_id    : dataset identifier string
  channels      : total number of channels in the MSI sample (for validation)

The sample .npz files are the same format as v1:
  patch: (C, H, W) or (H, W, C) float/uint array
  mz:    (C,) float array of m/z values (optional but preferred)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ============================================================
# HELPERS
# ============================================================

DATA_ROOT: str | None = None   # set at runtime via set_data_root()


def set_data_root(path: str) -> None:
    """Set the base directory for resolving relative sample_path values."""
    global DATA_ROOT
    DATA_ROOT = str(path)


def _resolve_path(sample_path: str, base_dir: str = "") -> str:
    p = Path(sample_path)
    if p.is_absolute() and p.exists():
        return str(p)
    if base_dir:
        c = Path(base_dir) / p
        if c.exists():
            return str(c)
    if DATA_ROOT:
        c = Path(DATA_ROOT) / p
        if c.exists():
            return str(c)
    return str(p)   # return as-is; np.load will raise a clear error


import functools

# np.load on a compressed npz must decompress the WHOLE (C, H, W) array to
# return a single channel slice. Since the channel-level CSV/dataset issues
# many requests against the same sample file (and DataLoader shuffling can
# interleave requests across samples), cache the decompressed stack per file
# so repeat channel reads of the same sample are free. Small LRU size keeps
# memory bounded — increase maxsize if workers have headroom.
@functools.lru_cache(maxsize=8)
def _load_patch_stack(npz_path: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as z:
        if "patch" not in z:
            raise KeyError(f"NPZ missing 'patch': {npz_path}")
        patch = z["patch"]
    if patch.ndim != 3:
        raise ValueError(f"Expected 3D patch, got {patch.shape}")
    return np.asarray(patch, dtype=np.float32)


def _load_channel_from_npz(npz_path: str, channel_idx: int) -> np.ndarray:
    """
    Load a single channel image (H, W) from an MSI sample .npz stack.
    All samples in the dataset use (C, H, W) layout.
    Returns float32 array. Uses an LRU cache on the full decompressed
    stack so repeated channels from the same sample don't re-decompress.
    """
    patch = _load_patch_stack(npz_path)

    # All samples are stored as (C, H, W)
    if channel_idx >= patch.shape[0]:
        raise IndexError(
            f"channel_idx={channel_idx} out of bounds for patch shape {patch.shape} in {npz_path}"
        )
    return patch[channel_idx].copy()


def _pad_to_square(x: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """Pad (1, H, W) to (1, S, S) where S = max(H, W), centered."""
    _, H, W = x.shape
    S = max(H, W)
    if H == W:
        return x
    pad_w = S - W
    pad_h = S - H
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    return F.pad(x, (left, right, top, bottom), mode="constant", value=pad_value)


# ============================================================
# DATASET
# ============================================================

class IonImageDataset(Dataset):
    """
    Channel-level dataset for Stage 1 pretraining and embedding extraction.

    Each item is one ion channel image from one MSI sample.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: sample_path, channel_idx.
        Optional: mz, split, dataset_id, channels.
    base_dir : str
        Prefix for resolving relative sample_path values.
    img_size : int
        Target spatial size after pad-to-square and resize (default 224).
    norm : str
        "per_channel_zscore" (default) | "tile_max" | "none"
    augment : bool
        If True, apply spatial augmentations (h/v flip).
        Set False for deterministic embedding extraction.
    eps : float
        Small constant for z-score normalization stability.

    Returns (per __getitem__)
    -------------------------
    dict with keys:
      "pixel_values"  : Tensor (1, img_size, img_size)
      "mz"            : float tensor scalar (0.0 if unknown)
      "channel_idx"   : int tensor scalar
      "row_idx"       : int tensor scalar (index into df)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str = "",
        img_size: int = 224,
        norm: str = "per_channel_zscore",
        augment: bool = False,
        eps: float = 1e-6,
        pad_value: float = 0.0,
        resize_mode: str = "nearest",
    ):
        self.df          = df.reset_index(drop=True)
        self.base_dir    = base_dir
        self.img_size    = int(img_size)
        self.norm        = norm
        self.augment     = augment
        self.eps         = float(eps)
        self.pad_value   = float(pad_value)
        self.resize_mode = resize_mode

        if "sample_path" not in self.df.columns:
            raise ValueError("df must contain 'sample_path'")
        if "channel_idx" not in self.df.columns:
            raise ValueError("df must contain 'channel_idx'")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = _resolve_path(str(row["sample_path"]), self.base_dir)
        ch   = int(row["channel_idx"])
        mz   = float(row["mz"]) if "mz" in row.index and pd.notna(row["mz"]) else 0.0

        img = _load_channel_from_npz(path, ch)           # (H, W)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(img).unsqueeze(0).float()   # (1, H, W)

        # Spatial flips (training only) — safe on raw intensities
        if self.augment:
            if torch.rand(()).item() > 0.5:
                x = torch.flip(x, dims=[2])   # horizontal flip
            if torch.rand(()).item() > 0.5:
                x = torch.flip(x, dims=[1])   # vertical flip

        # Pad to square, then resize
        x = _pad_to_square(x, self.pad_value)
        if x.shape[-2:] != (self.img_size, self.img_size):
            align = False if self.resize_mode == "bilinear" else None
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode=self.resize_mode,
                align_corners=align,
            ).squeeze(0)

        # Normalize
        if self.norm == "per_channel_zscore":
            mean = x.mean()
            std  = x.std().clamp_min(self.eps)
            x    = ((x - mean) / std).clamp(-5.0, 5.0)
        elif self.norm == "tile_max":
            vmax = x.max()
            if vmax > 0:
                x = x / vmax
        elif self.norm != "none":
            raise ValueError(f"Unknown norm={self.norm!r}")

        # Random intensity scaling (training only) — must run AFTER normalization,
        # since raw ion intensities are not bounded to [0,1] and clamping them
        # pre-normalization saturates almost every nonzero pixel to a flat 1.0,
        # destroying within-tissue texture before the model ever sees it.
        if self.augment and self.norm in ("tile_max", "none"):
            scale = 0.5 + torch.rand(()).item()
            x = (x * scale).clamp(0.0, 1.0)

        return {
            "pixel_values": x,                                      # (1, img_size, img_size)
            "mz":           torch.tensor(mz,  dtype=torch.float32),
            "channel_idx":  torch.tensor(ch,  dtype=torch.long),
            "row_idx":      torch.tensor(idx, dtype=torch.long),
        }


# ============================================================
# COLLATOR
# ============================================================

def ion_collator(features: list[dict]) -> dict:
    """Collate IonImageDataset items into a batched dict."""
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "mz":           torch.stack([f["mz"]           for f in features]),
        "channel_idx":  torch.stack([f["channel_idx"]  for f in features]),
        "row_idx":      torch.stack([f["row_idx"]       for f in features]),
    }


def mae_ion_collator(features: list[dict]) -> dict:
    """
    Minimal collator for HuggingFace Trainer MAE pretraining.
    Only returns pixel_values — Trainer does not accept extra keys.
    """
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
    }


# ============================================================
# SAMPLE-GROUPED DATASET (for Stage 2 and sample-level inference)
# ============================================================

class SampleGroupedDataset(Dataset):
    """
    Sample-level dataset that groups all channel rows belonging to the same
    MSI sample and returns them together.  Used by Stage 2 (Channel Aggregator)
    and for sample-level embedding extraction.

    Each __getitem__ returns a variable-length list of channel items for
    one MSI sample.  Use sample_grouped_collator to batch these.

    Parameters
    ----------
    df : pd.DataFrame
        Channel-level metadata (same schema as IonImageDataset).
    max_channels : int
        Maximum channels to use per MSI sample (truncate or pad to this number).
        Channels are taken in CSV order (MSM-ranked, highest first).
    **kwargs
        Forwarded to IonImageDataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_channels: int = 32,
        base_dir: str = "",
        img_size: int = 224,
        norm: str = "per_channel_zscore",
        augment: bool = False,
        eps: float = 1e-6,
        pad_value: float = 0.0,
        resize_mode: str = "nearest",
    ):
        self.max_channels = int(max_channels)
        self._ion_ds = IonImageDataset(
            df=df, base_dir=base_dir, img_size=img_size,
            norm=norm, augment=augment, eps=eps,
            pad_value=pad_value, resize_mode=resize_mode,
        )
        # Group row indices by MSI sample (sample_path)
        df_r = df.reset_index(drop=True)
        self._sample_groups: list[list[int]] = []
        for _, grp in df_r.groupby("sample_path", sort=False):
            idxs = grp.index.tolist()
            self._sample_groups.append(idxs[:self.max_channels])

    def __len__(self) -> int:
        return len(self._sample_groups)

    def __getitem__(self, sample_idx: int) -> list[dict]:
        """Returns a list of IonImageDataset items for one MSI sample."""
        return [self._ion_ds[i] for i in self._sample_groups[sample_idx]]


# Keep TileGroupedDataset as an alias for backward compatibility
TileGroupedDataset = SampleGroupedDataset


def sample_grouped_collator(sample_batches: list[list[dict]]) -> dict:
    """
    Collate a batch of MSI samples, each sample being a list of channel dicts.

    Returns
    -------
    dict with:
      "pixel_values"  : (B, C_max, 1, H, W) — padded to max channels in batch
      "mz"            : (B, C_max)
      "channel_mask"  : (B, C_max) bool — True for real channels, False for padding
      "sample_sizes"  : list[int] — number of real channels per MSI sample
    """
    B = len(sample_batches)
    C_max = max(len(t) for t in sample_batches)
    H = sample_batches[0][0]["pixel_values"].shape[-2]
    W = sample_batches[0][0]["pixel_values"].shape[-1]

    pv      = torch.zeros(B, C_max, 1, H, W,  dtype=torch.float32)
    mz      = torch.zeros(B, C_max,            dtype=torch.float32)
    mask    = torch.zeros(B, C_max,            dtype=torch.bool)
    sizes   = []

    for b, channels in enumerate(sample_batches):
        C = len(channels)
        sizes.append(C)
        for c, item in enumerate(channels):
            pv[b, c]   = item["pixel_values"]
            mz[b, c]   = item["mz"]
            mask[b, c] = True

    return {
        "pixel_values":  pv,      # (B, C_max, 1, H, W)
        "mz":            mz,      # (B, C_max)
        "channel_mask":  mask,    # (B, C_max)
        "sample_sizes":  sizes,   # list[int] — samples per batch
    }


# Keep tile_grouped_collator as an alias for backward compatibility
tile_grouped_collator = sample_grouped_collator


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def build_ion_splits(
    csv_path: str,
    base_dir: str = "",
    img_size: int = 224,
    norm: str = "per_channel_zscore",
    augment_train: bool = True,
) -> tuple[IonImageDataset, IonImageDataset | None, IonImageDataset | None]:
    """
    Build train / val / test IonImageDatasets from a channel-level CSV.
    Returns (train_ds, val_ds, test_ds).  val/test are None if absent.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)  if "split" in df and (df["split"] == "val").any()  else None
    df_test  = df[df["split"] == "test"].reset_index(drop=True) if "split" in df and (df["split"] == "test").any() else None

    shared = dict(base_dir=base_dir, img_size=img_size, norm=norm)

    train_ds = IonImageDataset(df_train, augment=augment_train, **shared)
    val_ds   = IonImageDataset(df_val,   augment=False,         **shared) if df_val  is not None else None
    test_ds  = IonImageDataset(df_test,  augment=False,         **shared) if df_test is not None else None

    return train_ds, val_ds, test_ds


def build_sample_splits(
    csv_path: str,
    max_channels: int = 32,
    base_dir: str = "",
    img_size: int = 224,
    norm: str = "per_channel_zscore",
    augment_train: bool = False,
) -> tuple[SampleGroupedDataset, SampleGroupedDataset | None, SampleGroupedDataset | None]:
    """
    Build train / val / test SampleGroupedDatasets from a channel-level CSV.
    Used for Stage 2 pretraining and sample-level inference.
    """
    df = pd.read_csv(csv_path, sep=None, engine="python")

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)  if "split" in df and (df["split"] == "val").any()  else None
    df_test  = df[df["split"] == "test"].reset_index(drop=True) if "split" in df and (df["split"] == "test").any() else None

    shared = dict(max_channels=max_channels, base_dir=base_dir,
                  img_size=img_size, norm=norm)

    train_ds = SampleGroupedDataset(df_train, augment=augment_train, **shared)
    val_ds   = SampleGroupedDataset(df_val,   augment=False,         **shared) if df_val  is not None else None
    test_ds  = SampleGroupedDataset(df_test,  augment=False,         **shared) if df_test is not None else None

    return train_ds, val_ds, test_ds


# Keep build_tile_splits as an alias for backward compatibility
build_tile_splits = build_sample_splits
