"""
models/resnet_encoder.py
------------------------
Stage 1: ResNet-18 spatial encoder with Barlow Twins pretraining.

Replaces ViT-MAE. CNN local receptive fields preserve spatial structure
by construction — no need to learn it from pixel reconstruction.

Architecture
------------
  Input  : (B, 1, 224, 224) single-channel ion image
  Encoder: ResNet-18 (modified for 1-channel input, trained from scratch)
    - layer2 output → 28×28 × 128  → projected to (B, 784, D_patch)  patch tokens
    - layer4 output → global avg    → projected to (B, D_cls)          CLS token

  D_patch = D_cls = EMBED_DIM (default 256)

Outputs compatible with Stage 2 channel aggregator (drop-in for ViT-MAE encoder).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models

EMBED_DIM  = 256   # patch and CLS embedding dimension
IMG_SIZE   = 224
PATCH_SIZE = 8     # 28×28 = 784 patches (layer2 stride = 8)


class ResNetSpatialEncoder(nn.Module):
    """
    ResNet-18 backbone that returns spatial patch tokens + CLS token.

    layer2 (stride-8) output gives a 28×28 feature map — one feature vector
    per 8×8 pixel patch, preserving spatial structure via CNN locality.
    layer4 global-avg-pool gives a single vector summarising the whole image.
    Both are projected to EMBED_DIM via a linear layer.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        base = tv_models.resnet18(weights=None)

        # single-channel input (ion images are grayscale)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # stride-4   → 56×56 × 64
        self.layer2 = base.layer2   # stride-8   → 28×28 × 128  ← patch tokens
        self.layer3 = base.layer3   # stride-16  → 14×14 × 256
        self.layer4 = base.layer4   # stride-32  →  7×7  × 512
        self.gap    = nn.AdaptiveAvgPool2d(1)

        self.patch_proj = nn.Linear(128, embed_dim)
        self.cls_proj   = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 1, H, W) float tensor

        Returns
        -------
        cls_token    : (B, embed_dim)
        patch_tokens : (B, 784, embed_dim)  — 28×28 spatial grid
        """
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)          # (B, 128, 28, 28)
        x = self.layer3(f2)
        x = self.layer4(x)           # (B, 512, 7, 7)

        # CLS: global average pool
        cls = self.gap(x).flatten(1)                     # (B, 512)
        cls = self.cls_proj(cls)                          # (B, D)

        # Patch tokens: reshape spatial map → sequence
        B, C, H, W = f2.shape
        patches = f2.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 784, 128)
        patches = self.patch_proj(patches)                        # (B, 784, D)

        return cls, patches


# ── Barlow Twins projector (used only during pretraining) ────────────────────

class BarlowProjector(nn.Module):
    """3-layer MLP projector head for Barlow Twins (discarded after pretraining)."""

    def __init__(self, in_dim: int = EMBED_DIM, proj_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor,
                      lambda_off: float = 5e-3) -> torch.Tensor:
    """
    Barlow Twins loss on two projected views z1, z2 (B, D).
    Cross-correlation matrix should be identity: on-diagonal → 1, off-diagonal → 0.
    """
    B, D = z1.shape
    # normalise along batch dimension
    z1n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c   = (z1n.T @ z2n) / B                    # (D, D) cross-correlation matrix
    on_diag  = (c.diagonal() - 1).pow(2).sum()
    off_diag = (c.flatten()[1:].view(D-1, D+1)[:, :-1]).pow(2).sum()
    return on_diag + lambda_off * off_diag


def spatial_coherence_loss(patch_tokens: torch.Tensor,
                            n_grid: int = 28) -> torch.Tensor:
    """
    Spatial coherence auxiliary loss.
    Penalises low cosine similarity between horizontally/vertically adjacent patches.
    Encourages the CNN features to vary smoothly across nearby tissue regions.

    patch_tokens : (B, N, D)  N = n_grid^2
    """
    import torch.nn.functional as F
    B, N, D = patch_tokens.shape
    tok = F.normalize(patch_tokens, dim=-1)
    grid = tok.view(B, n_grid, n_grid, D)

    sim_h = (grid[:, :, :-1, :] * grid[:, :, 1:, :]).sum(-1)   # (B, 28, 27)
    sim_v = (grid[:, :-1, :, :] * grid[:, 1:, :, :]).sum(-1)   # (B, 27, 28)

    return -(sim_h.mean() + sim_v.mean()) / 2   # minimise → maximise similarity


# ── Inference helpers (mirror ion_encoder.py API) ────────────────────────────

@dataclass
class IonEncoderOutputs:
    cls_tokens:   torch.Tensor   # (B, D)
    patch_tokens: torch.Tensor   # (B, 784, D)


@torch.no_grad()
def encode_ions(model: ResNetSpatialEncoder,
                pixel_values: torch.Tensor) -> IonEncoderOutputs:
    """Run encoder, return CLS + patch tokens. No masking at inference."""
    model.eval()
    cls, patches = model(pixel_values)
    return IonEncoderOutputs(cls_tokens=cls, patch_tokens=patches)


def build_ion_encoder_for_pretraining() -> ResNetSpatialEncoder:
    return ResNetSpatialEncoder(embed_dim=EMBED_DIM)


def build_ion_encoder_for_inference(checkpoint_path: str) -> ResNetSpatialEncoder:
    model = ResNetSpatialEncoder(embed_dim=EMBED_DIM)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["encoder"])
    return model
