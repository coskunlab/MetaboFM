"""
models/channel_aggregator.py
-----------------------------
Stage 2: Channel Aggregator.

A permutation-invariant Transformer that takes the set of Stage 1 CLS tokens
for all channels in a tile and produces:
  - Aggregator CLS token  (512,)    → sample-level embedding
  - Refined channel tokens (C, 512) → cross-channel-aware channel embeddings

Permutation invariance is achieved by:
  - Using no fixed positional encoding by default.
  - Optionally injecting m/z as a continuous positional signal via a small
    learned MLP (soft ordering by molecular mass, compatible with variable
    and non-uniform m/z axes).

Pretraining objective: masked channel prediction.
  - Randomly mask ~40% of channel tokens.
  - Predict the masked CLS embeddings from context using a projection head.
  - Loss: cosine embedding loss on masked positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

STAGE1_DIM   = 256    # Stage 1 CLS token dimension (ResNet-18 + Barlow Twins)
AGG_DIM      = 512    # Channel Aggregator hidden dimension
AGG_HEADS    = 8
AGG_LAYERS   = 4
AGG_FF_DIM   = 2048
AGG_DROPOUT  = 0.1
MASK_RATIO   = 0.40   # fraction of channels masked during pretraining
MZ_EMB_DIM   = 64     # dimension of m/z positional embedding (0 = disabled)


# ============================================================
# M/Z POSITIONAL EMBEDDING
# ============================================================

class MzPositionalEmbedding(nn.Module):
    """
    Maps a scalar m/z value to a continuous embedding via a small MLP.
    Handles variable and non-uniform m/z axes — no discretization needed.

    Input:  (B, C) float tensor of m/z values (0.0 for unknown channels)
    Output: (B, C, mz_emb_dim) positional embeddings
    """

    def __init__(self, emb_dim: int = MZ_EMB_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        # mz: (B, C)
        return self.net(mz.unsqueeze(-1))   # (B, C, emb_dim)


# ============================================================
# CHANNEL AGGREGATOR
# ============================================================

class ChannelAggregator(nn.Module):
    """
    Permutation-invariant Transformer over ion CLS tokens.

    Parameters
    ----------
    stage1_dim : int
        Dimension of incoming Stage 1 CLS tokens (256 for ResNet-18 + Barlow Twins).
    agg_dim : int
        Internal hidden dimension of the aggregator.
    n_heads, n_layers, ff_dim, dropout : Transformer hyperparams.
    use_mz_embedding : bool
        If True, add a learned m/z positional embedding to each channel token.
        If False, the model is fully permutation-invariant.
    mz_emb_dim : int
        Dimension of the m/z embedding (only used when use_mz_embedding=True).
    """

    def __init__(
        self,
        stage1_dim:       int  = STAGE1_DIM,
        agg_dim:          int  = AGG_DIM,
        n_heads:          int  = AGG_HEADS,
        n_layers:         int  = AGG_LAYERS,
        ff_dim:           int  = AGG_FF_DIM,
        dropout:          float = AGG_DROPOUT,
        use_mz_embedding: bool  = True,
        mz_emb_dim:       int  = MZ_EMB_DIM,
    ):
        super().__init__()
        self.agg_dim = agg_dim
        self.use_mz_embedding = use_mz_embedding

        # Project Stage 1 CLS tokens into aggregator dimension
        in_dim = stage1_dim + (mz_emb_dim if use_mz_embedding else 0)
        self.input_proj = nn.Linear(in_dim, agg_dim)

        # Learnable [AGG_CLS] token — like BERT's [CLS]
        self.agg_cls_token = nn.Parameter(torch.zeros(1, 1, agg_dim))
        nn.init.trunc_normal_(self.agg_cls_token, std=0.02)

        # Optional m/z positional embedding
        if use_mz_embedding:
            self.mz_emb = MzPositionalEmbedding(emb_dim=mz_emb_dim)
        else:
            self.mz_emb = None

        # Transformer encoder (no fixed positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=agg_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(agg_dim),
        )

        # Output projection for refined channel embeddings
        self.channel_proj = nn.Linear(agg_dim, agg_dim)
        # Output projection for sample CLS
        self.sample_proj  = nn.Linear(agg_dim, agg_dim)

        # Masked channel prediction head (used during pretraining only)
        self.mask_pred_head = nn.Sequential(
            nn.Linear(agg_dim, agg_dim),
            nn.GELU(),
            nn.LayerNorm(agg_dim),
            nn.Linear(agg_dim, stage1_dim),  # predict Stage 1 CLS embedding
        )

    def forward(
        self,
        cls_tokens:    torch.Tensor,         # (B, C, stage1_dim)
        mz:            torch.Tensor | None,  # (B, C) float, optional
        channel_mask:  torch.Tensor | None = None,  # (B, C) bool, True=real
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        cls_tokens : (B, C, stage1_dim)
            Stage 1 CLS tokens for each channel in the tile.
        mz : (B, C) or None
            m/z values for positional embedding. 0.0 for unknown/padded.
        channel_mask : (B, C) bool or None
            True for real channels, False for padding.
            Used as key_padding_mask in the Transformer.

        Returns
        -------
        dict with:
          "sample_cls"      : (B, agg_dim) — sample-level embedding
          "channel_refined" : (B, C, agg_dim) — per-channel embeddings
        """
        B, C, _ = cls_tokens.shape

        # Optionally add m/z positional embedding
        if self.use_mz_embedding and mz is not None and self.mz_emb is not None:
            mz_emb = self.mz_emb(mz)          # (B, C, mz_emb_dim)
            tokens = torch.cat([cls_tokens, mz_emb], dim=-1)   # (B, C, stage1+mz)
        else:
            tokens = cls_tokens

        tokens = self.input_proj(tokens)       # (B, C, agg_dim)

        # Prepend learnable [AGG_CLS] token
        agg_cls = self.agg_cls_token.expand(B, -1, -1)   # (B, 1, agg_dim)
        tokens  = torch.cat([agg_cls, tokens], dim=1)    # (B, C+1, agg_dim)

        # Build key_padding_mask: True positions are IGNORED in attention
        # Note: PyTorch TransformerEncoder uses True = ignore convention
        if channel_mask is not None:
            # prepend True for the AGG_CLS token (always attend to it)
            cls_col = torch.ones(B, 1, dtype=torch.bool, device=cls_tokens.device)
            full_mask = torch.cat([cls_col, channel_mask], dim=1)  # (B, C+1)
            key_pad   = ~full_mask   # True = padding (ignore)
        else:
            key_pad = None

        out = self.transformer(tokens, src_key_padding_mask=key_pad)  # (B, C+1, agg_dim)

        sample_cls      = self.sample_proj(out[:, 0, :])    # (B, agg_dim)
        channel_refined = self.channel_proj(out[:, 1:, :])  # (B, C, agg_dim)

        return {
            "sample_cls":      sample_cls,
            "channel_refined": channel_refined,
        }

    def predict_masked(
        self,
        refined_tokens: torch.Tensor,   # (B, C, agg_dim)
        mask:           torch.Tensor,   # (B, C) bool, True = masked position
    ) -> torch.Tensor:
        """
        Project masked positions to Stage 1 CLS space for the pretraining loss.
        Returns (B, C, stage1_dim) — only masked positions are meaningful.
        """
        return self.mask_pred_head(refined_tokens)   # (B, C, stage1_dim)


# ============================================================
# MASKED CHANNEL PRETRAINING LOSS
# ============================================================

def masked_channel_loss(
    predictions:  torch.Tensor,   # (B, C, stage1_dim) — from predict_masked
    targets:      torch.Tensor,   # (B, C, stage1_dim) — original Stage 1 CLS
    mask:         torch.Tensor,   # (B, C) bool, True = masked
) -> torch.Tensor:
    """
    Cosine embedding loss on masked channel positions.
    Encourages predicted embeddings to align with the original Stage 1 CLS.
    """
    B, C, D = predictions.shape
    mask_flat = mask.reshape(-1)                               # (B*C,)
    pred_flat = predictions.reshape(B * C, D)[mask_flat]      # (M, D)
    tgt_flat  = targets.reshape(B * C, D)[mask_flat]          # (M, D)

    if pred_flat.shape[0] == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    # Cosine embedding loss: target label=1 → maximise similarity
    labels = torch.ones(pred_flat.shape[0], device=predictions.device)
    return F.cosine_embedding_loss(pred_flat, tgt_flat, labels)


# ============================================================
# FACTORY
# ============================================================

def build_channel_aggregator(use_mz_embedding: bool = True) -> ChannelAggregator:
    return ChannelAggregator(use_mz_embedding=use_mz_embedding)


def load_channel_aggregator(checkpoint_path: str) -> ChannelAggregator:
    model = ChannelAggregator()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model
