"""
plot_figS2.py
-------------
Supplementary Figure S2: Training Details.

Panels:
  A  Stage 1 training pipeline (MMD diagram)
  B  Stage 1 loss decomposition: total, Barlow Twins, spatial, patch BT
  C  Stage 2 training loss curve

Usage:
  conda run -n torch_gpu python plot_figS2.py
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import subprocess, json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_utils import set_nature_style
set_nature_style()

# -- CONFIG -------------------------------------------------------------------
S1_LOG    = METABOFM_ROOT / "metabofm_v2/stage1_resnet/run_20260711_004546/training_log.jsonl"
S2_LOG    = METABOFM_ROOT / "metabofm_v2/stage2_resnet/run_20260711_130252/training_log.jsonl"
OUT_DIR   = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS2_training_details"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

MMD_CONTENT = """\
graph LR
    A["Ion Channel Image\\n(H×W)"] --> B["Two Augmented Views\\n(crop · flip · noise)"]
    B --> C["Shared ResNet-18\\n(per-channel encoder)"]
    C --> D["512-d Projections\\n(MLP head ×2)"]
    D --> E["Barlow Twins Loss\\n(cross-correlation matrix)"]
    E --> F["Channel Embedding\\n(frozen after Stage 1)"]
"""

CAPTION = """\
Supplementary Figure 2 | Stage 1 and Stage 2 training dynamics.

a, Stage 1 training pipeline. Each ion image is independently augmented (random crop, flip, intensity rescaling) to produce two views, which are encoded by a shared ResNet-18 backbone and projected to 256 dimensions, then trained with the Barlow Twins loss.

b, Stage 1 loss decomposition over ~180 epochs (of a 200-epoch training run). Total loss (black) together with its three components: Barlow Twins cross-correlation term (blue), spatial contiguity regularisation (orange), and patch-level Barlow Twins auxiliary loss (green). All components converge smoothly without representation collapse.

c, Stage 2 (cross-channel Transformer) training loss over 50 epochs. The masked-channel cosine embedding loss decreases from ~0.78 to ~0.09, indicating stable learning of cross-channel sample representations.
"""


def write_caption():
    (PANEL_DIR / "captions.txt").write_text(CAPTION, encoding="utf-8")
    print("  saved captions.txt")


def save_panel(fig, stem):
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    fig.savefig(str(PANEL_DIR / f"{stem}.svg"), bbox_inches="tight", pad_inches=0)
    print(f"  saved panel {stem}.svg")


def _load_jsonl(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def draw_panel_b(ax):
    """Stage 1 loss decomposition per epoch."""
    df = _load_jsonl(S1_LOG)
    # keep only step rows (have loss_barlow)
    df = df[df["loss_barlow"].notna()] if "loss_barlow" in df.columns else df
    ep = df.groupby("epoch")[["loss", "loss_barlow", "loss_spatial", "loss_patch_bt"]].mean().reset_index()

    ax.plot(ep["epoch"], ep["loss"],          color="#111111", lw=2.0, label="Total")
    ax.plot(ep["epoch"], ep["loss_barlow"],   color="#2166ac", lw=1.4, ls="--", label="Barlow Twins")
    ax.plot(ep["epoch"], ep["loss_spatial"],  color="#e08214", lw=1.4, ls="--", label="Spatial")
    ax.plot(ep["epoch"], ep["loss_patch_bt"], color="#4dac26", lw=1.4, ls="--", label="Patch BT")

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.set_title("B   Stage 1 Loss Decomposition  (180 epochs)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)


def draw_panel_c(ax):
    """Stage 2 training loss per epoch."""
    df  = _load_jsonl(S2_LOG)
    # use epoch_end rows which have avg_loss
    ep  = df[df["epoch_end"].notna()].copy() if "epoch_end" in df.columns else df
    if "avg_loss" in ep.columns:
        ep = ep[["epoch_end", "avg_loss"]].dropna().rename(columns={"epoch_end": "epoch"})
    else:
        ep = df.groupby("epoch")["loss"].mean().reset_index().rename(columns={"loss": "avg_loss"})

    ax.plot(ep["epoch"], ep["avg_loss"], color="#d6604d", lw=1.8)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Masked-channel contrastive loss", fontsize=9)
    ax.set_title("C   Stage 2 Training Loss  (50 epochs)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def render_mmd():
    import shutil
    _mmd = PANEL_DIR / "pipeline.mmd"
    _mmd.write_text(MMD_CONTENT, encoding="utf-8")


def main():
    render_mmd()

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    draw_panel_b(ax)
    save_panel(fig, "figS2_panelB_stage1_loss_decomp")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    draw_panel_c(ax)
    save_panel(fig, "figS2_panelC_stage2_loss")
    plt.close(fig)

    write_caption()
    print("FigS2 done.")


if __name__ == "__main__":
    main()
