"""
pretrain_stage1.py
------------------
Stage 1 pretraining: ResNet-18 spatial encoder with Barlow Twins.

Training objective
------------------
  L = L_barlow  +  alpha * L_spatial_coherence

  L_barlow            : Barlow Twins on image-level CLS tokens from two
                        augmented views of the same ion channel image.
                        Forces the encoder to produce consistent representations
                        across augmentations (flip, crop, intensity jitter).

  L_spatial_coherence : Cosine similarity between adjacent patch tokens
                        should be high. Directly supervises spatial structure
                        in the feature map â€” what ViT-MAE failed to learn.

Usage
-----
  python pretrain_stage1_resnet.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from dataset import IonImageDataset, set_data_root
from models.resnet_encoder import (
    ResNetSpatialEncoder, BarlowProjector,
    barlow_twins_loss, spatial_coherence_loss,
    EMBED_DIM,
)

CSV_PATH   = str(METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv")
DATA_ROOT  = str(MSI_RAW_DIR)
BASE_DIR   = METABOFM_ROOT / "metabofm_v2/stage1_resnet"

IMG_SIZE      = 224
BATCH_SIZE    = 128    # Barlow Twins benefits from large batches; RTX 4090 fits 128 at ResNet-18
NUM_EPOCHS    = 200
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 1
ALPHA_SPATIAL = 0.1   # weight for spatial coherence auxiliary loss
ALPHA_PATCH   = 0.1   # weight for patch-level Barlow Twins (only used if PATCH_CONTRASTIVE=True)
PROJ_DIM      = 2048
PATCH_PROJ_DIM = 512  # smaller projector for patches (B*784 samples >> B, so lower dim is fine)
PATCH_CONTRASTIVE = True  # apply Barlow Twins on patch tokens in addition to CLS tokens
SAVE_EVERY    = 5     # save checkpoint every N epochs
LOG_EVERY     = 50    # steps

SMOKE_TEST  = False
RESUME_FROM = None   # set to a checkpoint path to resume, e.g. r"Y:\...\checkpoint_epoch002.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

class IonAugmentation:
    """
    Two-view augmentation for Barlow Twins on single-channel ion images.
    Avoids colour jitter / grayscale (already grayscale).
    Uses intensity jitter instead.
    """
    def __init__(self, img_size: int = IMG_SIZE):
        self.img_size = img_size

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (1, H, W) float tensor â†’ two augmented views."""
        return self._augment(x), self._augment(x)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = TF.hflip(x)
        # random vertical flip
        if torch.rand(1).item() > 0.5:
            x = TF.vflip(x)
        # random resized crop (80-100% of image)
        scale = 0.8 + 0.2 * torch.rand(1).item()
        crop_h = int(x.shape[1] * scale)
        crop_w = int(x.shape[2] * scale)
        i = torch.randint(0, x.shape[1] - crop_h + 1, (1,)).item()
        j = torch.randint(0, x.shape[2] - crop_w + 1, (1,)).item()
        x = x[:, i:i+crop_h, j:j+crop_w]
        x = TF.resize(x, [self.img_size, self.img_size],
                       interpolation=TF.InterpolationMode.NEAREST)
        # intensity jitter (scale 0.6â€“1.4, clamp to [0,1])
        scale = 0.6 + 0.8 * torch.rand(1).item()
        x = (x * scale).clamp(0.0, 1.0)
        # optional Gaussian blur
        if torch.rand(1).item() > 0.5:
            sigma = 0.1 + 1.9 * torch.rand(1).item()
            x = TF.gaussian_blur(x, kernel_size=5, sigma=sigma)
        return x


class TwoViewDataset(Dataset):
    """Wraps IonImageDataset to return two augmented views per sample."""
    def __init__(self, base_ds: IonImageDataset):
        self.base_ds  = base_ds
        self.augment  = IonAugmentation()

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict:
        item  = self.base_ds[idx]
        pv    = item["pixel_values"]   # (1, H, W)
        v1, v2 = self.augment(pv)
        return {"view1": v1, "view2": v2}


def two_view_collator(batch):
    v1 = torch.stack([b["view1"] for b in batch])
    v2 = torch.stack([b["view2"] for b in batch])
    return {"view1": v1, "view2": v2}

def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def main():
    # â”€â”€ Resume or fresh start â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    resume_ckpt = Path(RESUME_FROM) if RESUME_FROM else None
    if resume_ckpt:
        # reuse the existing run directory (parent of the checkpoint)
        output_dir = resume_ckpt.parent
        run_tag    = output_dir.name.removeprefix("run_")
        print(f"[INFO] Resuming from {resume_ckpt.name}  â†’  {output_dir}")
    else:
        run_tag    = "smoke" if SMOKE_TEST else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = BASE_DIR / f"run_{run_tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Run dir: {output_dir}")
    log_path = output_dir / "training_log.jsonl"

    set_data_root(DATA_ROOT)

    # â”€â”€ Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    if SMOKE_TEST:
        df = df.sample(n=500, random_state=42).reset_index(drop=True)

    base_ds = IonImageDataset(df=df, base_dir=DATA_ROOT, img_size=IMG_SIZE,
                              norm="tile_max", augment=False)  # augmentation done in TwoViewDataset
    train_ds = TwoViewDataset(base_ds)
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0 if SMOKE_TEST else 4,
                          pin_memory=(DEVICE == "cuda"),
                          persistent_workers=False if SMOKE_TEST else True,
                          drop_last=True, collate_fn=two_view_collator)

    print(f"[INFO] Dataset: {len(train_ds)} channel images  |  {len(loader)} steps/epoch")
    print(f"[INFO] Device: {DEVICE}  |  batch: {BATCH_SIZE}  |  epochs: {NUM_EPOCHS}")

    # â”€â”€ Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    encoder        = ResNetSpatialEncoder(embed_dim=EMBED_DIM).to(DEVICE)
    projector      = BarlowProjector(in_dim=EMBED_DIM, proj_dim=PROJ_DIM).to(DEVICE)
    patch_projector = BarlowProjector(in_dim=EMBED_DIM, proj_dim=PATCH_PROJ_DIM).to(DEVICE) \
                      if PATCH_CONTRASTIVE else None
    n_enc  = sum(p.numel() for p in encoder.parameters()) / 1e6
    n_proj = sum(p.numel() for p in projector.parameters()) / 1e6
    n_ppj  = sum(p.numel() for p in patch_projector.parameters()) / 1e6 if patch_projector else 0
    print(f"[INFO] Encoder: {n_enc:.1f}M  |  CLS projector: {n_proj:.1f}M  |  Patch projector: {n_ppj:.1f}M")

    # â”€â”€ Optimiser + scheduler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    params = list(encoder.parameters()) + list(projector.parameters()) + \
             (list(patch_projector.parameters()) if patch_projector else [])
    opt    = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps   = NUM_EPOCHS * len(loader)
    warmup_steps  = WARMUP_EPOCHS * len(loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))   # cosine decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # â”€â”€ Save config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    config = {
        "csv_path": CSV_PATH, "data_root": DATA_ROOT,
        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS, "alpha_spatial": ALPHA_SPATIAL,
        "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM,
        "patch_contrastive": PATCH_CONTRASTIVE, "patch_proj_dim": PATCH_PROJ_DIM,
        "alpha_patch": ALPHA_PATCH,
        "n_samples": len(train_ds), "smoke_test": SMOKE_TEST,
        "started_utc": datetime.datetime.utcnow().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # â”€â”€ Resume: load weights + optimizer + scheduler state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None
    global_step  = 0
    start_epoch  = 1

    if resume_ckpt:
        ckpt = torch.load(str(resume_ckpt), map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        projector.load_state_dict(ckpt["projector"])
        if patch_projector and "patch_projector" in ckpt:
            patch_projector.load_state_dict(ckpt["patch_projector"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt["global_step"]
        for _ in range(global_step):
            scheduler.step()
        print(f"[INFO] Resumed: epoch {ckpt['epoch']} done, continuing from epoch {start_epoch}  (global_step={global_step})")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        encoder.train(); projector.train()
        epoch_losses = []

        if patch_projector:
            patch_projector.train()

        for batch in loader:
            v1 = batch["view1"].to(DEVICE)
            v2 = batch["view2"].to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                cls1, pat1 = encoder(v1)
                cls2, pat2 = encoder(v2)

                z1 = projector(cls1)
                z2 = projector(cls2)

                l_bt = barlow_twins_loss(z1, z2)
                l_sp = spatial_coherence_loss(pat1) + spatial_coherence_loss(pat2)
                loss = l_bt + ALPHA_SPATIAL * l_sp

                l_pbt = torch.tensor(0.0, device=DEVICE)

            if patch_projector:
                # run in float32 outside autocast â€” (B*784, 512) matmuls overflow fp16
                B, N, D = pat1.shape
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    zp1 = patch_projector(pat1.detach().float().reshape(B * N, D))
                    zp2 = patch_projector(pat2.detach().float().reshape(B * N, D))
                    l_pbt = barlow_twins_loss(zp1, zp2) / PATCH_PROJ_DIM
                loss = loss + ALPHA_PATCH * l_pbt

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, max_norm=3.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=3.0)
                opt.step()

            scheduler.step()
            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % LOG_EVERY == 0 or global_step == 1:
                record = {
                    "utc": datetime.datetime.utcnow().isoformat(),
                    "epoch": epoch, "step": global_step,
                    "loss": round(loss.item(), 4),
                    "loss_barlow": round(l_bt.item(), 4),
                    "loss_spatial": round(l_sp.item(), 4),
                    "loss_patch_bt": round(l_pbt.item(), 4),
                    "lr": round(get_lr(opt), 7),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"  epoch {epoch:2d}  step {global_step:5d}  "
                      f"loss {loss.item():.4f}  bt {l_bt.item():.4f}  "
                      f"sp {l_sp.item():.4f}  pbt {l_pbt.item():.4f}  "
                      f"lr {get_lr(opt):.2e}", flush=True)

        print(f"[Epoch {epoch}] mean loss: {np.mean(epoch_losses):.4f}")

        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch:03d}.pt"
            ckpt_data = {
                "epoch": epoch,
                "global_step": global_step,
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": opt.state_dict(),
            }
            if patch_projector:
                ckpt_data["patch_projector"] = patch_projector.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  [SAVED] {ckpt_path.name}")

    # â”€â”€ Save final encoder only (projector discarded after pretraining) â”€â”€â”€â”€â”€â”€â”€
    final_path = output_dir / "encoder_final.pt"
    torch.save({"encoder": encoder.state_dict(), "embed_dim": EMBED_DIM}, final_path)
    print(f"\n[OK] Final encoder saved: {final_path}")
    print(f"[OK] Run dir: {output_dir}")


if __name__ == "__main__":
    main()

