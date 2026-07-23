"""
pretrain_stage2.py
------------------
Stage 2 pretraining: Channel Aggregator (masked channel prediction).

Loads pre-extracted Stage 1 CLS embeddings (from extract_resnet_embeddings.py)
directly from disk — no image loading or Stage 1 inference during training.

For each MSI sample in a batch:
  1. Load pre-extracted CLS tokens for all channels (C, 256)
  2. Randomly mask ~40% of channel tokens (zero them out)
  3. Run Stage 2 Transformer on the masked token sequence
  4. Predict the Stage 1 CLS embedding of each masked channel
  5. Loss: cosine embedding loss on masked positions

Outputs (under BASE_DIR/run_<timestamp>/)
-----------------------------------------
  config.json
  training_log.jsonl
  checkpoints/step_XXXXXXX.pt
  stage2_aggregator_best.pt
  stage2_aggregator_final.pt

Usage
-----
  python pretrain_stage2.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.channel_aggregator import (
    build_channel_aggregator,
    masked_channel_loss,
    MASK_RATIO,
    STAGE1_DIM,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
CLS_NPY  = EMB_DIR / "resnet_cls_embeddings.npy"
META_CSV = EMB_DIR / "resnet_cls_meta.csv"

BASE_DIR = METABOFM_ROOT / "metabofm_v2/stage2_resnet"

MAX_CHANNELS     = 32
BATCH_SIZE       = 64     # larger batch is fine — no image loading overhead
NUM_EPOCHS       = 50
LEARNING_RATE    = 5e-5
WARMUP_STEPS     = 500
WEIGHT_DECAY     = 0.05
GRAD_CLIP        = 1.0
SAVE_EVERY       = 2000
LOG_EVERY        = 50
USE_MZ_EMBEDDING = True

SMOKE_TEST = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── DATASET ───────────────────────────────────────────────────────────────────

class SampleEmbeddingDataset(Dataset):
    """
    One item = one MSI sample (all its channels).
    Reads directly from the pre-extracted CLS memmap — no image I/O.

    Returns:
      cls_tokens   : (MAX_CHANNELS, STAGE1_DIM)  float32, zero-padded
      mz           : (MAX_CHANNELS,)              float32, zero-padded
      channel_mask : (MAX_CHANNELS,)              bool, True = real channel
    """

    def __init__(self, meta_df: pd.DataFrame, cls_array: np.ndarray,
                 max_channels: int = MAX_CHANNELS):
        self.cls   = cls_array
        self.max_c = max_channels

        self.samples: list[tuple[list[int], list[float]]] = []
        for _, grp in meta_df.groupby("sample_path", sort=False):
            self.samples.append((grp.index.tolist(), grp["mz"].tolist()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row_ids, mz_vals = self.samples[idx]
        C = min(len(row_ids), self.max_c)

        cls_out  = np.zeros((self.max_c, STAGE1_DIM), dtype=np.float32)
        mz_out   = np.zeros(self.max_c,               dtype=np.float32)
        mask_out = np.zeros(self.max_c,               dtype=bool)

        cls_out[:C]  = self.cls[row_ids[:C]]
        mz_out[:C]   = mz_vals[:C]
        mask_out[:C] = True

        return {
            "cls_tokens":   torch.from_numpy(cls_out),
            "mz":           torch.from_numpy(mz_out),
            "channel_mask": torch.from_numpy(mask_out),
        }


# ── HELPERS ───────────────────────────────────────────────────────────────────

def log_step(log_file: Path, record: dict) -> None:
    record["utc"] = datetime.datetime.utcnow().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def random_channel_mask(B: int, C: int, mask_ratio: float,
                        device: torch.device) -> torch.Tensor:
    return torch.rand(B, C, device=device) < mask_ratio


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    for p in (CLS_NPY, META_CSV):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\nRun extract_resnet_embeddings.py first."
            )

    run_tag    = "smoke" if SMOKE_TEST else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_DIR / f"run_{run_tag}"
    log_file   = output_dir / "training_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Run dir  : {output_dir}")
    print(f"[INFO] Device   : {DEVICE}")

    print("[LOAD] Pre-extracted CLS embeddings ...")
    cls_array = np.load(str(CLS_NPY), mmap_mode="r")
    meta_df   = pd.read_csv(META_CSV)
    print(f"  shape={cls_array.shape}  samples={meta_df['sample_path'].nunique():,}")

    if SMOKE_TEST:
        uniq    = meta_df["sample_path"].unique()[:50]
        meta_df = meta_df[meta_df["sample_path"].isin(uniq)].reset_index(drop=True)
        print(f"[SMOKE] {meta_df['sample_path'].nunique()} samples, 3 epochs")

    dataset  = SampleEmbeddingDataset(meta_df, cls_array, max_channels=MAX_CHANNELS)
    n_epochs = 3 if SMOKE_TEST else NUM_EPOCHS

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
        persistent_workers=True,
    )
    print(f"[INFO] Samples  : {len(dataset):,}  batch={BATCH_SIZE}  "
          f"steps/epoch={len(loader)}  epochs={n_epochs}")

    stage2  = build_channel_aggregator(use_mz_embedding=USE_MZ_EMBEDDING).to(DEVICE)
    n_param = sum(p.numel() for p in stage2.parameters()) / 1e6
    print(f"[INFO] Stage 2  : {n_param:.1f}M parameters")

    cfg = {
        "cls_npy":          str(CLS_NPY),
        "meta_csv":         str(META_CSV),
        "stage1_dim":       STAGE1_DIM,
        "max_channels":     MAX_CHANNELS,
        "batch_size":       BATCH_SIZE,
        "num_epochs":       n_epochs,
        "learning_rate":    LEARNING_RATE,
        "warmup_steps":     WARMUP_STEPS,
        "weight_decay":     WEIGHT_DECAY,
        "grad_clip":        GRAD_CLIP,
        "mask_ratio":       float(MASK_RATIO),
        "use_mz_embedding": USE_MZ_EMBEDDING,
        "n_samples":        len(dataset),
        "smoke_test":       SMOKE_TEST,
        "started_utc":      datetime.datetime.utcnow().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    optimizer   = torch.optim.AdamW(stage2.parameters(),
                                    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * n_epochs
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        total_steps=max(total_steps, 1),
        pct_start=min(WARMUP_STEPS / max(total_steps, 1), 0.3),
        anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, n_epochs + 1):
        stage2.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}"):
            cls_tokens = batch["cls_tokens"].to(DEVICE)    # (B, C, 256)
            mz         = batch["mz"].to(DEVICE)            # (B, C)
            cmask      = batch["channel_mask"].to(DEVICE)  # (B, C) bool
            B, C, _    = cls_tokens.shape

            mask      = random_channel_mask(B, C, MASK_RATIO, DEVICE) & cmask
            cls_input = cls_tokens.clone()
            cls_input[mask] = 0.0

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out   = stage2(
                    cls_tokens=cls_input,
                    mz=mz if USE_MZ_EMBEDDING else None,
                    channel_mask=cmask,
                )
                preds = stage2.predict_masked(out["channel_refined"], mask)
                loss  = masked_channel_loss(preds, cls_tokens, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(stage2.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step:6d}  loss={loss.item():.4f}  lr={lr:.2e}")
                log_step(log_file, {
                    "step":     global_step,
                    "epoch":    epoch,
                    "loss":     round(loss.item(), 6),
                    "lr":       lr,
                    "n_masked": int(mask.sum().item()),
                })

            if not SMOKE_TEST and global_step % SAVE_EVERY == 0:
                ckpt = output_dir / "checkpoints" / f"step_{global_step:07d}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(stage2.state_dict(), ckpt)
                print(f"  [CKPT] {ckpt.name}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}")
        log_step(log_file, {"epoch_end": epoch, "avg_loss": round(avg_loss, 6),
                             "global_step": global_step})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(stage2.state_dict(), output_dir / "stage2_aggregator_best.pt")

    torch.save(stage2.state_dict(), output_dir / "stage2_aggregator_final.pt")
    log_step(log_file, {"training_complete": True, "best_avg_loss": round(best_loss, 6),
                         "total_steps": global_step})
    print(f"\n[DONE] Best avg loss: {best_loss:.4f}")
    print(f"[DONE] Outputs: {output_dir}")


if __name__ == "__main__":
    main()
