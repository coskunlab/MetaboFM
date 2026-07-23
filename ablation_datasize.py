"""
ablation_datasize.py
--------------------
Training-data-size ablation for Stage 2 (ChannelAggregator).

For each fraction in FRACTIONS:
  1. Subsample that fraction of MSI training samples (seeded, stratified by organ)
  2. Train Stage 2 from scratch on the subset (Stage 1 weights fixed)
  3. Extract Stage 2 sample CLS embeddings for ALL 5,600 samples
  4. Run leave-one-study-out organ retrieval (macro R@1)

Outputs
-------
  OUT_DIR/ablation_datasize_results.csv   — fraction, macro_R1, weighted_R1
  OUT_DIR/ablation_datasize_ckpts/        — one checkpoint per fraction

Usage
-----
  conda run -n torch_gpu python ablation_datasize.py
"""

from __future__ import annotations

import json
import sys
import datetime
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "code_v2"))
sys.path.insert(0, str(Path(__file__).parent))

# ── PATHS ──────────────────────────────────────────────────────────────────────
EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
CLS_NPY  = EMB_DIR / "resnet_cls_embeddings.npy"
META_CSV = EMB_DIR / "resnet_cls_meta.csv"
CHAN_META = EMB_DIR / "stage2_channel_meta.csv"

OUT_DIR  = METABOFM_ROOT / "outputs/ablation_datasize"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "ckpts"
CKPT_DIR.mkdir(exist_ok=True)

# ── ABLATION CONFIG ────────────────────────────────────────────────────────────
FRACTIONS    = [0.25, 0.50, 0.75, 1.00]
SEED         = 42
NUM_EPOCHS   = 30          # fewer epochs than full training (50) — ablation speed
BATCH_SIZE   = 64
LR           = 5e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.05
GRAD_CLIP    = 1.0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# retrieval eval
KS           = [1, 5, 10]
MIN_ORGAN_N  = 10
MIN_DATASETS = 2

# ── IMPORTS FROM CODEBASE ──────────────────────────────────────────────────────
from models.channel_aggregator import (
    build_channel_aggregator,
    masked_channel_loss,
    MASK_RATIO,
    STAGE1_DIM,
)
from pretrain_stage2 import SampleEmbeddingDataset, MAX_CHANNELS, USE_MZ_EMBEDDING


# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_full_meta():
    """Load channel-level meta with organ/organism/dataset_id for eval."""
    ch = pd.read_csv(CHAN_META,
                     usecols=["sample_path", "Organism_Part", "organism", "dataset_id"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    samp["organ"] = samp["Organism_Part"].replace({"Kideny": "Kidney", "colon": "Colon"})
    return samp


def subsample_train_meta(meta_df: pd.DataFrame, fraction: float,
                         seed: int = SEED) -> pd.DataFrame:
    """
    Return a channel-level meta_df containing only `fraction` of MSI samples.
    Stratified by organ so all organs remain represented.
    """
    if fraction >= 1.0:
        return meta_df

    samp_meta = pd.read_csv(CHAN_META,
                             usecols=["sample_path", "Organism_Part"])
    samp_meta["organ"] = samp_meta["Organism_Part"].replace(
        {"Kideny": "Kidney", "colon": "Colon"})
    unique_samps = samp_meta.drop_duplicates("sample_path")[["sample_path", "organ"]]

    rng = np.random.default_rng(seed)
    chosen = []
    for organ, grp in unique_samps.groupby("organ"):
        n_keep = max(1, round(len(grp) * fraction))
        chosen.extend(rng.choice(grp["sample_path"].values, size=n_keep,
                                 replace=False).tolist())

    kept = set(chosen)
    return meta_df[meta_df["sample_path"].isin(kept)].reset_index(drop=True)


def train_stage2(meta_df: pd.DataFrame, cls_array: np.ndarray,
                 fraction: float) -> nn.Module:
    """Train Stage 2 from scratch on `meta_df` subset."""
    n_samp = meta_df["sample_path"].nunique()
    print(f"\n  [TRAIN] fraction={fraction:.0%}  samples={n_samp:,}  "
          f"epochs={NUM_EPOCHS}")

    dataset = SampleEmbeddingDataset(meta_df, cls_array,
                                     max_channels=MAX_CHANNELS)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=(DEVICE == "cuda"),
                         drop_last=True, persistent_workers=True)

    model = build_channel_aggregator(use_mz_embedding=USE_MZ_EMBEDDING).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * NUM_EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=max(total_steps, 1),
        pct_start=min(WARMUP_STEPS / max(total_steps, 1), 0.3),
        anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_loss, best_state = float("inf"), None
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            cls_tok  = batch["cls_tokens"].to(DEVICE)
            mz       = batch["mz"].to(DEVICE)
            ch_mask  = batch["channel_mask"].to(DEVICE)
            B, C, _  = cls_tok.shape
            rand_mask = torch.rand(B, C, device=DEVICE) < MASK_RATIO
            rand_mask = rand_mask & ch_mask

            cls_input = cls_tok.clone()
            cls_input[rand_mask] = 0.0

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out   = model(cls_tokens=cls_input,
                              mz=mz if USE_MZ_EMBEDDING else None,
                              channel_mask=ch_mask)
                preds = model.predict_masked(out["channel_refined"], rand_mask)
                loss  = masked_channel_loss(preds, cls_tok, rand_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()
            epoch_loss += loss.item(); n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        if epoch % 5 == 0:
            print(f"    epoch {epoch}/{NUM_EPOCHS}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    # save checkpoint
    frac_tag = f"frac{int(fraction*100):03d}"
    torch.save(best_state, CKPT_DIR / f"stage2_{frac_tag}.pt")
    print(f"    best loss={best_loss:.4f}")
    return model


@torch.no_grad()
def extract_embeddings(model: nn.Module, cls_array: np.ndarray,
                       full_meta: pd.DataFrame) -> np.ndarray:
    """Extract sample CLS embeddings for ALL samples using trained model."""
    model.eval()
    meta_df  = pd.read_csv(META_CSV)
    dataset  = SampleEmbeddingDataset(meta_df, cls_array,
                                      max_channels=MAX_CHANNELS)
    loader   = DataLoader(dataset, batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=(DEVICE == "cuda"))
    agg_dim  = model.agg_dim
    all_cls  = []
    for batch in loader:
        cls_tok  = batch["cls_tokens"].to(DEVICE)
        mz       = batch["mz"].to(DEVICE)
        ch_mask  = batch["channel_mask"].to(DEVICE)
        out      = model(cls_tokens=cls_tok,
                         mz=mz if USE_MZ_EMBEDDING else None,
                         channel_mask=ch_mask)
        all_cls.append(out["sample_cls"].cpu().numpy())
    return np.concatenate(all_cls, axis=0).astype(np.float32)


def eval_retrieval(emb: np.ndarray, sm: pd.DataFrame,
                   batch_size: int = 256) -> dict:
    """Leave-one-study-out macro R@k and weighted R@k."""
    normed   = normalize(emb, norm="l2")
    organs   = sm["organ"].values
    ds_ids   = sm["dataset_id"].values
    n        = len(sm)

    # filter organs
    organ_counts   = sm["organ"].value_counts()
    organ_datasets = sm.groupby("organ")["dataset_id"].nunique()
    keep = set(organ_counts[
        (organ_counts >= MIN_ORGAN_N) & (organ_datasets >= MIN_DATASETS)
    ].index)

    records = []
    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        sims  = normed[start:end] @ normed.T
        for bi, gi in enumerate(range(start, end)):
            if organs[gi] not in keep:
                continue
            row = sims[bi].copy()
            row[ds_ids == ds_ids[gi]] = -np.inf
            row[gi] = -np.inf
            nn_org = organs[np.argsort(-row)[:max(KS)]]
            records.append({
                "organ": organs[gi],
                **{f"r@{k}": float((nn_org[:k] == organs[gi]).mean())
                   for k in KS},
            })

    df = pd.DataFrame(records)
    per_organ = df.groupby("organ")[[f"r@{k}" for k in KS]].mean()
    weights   = per_organ.index.map(organ_counts)
    result    = {}
    for k in KS:
        col = f"r@{k}"
        result[f"macro_r{k}"]    = per_organ[col].mean()
        result[f"weighted_r{k}"] = np.average(per_organ[col], weights=weights)
    result["n_organs"] = len(per_organ)
    return result


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print(f"[DEVICE] {DEVICE}")
    print("[LOAD] Stage 1 embeddings …")
    cls_array = np.load(str(CLS_NPY), mmap_mode="r")
    meta_df   = pd.read_csv(META_CSV)
    sm_eval   = load_full_meta()
    print(f"  cls shape: {cls_array.shape}  |  eval samples: {len(sm_eval)}")

    # Stage 1 baseline (no Stage 2 training) — mean-pool channel CLS per sample
    print("\n[BASELINE] Stage 1 mean-pool …")
    sp_to_si  = {sp: i for i, sp in enumerate(sm_eval["sample_path"])}
    emb_s1    = np.zeros((len(sm_eval), cls_array.shape[1]), dtype=np.float32)
    counts    = np.zeros(len(sm_eval), dtype=np.int32)
    for row, sp in enumerate(meta_df["sample_path"]):
        si = sp_to_si.get(sp)
        if si is not None:
            emb_s1[si] += cls_array[row]
            counts[si]  += 1
    valid = counts > 0
    emb_s1[valid] /= counts[valid, None]
    s1_metrics = eval_retrieval(emb_s1, sm_eval)
    print(f"  Stage 1 macro R@1 = {s1_metrics['macro_r1']:.3f}")

    rows = []
    for fraction in FRACTIONS:
        print(f"\n{'='*60}")
        print(f"FRACTION {fraction:.0%}")
        print('='*60)

        sub_meta = subsample_train_meta(meta_df, fraction)
        n_train  = sub_meta["sample_path"].nunique()
        print(f"  Training on {n_train:,} / {meta_df['sample_path'].nunique():,} samples")

        model   = train_stage2(sub_meta, cls_array, fraction)
        emb_s2  = extract_embeddings(model, cls_array, sm_eval)
        metrics = eval_retrieval(emb_s2, sm_eval)

        row = {"fraction": fraction, "n_train": n_train, **metrics}
        rows.append(row)
        print(f"  macro R@1 = {metrics['macro_r1']:.3f}  "
              f"(Stage 1 baseline = {s1_metrics['macro_r1']:.3f})")

        # free GPU memory between runs
        del model
        torch.cuda.empty_cache()

    # add Stage 1 as fraction=0 reference
    rows.insert(0, {"fraction": 0.0, "n_train": 0, **s1_metrics})

    df_out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "ablation_datasize_results.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[DONE] Results → {out_csv}")
    print(df_out[["fraction", "n_train", "macro_r1", "weighted_r1"]].to_string(index=False))


if __name__ == "__main__":
    main()
