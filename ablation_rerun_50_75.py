"""
ablation_rerun_50_75.py
-----------------------
Re-run Stage 2 ablation at 50% and 75% fractions with 50 epochs
(matching full training schedule) to fix non-monotonic dip from 30-epoch run.

Merges results back into ablation_datasize_results.csv.

Usage:
  conda run -n torch_gpu python ablation_rerun_50_75.py
"""

from __future__ import annotations
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
CLS_NPY  = EMB_DIR / "resnet_cls_embeddings.npy"
META_CSV = EMB_DIR / "resnet_cls_meta.csv"
CHAN_META = EMB_DIR / "stage2_channel_meta.csv"
OUT_DIR  = METABOFM_ROOT / "outputs/ablation_datasize"
CKPT_DIR = OUT_DIR / "ckpts"

FRACTIONS    = [0.50, 0.75]
SEED         = 42
NUM_EPOCHS   = 50
BATCH_SIZE   = 64
LR           = 5e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.05
GRAD_CLIP    = 1.0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
KS           = [1, 5, 10]
MIN_ORGAN_N  = 10
MIN_DATASETS = 2

from models.channel_aggregator import (
    build_channel_aggregator, masked_channel_loss, MASK_RATIO, STAGE1_DIM,
)
from pretrain_stage2 import SampleEmbeddingDataset, MAX_CHANNELS, USE_MZ_EMBEDDING


def load_full_meta():
    ch = pd.read_csv(CHAN_META,
                     usecols=["sample_path", "Organism_Part", "organism", "dataset_id"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    samp["organ"] = samp["Organism_Part"].replace({"Kideny": "Kidney", "colon": "Colon"})
    return samp


def subsample_train_meta(meta_df, fraction, seed=SEED):
    samp_meta = pd.read_csv(CHAN_META, usecols=["sample_path", "Organism_Part"])
    samp_meta["organ"] = samp_meta["Organism_Part"].replace(
        {"Kideny": "Kidney", "colon": "Colon"})
    unique_samps = samp_meta.drop_duplicates("sample_path")[["sample_path", "organ"]]
    rng = np.random.default_rng(seed)
    chosen = []
    for organ, grp in unique_samps.groupby("organ"):
        n_keep = max(1, round(len(grp) * fraction))
        chosen.extend(rng.choice(grp["sample_path"].values,
                                 size=n_keep, replace=False).tolist())
    return meta_df[meta_df["sample_path"].isin(set(chosen))].reset_index(drop=True)


def train_stage2(meta_df, cls_array, fraction):
    n_samp = meta_df["sample_path"].nunique()
    print(f"\n  [TRAIN] fraction={fraction:.0%}  samples={n_samp:,}  epochs={NUM_EPOCHS}")
    dataset = SampleEmbeddingDataset(meta_df, cls_array, max_channels=MAX_CHANNELS)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=(DEVICE == "cuda"),
                         drop_last=True, persistent_workers=True)
    model = build_channel_aggregator(use_mz_embedding=USE_MZ_EMBEDDING).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total = len(loader) * NUM_EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=max(total, 1),
        pct_start=min(WARMUP_STEPS / max(total, 1), 0.3), anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    best_loss, best_state = float("inf"), None
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, n_b = 0.0, 0
        for batch in loader:
            cls_tok  = batch["cls_tokens"].to(DEVICE)
            mz       = batch["mz"].to(DEVICE)
            ch_mask  = batch["channel_mask"].to(DEVICE)
            B, C, _  = cls_tok.shape
            rand_mask = (torch.rand(B, C, device=DEVICE) < MASK_RATIO) & ch_mask
            cls_input = cls_tok.clone(); cls_input[rand_mask] = 0.0
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                out   = model(cls_tokens=cls_input,
                              mz=mz if USE_MZ_EMBEDDING else None,
                              channel_mask=ch_mask)
                preds = model.predict_masked(out["channel_refined"], rand_mask)
                loss  = masked_channel_loss(preds, cls_tok, rand_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            epoch_loss += loss.item(); n_b += 1
        avg = epoch_loss / max(n_b, 1)
        if epoch % 10 == 0:
            print(f"    epoch {epoch}/{NUM_EPOCHS}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    frac_tag = f"frac{int(fraction*100):03d}_ep50"
    torch.save(best_state, CKPT_DIR / f"stage2_{frac_tag}.pt")
    print(f"    best loss={best_loss:.4f}")
    return model


@torch.no_grad()
def extract_embeddings(model, cls_array):
    model.eval()
    meta_df = pd.read_csv(META_CSV)
    dataset = SampleEmbeddingDataset(meta_df, cls_array, max_channels=MAX_CHANNELS)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False,
                         num_workers=4, pin_memory=(DEVICE == "cuda"))
    all_cls = []
    for batch in loader:
        cls_tok = batch["cls_tokens"].to(DEVICE)
        mz      = batch["mz"].to(DEVICE)
        ch_mask = batch["channel_mask"].to(DEVICE)
        out     = model(cls_tokens=cls_tok,
                        mz=mz if USE_MZ_EMBEDDING else None,
                        channel_mask=ch_mask)
        all_cls.append(out["sample_cls"].cpu().numpy())
    return np.concatenate(all_cls, axis=0).astype(np.float32)


def eval_retrieval(emb, sm, batch_size=256):
    normed = normalize(emb, norm="l2")
    organs = sm["organ"].values
    ds_ids = sm["dataset_id"].values
    n      = len(sm)
    organ_counts   = sm["organ"].value_counts()
    organ_datasets = sm.groupby("organ")["dataset_id"].nunique()
    keep = set(organ_counts[
        (organ_counts >= MIN_ORGAN_N) & (organ_datasets >= MIN_DATASETS)].index)
    records = []
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        sims = normed[start:end] @ normed.T
        for bi, gi in enumerate(range(start, end)):
            if organs[gi] not in keep:
                continue
            row = sims[bi].copy()
            row[ds_ids == ds_ids[gi]] = -np.inf
            row[gi] = -np.inf
            nn_org = organs[np.argsort(-row)[:max(KS)]]
            records.append({"organ": organs[gi],
                            **{f"r@{k}": float((nn_org[:k] == organs[gi]).mean())
                               for k in KS}})
    df = pd.DataFrame(records)
    per_organ = df.groupby("organ")[[f"r@{k}" for k in KS]].mean()
    weights   = per_organ.index.map(organ_counts)
    result    = {}
    for k in KS:
        result[f"macro_r{k}"]    = per_organ[f"r@{k}"].mean()
        result[f"weighted_r{k}"] = np.average(per_organ[f"r@{k}"], weights=weights)
    result["n_organs"] = len(per_organ)
    return result


def main():
    print(f"[DEVICE] {DEVICE}")
    cls_array = np.load(str(CLS_NPY), mmap_mode="r")
    meta_df   = pd.read_csv(META_CSV)
    sm_eval   = load_full_meta()

    results_csv = OUT_DIR / "ablation_datasize_results.csv"
    df_existing = pd.read_csv(results_csv)

    new_rows = []
    for fraction in FRACTIONS:
        print(f"\n{'='*60}\nFRACTION {fraction:.0%} — 50 epochs\n{'='*60}")
        sub_meta = subsample_train_meta(meta_df, fraction)
        n_train  = sub_meta["sample_path"].nunique()
        model    = train_stage2(sub_meta, cls_array, fraction)
        emb      = extract_embeddings(model, cls_array)
        metrics  = eval_retrieval(emb, sm_eval)
        row = {"fraction": fraction, "n_train": n_train, **metrics}
        new_rows.append(row)
        print(f"  macro R@1 = {metrics['macro_r1']:.3f}")
        del model; torch.cuda.empty_cache()

    # update existing results with re-run values
    df_new = pd.DataFrame(new_rows)
    df_merged = df_existing.copy()
    for _, r in df_new.iterrows():
        mask = df_merged["fraction"] == r["fraction"]
        for col in df_new.columns:
            df_merged.loc[mask, col] = r[col]

    df_merged.to_csv(results_csv, index=False)
    print(f"\n[DONE] Updated {results_csv}")
    print(df_merged[["fraction", "n_train", "macro_r1", "weighted_r1"]].to_string(index=False))


if __name__ == "__main__":
    main()
