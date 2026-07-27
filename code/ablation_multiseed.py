"""
ablation_multiseed.py
---------------------
Re-run Stage 2 data-size ablation at all four fractions × two seeds,
each with 50 epochs. Reports mean ± std across seeds per fraction.

Outputs
-------
  OUT_DIR/ablation_multiseed_raw.csv     — one row per (fraction, seed)
  OUT_DIR/ablation_multiseed_summary.csv — mean ± std per fraction
  (also overwrites ablation_datasize_results.csv with mean values for plot_figure9.py)

Usage:
  conda run -n torch_gpu python ablation_multiseed.py
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

sys.path.insert(0, str(Path(__file__).parent))

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
CLS_NPY  = EMB_DIR / "resnet_cls_embeddings.npy"
META_CSV = EMB_DIR / "resnet_cls_meta.csv"
CHAN_META = EMB_DIR / "stage2_channel_meta.csv"
OUT_DIR  = METABOFM_ROOT / "outputs/ablation_datasize"
CKPT_DIR = OUT_DIR / "ckpts_multiseed"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS  = [0.25, 0.50, 0.75, 1.00]
SEEDS      = [42, 123]
NUM_EPOCHS = 50
BATCH_SIZE = 64
LR         = 5e-5
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


def load_eval_meta():
    ch = pd.read_csv(CHAN_META,
                     usecols=["sample_path", "Organism_Part", "organism", "dataset_id"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    samp["organ"] = samp["Organism_Part"].replace({"Kideny": "Kidney", "colon": "Colon"})
    return samp


def subsample(meta_df, fraction, seed):
    if fraction >= 1.0:
        return meta_df
    samp_meta = pd.read_csv(CHAN_META, usecols=["sample_path", "Organism_Part"])
    samp_meta["organ"] = samp_meta["Organism_Part"].replace(
        {"Kideny": "Kidney", "colon": "Colon"})
    unique = samp_meta.drop_duplicates("sample_path")[["sample_path", "organ"]]
    rng = np.random.default_rng(seed)
    chosen = []
    for _, grp in unique.groupby("organ"):
        n = max(1, round(len(grp) * fraction))
        chosen.extend(rng.choice(grp["sample_path"].values, size=n, replace=False).tolist())
    return meta_df[meta_df["sample_path"].isin(set(chosen))].reset_index(drop=True)


def train(meta_df, cls_array, fraction, seed):
    n = meta_df["sample_path"].nunique()
    print(f"  [TRAIN] frac={fraction:.0%} seed={seed} samples={n:,} epochs={NUM_EPOCHS}")
    dataset = SampleEmbeddingDataset(meta_df, cls_array, max_channels=MAX_CHANNELS)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=(DEVICE == "cuda"),
                         drop_last=True, persistent_workers=True)
    torch.manual_seed(seed)
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
        ep_loss, nb = 0.0, 0
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
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            ep_loss += loss.item(); nb += 1
        avg = ep_loss / max(nb, 1)
        if epoch % 10 == 0:
            print(f"    epoch {epoch}/{NUM_EPOCHS}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tag = f"frac{int(fraction*100):03d}_seed{seed}"
    torch.save(best_state, CKPT_DIR / f"stage2_{tag}.pt")
    print(f"    best loss={best_loss:.4f}")
    return model


@torch.no_grad()
def extract(model, cls_array):
    model.eval()
    meta_df = pd.read_csv(META_CSV)
    dataset = SampleEmbeddingDataset(meta_df, cls_array, max_channels=MAX_CHANNELS)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False,
                         num_workers=4, pin_memory=(DEVICE == "cuda"))
    all_cls = []
    for batch in loader:
        out = model(cls_tokens=batch["cls_tokens"].to(DEVICE),
                    mz=batch["mz"].to(DEVICE) if USE_MZ_EMBEDDING else None,
                    channel_mask=batch["channel_mask"].to(DEVICE))
        all_cls.append(out["sample_cls"].cpu().numpy())
    return np.concatenate(all_cls, axis=0).astype(np.float32)


def eval_retrieval(emb, sm, batch_size=256):
    normed = normalize(emb, norm="l2")
    organs = sm["organ"].values
    ds_ids = sm["dataset_id"].values
    n      = len(sm)
    counts   = sm["organ"].value_counts()
    n_ds     = sm.groupby("organ")["dataset_id"].nunique()
    keep     = set(counts[(counts >= MIN_ORGAN_N) & (n_ds >= MIN_DATASETS)].index)
    records  = []
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
    df     = pd.DataFrame(records)
    per_o  = df.groupby("organ")[[f"r@{k}" for k in KS]].mean()
    w      = per_o.index.map(counts)
    result = {}
    for k in KS:
        result[f"macro_r{k}"]    = per_o[f"r@{k}"].mean()
        result[f"weighted_r{k}"] = np.average(per_o[f"r@{k}"], weights=w)
    result["n_organs"] = len(per_o)
    return result


def main():
    print(f"[DEVICE] {DEVICE}")
    cls_array = np.load(str(CLS_NPY), mmap_mode="r")
    meta_df   = pd.read_csv(META_CSV)
    sm_eval   = load_eval_meta()

    # Stage 1 baseline once
    print("\n[BASELINE] Stage 1 …")
    sp_to_si = {sp: i for i, sp in enumerate(sm_eval["sample_path"])}
    emb_s1   = np.zeros((len(sm_eval), cls_array.shape[1]), dtype=np.float32)
    counts   = np.zeros(len(sm_eval), dtype=np.int32)
    for row, sp in enumerate(meta_df["sample_path"]):
        si = sp_to_si.get(sp)
        if si is not None:
            emb_s1[si] += cls_array[row]; counts[si] += 1
    valid = counts > 0
    emb_s1[valid] /= counts[valid, None]
    s1 = eval_retrieval(emb_s1, sm_eval)
    print(f"  Stage 1 macro R@1 = {s1['macro_r1']:.3f}")

    raw_rows = []
    total = len(FRACTIONS) * len(SEEDS)
    done  = 0
    for fraction in FRACTIONS:
        for seed in SEEDS:
            done += 1
            print(f"\n{'='*60}")
            print(f"[{done}/{total}]  FRACTION {fraction:.0%}  SEED {seed}")
            print('='*60)
            sub   = subsample(meta_df, fraction, seed)
            n_tr  = sub["sample_path"].nunique()
            model = train(sub, cls_array, fraction, seed)
            emb   = extract(model, cls_array)
            met   = eval_retrieval(emb, sm_eval)
            raw_rows.append({"fraction": fraction, "seed": seed,
                             "n_train": n_tr, **met})
            print(f"  macro R@1={met['macro_r1']:.3f}  weighted R@1={met['weighted_r1']:.3f}")
            del model; torch.cuda.empty_cache()

    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(OUT_DIR / "ablation_multiseed_raw.csv", index=False)
    print(f"\n[SAVED] raw → {OUT_DIR / 'ablation_multiseed_raw.csv'}")

    # Summary: mean ± std per fraction
    summary_rows = [{"fraction": 0.0, "n_train": 0,
                     **{k: s1[k] for k in s1},
                     **{k.replace("_r", "_r_std"): 0.0
                        for k in s1 if k.startswith("macro") or k.startswith("weighted")}}]
    for fraction, grp in df_raw.groupby("fraction"):
        row = {"fraction": fraction, "n_train": int(grp["n_train"].mean())}
        for col in [f"macro_r{k}" for k in KS] + [f"weighted_r{k}" for k in KS]:
            row[col]                       = grp[col].mean()
            row[col.replace("_r", "_std_r")] = grp[col].std()
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows).sort_values("fraction").reset_index(drop=True)
    df_sum.to_csv(OUT_DIR / "ablation_multiseed_summary.csv", index=False)

    # Also overwrite results csv for plot_figure9.py compatibility
    df_sum.rename(columns={"std_rmacro_r1": "macro_r1_std",
                            "std_rweighted_r1": "weighted_r1_std"},
                  errors="ignore").to_csv(
        OUT_DIR / "ablation_datasize_results.csv", index=False)

    print(f"\n[DONE] Summary:")
    print(df_sum[["fraction", "n_train", "macro_r1"]].to_string(index=False))


if __name__ == "__main__":
    main()
