"""
compute_rawpixel_baseline.py
-----------------------------
Raw-pixel, no-learning baseline for the leave-one-dataset-out organ retrieval
benchmark in probe_crossdataset_retrieval.py.

Motivation: organ retrieval from a *learned* representation (Stage 1, Stage 2)
is only evidence of something non-trivial if the raw visual appearance of an
MSI sample does not already trivially give away its organ. This script builds
a per-sample feature vector directly from raw pixel content -- no trained
encoder, no m/z identity, no annotation metadata -- and evaluates it with the
exact same leave-one-dataset-out cosine-retrieval protocol used for Stage 1 /
Stage 2 / the m/z-only baseline, so the numbers are directly comparable.

Feature: mean-intensity projection across all retained channels for a sample,
percentile-normalised (1st-99th, matching the normalisation used for display
elsewhere in this project), downsampled to a small fixed grid, and flattened.
This is a classic non-learned "raw pixel" descriptor (cf. pixel-kNN baselines
in image classification literature) -- deliberately not a hand-engineered
metabolomics feature.

Usage:
  conda run -n torch_gpu python compute_rawpixel_baseline.py
"""

from __future__ import annotations
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import normalize

from plot_utils import MSI_DATA
from probe_crossdataset_retrieval import (
    load_metadata, cross_dataset_recall, summarise, KS,
)

OUT_DIR = METABOFM_ROOT / "outputs/crossdataset_retrieval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = 24   # downsample resolution -- small enough to guarantee "no learning",
            # large enough to preserve coarse organ-level shape/texture


def _raw_pixel_feature(sample_path: str) -> np.ndarray | None:
    p = MSI_DATA / Path(sample_path).name
    if not p.exists():
        return None
    patch = np.load(str(p))["patch"].astype(np.float32)   # (C, H, W)
    mean_proj = patch.mean(axis=0)                          # (H, W)

    nz = mean_proj[mean_proj > 0]
    if nz.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.percentile(nz, [1, 99])
    norm = np.clip((mean_proj - lo) / max(hi - lo, 1e-6), 0, 1)

    small = np.array(
        Image.fromarray((norm * 255).astype(np.uint8)).resize((GRID, GRID), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return small.flatten()   # (GRID*GRID,)


def build_rawpixel_embeddings(sm: pd.DataFrame) -> np.ndarray:
    print("  [raw pixel] building per-sample mean-projection features …")
    n = len(sm)
    emb = np.zeros((n, GRID * GRID), dtype=np.float32)
    n_missing = 0
    for i, sp in enumerate(sm["sample_path"]):
        feat = _raw_pixel_feature(sp)
        if feat is None:
            n_missing += 1
            continue
        emb[i] = feat
        if i % 500 == 0:
            print(f"    {i}/{n} …", end="\r")
    print(f"    {n}/{n} done ({n_missing} missing files)    ")
    return emb


def main():
    print("[LOAD] metadata …")
    sm, _ = load_metadata()
    print(f"  {len(sm)} samples, {sm['dataset_id'].nunique()} datasets, "
          f"{sm['organ'].nunique()} organs")

    emb_raw = build_rawpixel_embeddings(sm)
    np.save(str(OUT_DIR / "rawpixel_embeddings.npy"), emb_raw)
    sm[["sample_path", "organ", "dataset_id"]].to_csv(
        OUT_DIR / "rawpixel_embeddings_meta.csv", index=False)
    print(f"  saved rawpixel_embeddings.npy {emb_raw.shape} + meta")

    raw_normed = normalize(emb_raw, norm="l2")

    k_max = max(KS)
    print("\n[RETRIEVAL] Raw pixels (no learning) …")
    df_raw = cross_dataset_recall(raw_normed, sm, k_max)

    per_organ_raw, overall_raw = summarise(df_raw, sm, "Raw pixels (no learning)")

    per_organ_raw.to_csv(OUT_DIR / "crossdataset_retrieval_rawpixel_per_organ.csv", index=False)
    pd.DataFrame([overall_raw]).to_csv(OUT_DIR / "crossdataset_retrieval_rawpixel_overall.csv", index=False)

    print("\n" + "=" * 60)
    print("Raw-pixel baseline -- OVERALL (macro / weighted) Recall@k")
    print("=" * 60)
    for k in KS:
        print(f"  R@{k}: macro={overall_raw[f'macro_recall@{k}']:.3f}  "
              f"weighted={overall_raw[f'weighted_recall@{k}']:.3f}  "
              f"(random: macro={overall_raw[f'macro_random@{k}']:.3f}, "
              f"weighted={overall_raw[f'weighted_random@{k}']:.3f})")

    print("\n[DONE] outputs ->", OUT_DIR)


if __name__ == "__main__":
    main()
