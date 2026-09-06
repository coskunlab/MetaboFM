"""
probe_histology_comparison.py
------------------------------
Core experiment for the manuscript's H&E-comparison analysis: does MetaboFM's
learned spatial structure reveal anything not visible in the registered H&E
image for the same tissue section?

Pipeline, following the same conventions as probe_resnet_umap.py and
plot_utils.py (Stage 1 patch tokens -> PCA/UMAP spatial map, viridis for raw
ion images, pad-to-square + NEAREST-resize to 224 exactly as dataset.py):

  1. For each candidate dataset, pull the SAME 32 MSM-ranked channels used in
     training (the checkpoint's filtered manifest; channel_idx order == MSM rank,
     highest first —
     see dataset.py docstring), matched to METASPACE's live annotation table
     by m/z (falls back to a live API fetch when no local .npz cache exists).
  2. Run each channel through the trained Stage 1 ResNet-18 encoder
     (see --checkpoint; the same checkpoint used by plot_figure4.py/plot_figS7.py,
     e.g. the weights-v1 release's stage1_encoder_final.pt) to get 784 patch
     tokens (28x28 grid).
  3. Concatenate patch tokens across all 32 channels -> one (784, 32*D)
     multi-metabolite spatial fingerprint per sample.
  4. PCA + UMAP on that (784, D) matrix -> 2D spatial maps, upscaled to the
     ion-image grid.
  5. Save alongside the registration data from probe_optical_registration.py
     so plot_histology_comparison.py can render everything registered to the
     same H&E crop.

Usage
-----
  python probe_histology_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from metaspace import SMInstance

sys.path.insert(0, str(Path(__file__).parent))
from dataset import _pad_to_square
from models.resnet_encoder import build_ion_encoder_for_inference

# Use the exact filtered channel manifest recorded by the training run.  The
# unfiltered channels_v2.csv is not guaranteed to describe the checkpoint's
# training corpus.
CHANNEL_CSV = METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv"
DEFAULT_CHECKPOINT = METABOFM_ROOT / "checkpoints/stage1_encoder_final.pt"
CHECKPOINT = DEFAULT_CHECKPOINT  # overridden by --checkpoint in main()
OUT_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MZ_TOL = 0.001  # same tolerance as plot_utils.find_channel_for_mz

# Confirmed true-H&E candidates remaining in the comparison experiment.
# Placenta (no significant token-level PC1/annotation separation) and
# Pancreas (no annotatable region) were dropped and their files deleted.
CANDIDATES = {
    "Lung": "2023-06-27_22h58m39s",
    "Brain": "2019-11-25_17h14m31s",
}


def preprocess_channel(img_hw: np.ndarray) -> torch.Tensor:
    """Apply the checkpoint's exact inference preprocessing.

    This mirrors ``IonImageDataset(norm="tile_max", resize_mode="nearest")``:
    float32 input -> finite-value cleanup -> centered zero padding ->
    float-preserving nearest-neighbour resize -> tile-max normalization.
    Keeping the operation in torch avoids the former uint8 PIL round-trip,
    which quantized low-intensity MSI structure before inference.
    """
    if img_hw.ndim != 2:
        raise ValueError(f"Expected a 2D ion image, got shape={img_hw.shape}")

    img = np.asarray(img_hw, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.from_numpy(img).unsqueeze(0)  # (1, H, W), still float32
    x = _pad_to_square(x, pad_value=0.0)
    if x.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        x = F.interpolate(
            x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="nearest"
        ).squeeze(0)
    vmax = x.max()
    if vmax > 0:
        x = x / vmax
    return x


def build_robust_tissue_score(images: list[np.ndarray]) -> np.ndarray:
    """Aggregate ion images into a robust native-grid tissue score.

    Each channel is independently scaled by its 99th percentile among
    positive pixels and clipped to [0, 1].  A 10%-trimmed mean then prevents
    either a few very bright channels or a few empty channels from defining
    the tissue silhouette.  Unlike a raw sum, every channel has bounded
    influence and the result remains in [0, 1].
    """
    normalized = []
    for image in images:
        x = np.asarray(image, dtype=np.float32)
        positive = x[x > 0]
        scale = float(np.percentile(positive, 99)) if positive.size else 0.0
        if scale > 0:
            normalized.append(np.clip(x / scale, 0.0, 1.0))
        else:
            normalized.append(np.zeros_like(x, dtype=np.float32))

    stack = np.sort(np.stack(normalized, axis=0), axis=0)
    trim = int(np.floor(0.10 * stack.shape[0]))
    if trim and stack.shape[0] > 2 * trim:
        stack = stack[trim:-trim]
    return stack.mean(axis=0, dtype=np.float32)


@torch.no_grad()
def encode_channel(encoder, img_hw: np.ndarray) -> np.ndarray:
    """Preprocess one ion image and encode it to ``(784, D)`` tokens."""
    x = preprocess_channel(img_hw).unsqueeze(0).to(DEVICE)  # (1, 1, 224, 224)
    _, patches = encoder(x)
    return patches[0].cpu().numpy()


def match_trained_channels(sm: SMInstance, dataset_id: str, ch_csv_sub: pd.DataFrame):
    """For each trained channel_idx (MSM rank order), find the matching live
    METASPACE annotation image by m/z.

    The training manifest only stores m/z, so formula/adduct cannot be used as
    an independent join key.  When multiple live annotations are within the
    tolerance, choose deterministically by nearest m/z, then highest MSM, and
    save the selected formula/adduct and ambiguity count as provenance.
    """
    ds = sm.dataset(id=dataset_id)
    res = ds.results(database=("HMDB", "v4")).reset_index()
    if res.empty:
        # fall back to any available database
        res = ds.results().reset_index()

    matched = []
    for _, row in ch_csv_sub.sort_values("channel_idx").iterrows():
        target_mz = float(row["mz"])
        diffs = (res["mz"] - target_mz).abs()
        if diffs.empty or diffs.min() > MZ_TOL:
            print(f"  [WARN] channel_idx={row['channel_idx']} mz={target_mz:.4f}: no live match, skipping")
            continue
        candidates = res.loc[diffs <= MZ_TOL].copy()
        candidates["_mz_error"] = (candidates["mz"] - target_mz).abs()
        if "msm" not in candidates:
            candidates["msm"] = np.nan
        candidates["_msm_sort"] = candidates["msm"].fillna(-np.inf)
        candidates["_formula_sort"] = candidates["formula"].astype(str)
        candidates["_adduct_sort"] = candidates["adduct"].astype(str)
        candidates = candidates.sort_values(
            ["_mz_error", "_msm_sort", "_formula_sort", "_adduct_sort"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        best = candidates.iloc[0]
        try:
            imgs = ds.isotope_images(sf=best["formula"], adduct=best["adduct"])
            img = np.asarray(imgs._images[0], dtype=np.float32)
        except Exception as e:
            print(f"  [WARN] channel_idx={row['channel_idx']}: image fetch failed ({e}), skipping")
            continue
        n_nonfinite = int((~np.isfinite(img)).sum())
        if n_nonfinite:
            print(f"  [WARN] channel_idx={int(row['channel_idx'])}: replacing "
                  f"{n_nonfinite} non-finite intensities with zero")
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        matched.append({
            "channel_idx": int(row["channel_idx"]),
            "target_mz": target_mz,
            "live_mz": float(best["mz"]),
            "mz_error": float(best["_mz_error"]),
            "formula": str(best["formula"]),
            "adduct": str(best["adduct"]),
            "msm": float(best["msm"]) if pd.notna(best["msm"]) else np.nan,
            "n_candidates_within_tolerance": len(candidates),
            "n_nonfinite_replaced": n_nonfinite,
            "image": img,
        })
    return matched


def process_one(sm: SMInstance, encoder, ch_csv: pd.DataFrame, organ: str, dataset_id: str):
    print(f"\n=== {organ} ({dataset_id}) ===")
    sub = ch_csv[ch_csv.dataset_id == dataset_id]
    if sub.empty:
        print(f"[SKIP] no channels_v2.csv rows for {dataset_id}")
        return
    print(f"[INFO] {len(sub)} trained channels (MSM-ranked) to match")

    matched = match_trained_channels(sm, dataset_id, sub)
    if len(matched) < 5:
        print(f"[SKIP] only {len(matched)}/{len(sub)} channels matched, too few for a reliable embedding")
        return
    print(f"[INFO] matched {len(matched)}/{len(sub)} trained channels to live METASPACE images")

    shapes = {m["image"].shape for m in matched}
    if len(shapes) != 1:
        raise ValueError(f"Matched live ion images have inconsistent shapes: {sorted(shapes)}")
    imgs = [m["image"] for m in matched]
    H, W = imgs[0].shape

    duplicate_mz = len({round(m["target_mz"], 6) for m in matched}) != len(matched)
    if duplicate_mz:
        print("[WARN] duplicate trained m/z values are present; preserving them because "
              "they are distinct channels in the checkpoint manifest")
    ambiguous = sum(m["n_candidates_within_tolerance"] > 1 for m in matched)
    print(f"[INFO] {ambiguous}/{len(matched)} channels had multiple live annotations "
          f"within +/-{MZ_TOL:g} Da; deterministic selections saved as provenance")

    print(f"[INFO] encoding {len(imgs)} channels through Stage 1 ResNet-18 ({DEVICE}) ...")
    all_tokens = [encode_channel(encoder, im) for im in imgs]   # list of (784, D)
    stack = np.stack(all_tokens, axis=0)                        # (n_ch, 784, D)

    # Concatenate (not average) across channels per patch position: each of the
    # 784 spatial patches gets a (n_ch * D)-dim "cross-metabolite fingerprint"
    # vector. Mean-pooling was tried first and washed out fine structure (PC1
    # explained ~100% of variance -- just a generic tissue-density gradient);
    # concatenation preserves per-channel spatial variation instead of erasing it.
    concat_tokens = np.transpose(stack, (1, 0, 2)).reshape(stack.shape[1], -1)  # (784, n_ch*D)
    print(f"[INFO] concatenated tokens shape={concat_tokens.shape}")

    summed_ion = np.sum(imgs, axis=0)  # retained for comparison with legacy outputs
    robust_tissue_score = build_robust_tissue_score(imgs)

    # NOTE: PCA/UMAP are NOT run here — this machine's torch_gpu conda env has a
    # BLAS conflict that silently crashes (exit 127) inside sklearn/numpy.linalg
    # calls (same issue as matplotlib's savefig, see probe_optical_registration.py).
    # Only raw tokens are saved; embed_histology_comparison.py (base conda env)
    # does the PCA/UMAP + plotting.
    npz_path = OUT_DIR / f"{organ}_{dataset_id}_tokens_data.npz"
    np.savez(
        npz_path,
        organ=organ, dataset_id=dataset_id,
        n_channels_matched=len(matched),
        matched_channel_idx=np.array([m["channel_idx"] for m in matched], dtype=np.int32),
        matched_mz=np.array([m["target_mz"] for m in matched], dtype=np.float64),
        matched_live_mz=np.array([m["live_mz"] for m in matched], dtype=np.float64),
        matched_mz_error=np.array([m["mz_error"] for m in matched], dtype=np.float64),
        matched_formula=np.array([m["formula"] for m in matched]),
        matched_adduct=np.array([m["adduct"] for m in matched]),
        matched_msm=np.array([m["msm"] for m in matched], dtype=np.float64),
        matched_candidate_count=np.array(
            [m["n_candidates_within_tolerance"] for m in matched], dtype=np.int32
        ),
        matched_nonfinite_replaced=np.array(
            [m["n_nonfinite_replaced"] for m in matched], dtype=np.int32
        ),
        preprocessing=np.array(
            "float32->nan_to_num->center_pad_zero->nearest_224->tile_max"
        ),
        channel_manifest=np.array(str(CHANNEL_CSV)),
        checkpoint=np.array(str(CHECKPOINT)),
        summed_ion=summed_ion,
        robust_tissue_score=robust_tissue_score,
        channel_images=np.stack(imgs, axis=0).astype(np.float32),  # (n_ch, H, W), for PC-vs-raw-marker diagnostics
        concat_tokens=concat_tokens.astype(np.float32),
        H=H, W=W,
    )
    print(f"[DONE] saved -> {npz_path}")


def main():
    global CHECKPOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="Path to a Stage 1 encoder_final.pt (see weights-v1 release)")
    args = ap.parse_args()
    CHECKPOINT = Path(args.checkpoint)

    sm = SMInstance()
    ch_csv = pd.read_csv(CHANNEL_CSV)
    encoder = build_ion_encoder_for_inference(str(CHECKPOINT)).to(DEVICE)
    encoder.eval()
    print(f"[INFO] loaded Stage 1 encoder from {CHECKPOINT}")

    for organ, dataset_id in CANDIDATES.items():
        try:
            process_one(sm, encoder, ch_csv, organ, dataset_id)
        except Exception as e:
            print(f"[ERROR] {organ} ({dataset_id}): {type(e).__name__} {str(e)[:300]}")

    print("\n[NEXT] run embed_histology_comparison.py (base conda env) for PCA/UMAP + figures")


if __name__ == "__main__":
    main()
