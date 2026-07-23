"""
export_explorer_data.py
-----------------------
Exports data for the extended (multi-space) client-side embedding explorer:
  - "sample"  : Stage 2 sample-level embeddings (n=5,600), spatial thumbnail
                = mean-of-retained-channels projection per sample.
  - "chan1"   : Stage 1 channel-level embeddings, subsampled from the full
                158,405-channel corpus, spatial thumbnail = single-channel
                image for that specific channel.
  - "chan2"   : Stage 2 refined channel-level embeddings, same subsample
                indices as chan1 (so results are directly comparable),
                spatial thumbnail = the same single-channel images.

All embeddings int8-quantized; all thumbnails are 32x32 uint8 grayscale
(client renders them through a viridis colormap in JS, mean-projected /
per-channel intensity respectively). Output: one JS file assigning
window.EXPLORER_DATA_V2 = {...}.
"""

from __future__ import annotations
import base64
import io
import json
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from dataset import set_data_root, _resolve_path, _load_patch_stack

DATA_ROOT = str(MSI_RAW_DIR)
set_data_root(DATA_ROOT)

EMB_DIR  = METABOFM_ROOT / "outputs/embeddings_v2"
UMAP_DIR = METABOFM_ROOT / "outputs/sample_umap"
OUT_PATH = METABOFM_ROOT / "outputs" / "explorer_data.js"

# Thumbnails are PNG-compressed (not raw bytes) since MSI ion images have
# large uniform/padded regions that compress well (~2x smaller than raw
# uint8 in testing), which buys enough headroom to roughly double the linear
# resolution versus a raw-byte encoding at the same payload budget. Decoded
# lazily client-side (only the ~K shown points per click), not eagerly for
# all N points, since decoding is per-image async work.
SAMPLE_THUMB = 224      # sample-level (mean-of-channels) thumbnail size — full model-input resolution
CHAN_THUMB   = 224       # channel-level (single-channel) thumbnail size — shared by chan1/chan2, full model-input resolution
MODEL_INPUT_SIZE = 224  # matches dataset.py's img_size — pad-to-square then resize target
N_CHAN_TARGET = 3000    # approx. total channels after grouping by whole samples
RNG = np.random.RandomState(6740)


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def categorical(series: pd.Series, dtype=np.uint16):
    filled = series.fillna("Unknown").astype(str)
    uniques = sorted(filled.unique().tolist())
    idx = {v: i for i, v in enumerate(uniques)}
    codes = np.array([idx[v] for v in filled], dtype=dtype)
    return {"values": uniques, "codes_b64": b64(codes), "codes_dtype": np.dtype(dtype).name}


def quantize_i8(vecs: np.ndarray):
    normed = normalize(vecs.astype(np.float32), norm="l2")
    max_abs = np.abs(normed).max()
    scale = 127.0 / max_abs
    i8 = np.clip(np.round(normed * scale), -127, 127).astype(np.int8)
    return i8, float(scale)


def _pad_to_square(img2d: np.ndarray) -> np.ndarray:
    """Centered zero-pad (H, W) -> (S, S), matching dataset.py::_pad_to_square."""
    H, W = img2d.shape
    S = max(H, W)
    if H == W:
        return img2d
    out = np.zeros((S, S), dtype=img2d.dtype)
    top, left = (S - H) // 2, (S - W) // 2
    out[top:top + H, left:left + W] = img2d
    return out


def make_thumb_png(img2d: np.ndarray, out_size: int) -> bytes:
    """
    Reproduces the exact geometric preprocessing MetaboFM feeds to the model
    (dataset.py: pad-to-square, nearest-resize to 224x224) so the displayed
    image is not aspect-ratio-distorted, then downsamples (nearest, to
    preserve the same blocky/undistorted character) to out_size. Percentile-
    normalised for display contrast, PNG-encoded (optimize=True) — MSI ion
    images have large uniform/padded regions that compress well, so this
    buys meaningfully higher resolution than a raw-byte encoding at the same
    payload budget. Colour mapping (viridis) is applied client-side.
    """
    img = np.nan_to_num(img2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    padded = _pad_to_square(img)
    im224 = Image.fromarray(padded).resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.NEAREST)
    arr = np.array(im224, dtype=np.float32)

    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    im = Image.fromarray((arr * 255).astype(np.uint8))
    im = im.resize((out_size, out_size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _pack_blob(png_list: list[bytes]) -> tuple[bytes, np.ndarray]:
    """Concatenate PNG byte strings into one blob + a uint32 offset table
    (n+1 entries: offsets[i]:offsets[i+1] is item i's byte range)."""
    offsets = np.zeros(len(png_list) + 1, dtype=np.uint32)
    for i, b in enumerate(png_list):
        offsets[i + 1] = offsets[i] + len(b)
    blob = b"".join(png_list)
    return blob, offsets


def sample_thumbnails(sample_paths: list[str]) -> tuple[bytes, np.ndarray]:
    """Mean-of-retained-channels projection thumbnail per unique sample_path,
    using the padded/model-input geometry. Returns (concatenated PNG blob, offsets)."""
    pngs = [b""] * len(sample_paths)
    for i, sp in enumerate(sample_paths):
        try:
            resolved = _resolve_path(sp, "")
            stack = _load_patch_stack(resolved)   # (C, H, W)
            mean_img = stack.mean(axis=0)
            pngs[i] = make_thumb_png(mean_img, SAMPLE_THUMB)
        except Exception as e:
            print(f"  [warn] sample thumb failed for {sp}: {e}")
            pngs[i] = make_thumb_png(np.zeros((8, 8), dtype=np.float32), SAMPLE_THUMB)
        if (i + 1) % 500 == 0:
            print(f"  sample thumbs {i+1}/{len(sample_paths)}", flush=True)
    return _pack_blob(pngs)


def channel_thumbnails(rows: pd.DataFrame) -> tuple[bytes, np.ndarray]:
    """Single-channel image thumbnail (padded/model-input geometry) for each
    (sample_path, channel_idx) row, grouped by sample_path so each npz is
    decompressed only once. Shared between chan1/chan2 (same underlying image).
    Returns (concatenated PNG blob, offsets)."""
    n = len(rows)
    pngs = [b""] * n
    rows = rows.reset_index(drop=True)
    for sp, grp in rows.groupby("sample_path"):
        try:
            resolved = _resolve_path(sp, "")
            stack = _load_patch_stack(resolved)   # (C, H, W)
        except Exception as e:
            print(f"  [warn] channel thumb group failed for {sp}: {e}")
            for i in grp.index:
                pngs[i] = make_thumb_png(np.zeros((8, 8), dtype=np.float32), CHAN_THUMB)
            continue
        for i, ch in zip(grp.index, grp["channel_idx"]):
            ch = int(ch)
            img = stack[ch] if ch < stack.shape[0] else np.zeros((8, 8), dtype=np.float32)
            pngs[i] = make_thumb_png(img, CHAN_THUMB)
    return _pack_blob(pngs)


def _clean_organ_organism(df: pd.DataFrame, organ_col: str = "Organism_Part") -> pd.DataFrame:
    """
    Fixes known upstream metadata errors:
      - organ-name typos ("Kideny" -> "Kidney", "colon" -> "Colon")
      - a handful of rows (7 in the full corpus) where the organ name was
        mistakenly entered into the `organism` field instead of `Organism_Part`
        (e.g. organism="Mouse Brain" with Organism_Part left blank), which
        left both fields wrong: organism should be the species, and the organ
        name belongs in Organism_Part.
    """
    df = df.copy()
    df["organ"] = df[organ_col].replace({"Kideny": "Kidney", "colon": "Colon"})

    crossed = df["organism"].astype(str).str.fullmatch(r"Mouse\s+Brain", case=False, na=False)
    df.loc[crossed, "organism"] = "Mus musculus"
    df.loc[crossed & (df["organ"].isna() | (df["organ"].astype(str) == "nan")), "organ"] = "Brain"
    return df


def build_sample_space():
    print("=== sample space (Stage 2, n=5600) ===")
    emb  = np.load(str(EMB_DIR / "stage2_sample_cls.npy")).astype(np.float32)
    xy   = np.load(str(UMAP_DIR / "umap2d_stage2.npy")).astype(np.float32)
    ch   = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                       usecols=["sample_path", "Organism_Part", "organism",
                                "analyzerType", "polarity", "dataset_id", "ionisationSource"])
    samp = ch.drop_duplicates("sample_path").reset_index(drop=True)
    sm   = pd.read_csv(EMB_DIR / "stage2_sample_meta.csv")
    sm   = sm.merge(samp, on="sample_path", how="left")
    assert len(sm) == len(emb)
    sm = _clean_organ_organism(sm)

    emb_i8, scale = quantize_i8(emb)
    print("  thumbnails …")
    thumb_blob, thumb_offsets = sample_thumbnails(sm["sample_path"].tolist())

    return {
        "n": len(sm),
        "umap_b64": b64(xy),
        "emb_i8_b64": b64(emb_i8),
        "emb_dim": int(emb.shape[1]),
        "emb_scale": scale,
        "thumb_blob_b64": base64.b64encode(thumb_blob).decode("ascii"),
        "thumb_offsets_b64": b64(thumb_offsets),
        "thumb_size": SAMPLE_THUMB,
        "thumb_kind": "mean_projection",
        "organ": categorical(sm["organ"], np.uint8),
        "organism": categorical(sm["organism"], np.uint8),
        "analyzerType": categorical(sm["analyzerType"], np.uint8),
        "polarity": categorical(sm["polarity"], np.uint8),
        "ionisationSource": categorical(sm["ionisationSource"], np.uint8),
        "dataset": categorical(sm["dataset_id"], np.uint16),
    }


def select_channel_subsample_by_sample(meta: pd.DataFrame, target_n: int) -> np.ndarray:
    """
    Randomly select whole samples (not individual channels) and keep ALL of
    each selected sample's channels, until the running total reaches
    target_n. This guarantees that any sample appearing in the channel-level
    UMAP is fully represented (all its channels present), rather than a
    scattered handful of unrelated channels.
    """
    unique_samples = meta["sample_path"].unique().tolist()
    RNG.shuffle(unique_samples)
    chosen_idx = []
    total = 0
    by_sample = meta.groupby("sample_path").indices  # sample_path -> row positions
    for sp in unique_samples:
        idxs = by_sample[sp]
        chosen_idx.extend(idxs.tolist())
        total += len(idxs)
        if total >= target_n:
            break
    return np.sort(np.array(chosen_idx, dtype=np.int64))


def build_channel_space(emb_path: Path, meta_path: Path, label: str,
                        subsample_idx: np.ndarray, shared_thumbs):
    print(f"=== channel space ({label}, n={len(subsample_idx)}) ===")
    emb_full = np.load(str(emb_path)).astype(np.float32)
    meta_full = pd.read_csv(meta_path, usecols=[
        "sample_path", "channel_idx", "mz", "dataset_id",
        "organism", "polarity", "Organism_Part", "analyzerType", "ionisationSource"])

    emb = emb_full[subsample_idx]
    rows = meta_full.iloc[subsample_idx].reset_index(drop=True)
    rows = _clean_organ_organism(rows)

    import umap as umap_lib
    print("  fitting UMAP …")
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            metric="cosine", random_state=6740)
    xy = reducer.fit_transform(emb).astype(np.float32)

    emb_i8, scale = quantize_i8(emb)

    if shared_thumbs is None:
        print("  thumbnails …")
        shared_thumbs = channel_thumbnails(rows[["sample_path", "channel_idx"]])

    return {
        "n": len(rows),
        "umap_b64": b64(xy),
        "emb_i8_b64": b64(emb_i8),
        "emb_dim": int(emb.shape[1]),
        "emb_scale": scale,
        "thumb_size": CHAN_THUMB,
        "thumb_kind": "single_channel",
        "mz": [round(float(v), 4) if pd.notna(v) else None for v in rows["mz"]],
        "organ": categorical(rows["organ"], np.uint8),
        "organism": categorical(rows["organism"], np.uint8),
        "analyzerType": categorical(rows["analyzerType"], np.uint8),
        "polarity": categorical(rows["polarity"], np.uint8),
        "ionisationSource": categorical(rows["ionisationSource"], np.uint8),
        "dataset": categorical(rows["dataset_id"], np.uint16),
    }, shared_thumbs


def main():
    sample_space = build_sample_space()

    meta1 = pd.read_csv(EMB_DIR / "resnet_cls_meta.csv", usecols=["sample_path", "channel_idx"])
    subsample_idx = select_channel_subsample_by_sample(meta1, N_CHAN_TARGET)
    print(f"Selected {len(subsample_idx)} channels from "
          f"{meta1.iloc[subsample_idx]['sample_path'].nunique()} whole samples")

    # Sanity check: resnet_cls_meta.csv and stage2_channel_meta.csv must be
    # row-aligned (same channel in the same row index) since both embedding
    # arrays are indexed by the same subsample_idx.
    meta2_check = pd.read_csv(EMB_DIR / "stage2_channel_meta.csv",
                              usecols=["sample_path", "channel_idx"]).iloc[subsample_idx]
    meta1_check = meta1.iloc[subsample_idx]
    assert (meta1_check["sample_path"].values == meta2_check["sample_path"].values).all()
    assert (meta1_check["channel_idx"].values == meta2_check["channel_idx"].values).all()
    print("  row-alignment check OK between resnet_cls_meta.csv and stage2_channel_meta.csv")

    chan1_space, shared_thumbs = build_channel_space(
        EMB_DIR / "resnet_cls_embeddings.npy", EMB_DIR / "resnet_cls_meta.csv",
        "Stage 1 channel", subsample_idx, shared_thumbs=None)
    chan2_space, _ = build_channel_space(
        EMB_DIR / "stage2_channel_refined.npy", EMB_DIR / "stage2_channel_meta.csv",
        "Stage 2 channel", subsample_idx, shared_thumbs=shared_thumbs)

    chan_thumb_blob, chan_thumb_offsets = shared_thumbs

    data = {
        "sample": sample_space,
        "chan1": chan1_space,
        "chan2": chan2_space,
        "chan_thumb_blob_b64": base64.b64encode(chan_thumb_blob).decode("ascii"),
        "chan_thumb_offsets_b64": b64(chan_thumb_offsets),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("window.EXPLORER_DATA_V2 = ")
        json.dump(data, f)
        f.write(";\n")

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"[OK] wrote {OUT_PATH}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
