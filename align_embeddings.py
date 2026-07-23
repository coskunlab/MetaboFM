"""
align_embeddings.py
-------------------
Build channel-level CSR arrays for channels_v2_filtered.csv by mapping
each filtered channel row to the pre-built CSR arrays in the dump that
are indexed to channels_v2.csv (the full unfiltered set).

Since channels_v2_filtered.csv already contains `manifest_row` and is a
strict subset of channels_v2.csv (matched by sample_path + channel_idx),
we just look up the right index in the existing CSR arrays — no NPZ access.

Existing dump arrays (indexed to channels_v2.csv row order):
  channel_cand_rows_flat.npy      (total_links, int32)
  channel_cand_rows_offsets.npy   (N_v2+1, int64)

Output (indexed to channels_v2_filtered.csv row order):
  <out_dir>/v2_channel_cand_rows_flat.npy
  <out_dir>/v2_channel_cand_rows_offsets.npy
  <out_dir>/v2_channels_with_n_cand.csv   (filtered CSV + n_cand column)

Usage
-----
  python align_embeddings_v2.py
"""

from __future__ import annotations

from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

DUMP    = METABOFM_ROOT / "metaspace_images_dump"
OUT_DIR = METABOFM_ROOT / "outputs/embeddings_v2"

FILTERED_CSV = str(METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv")
V2_CSV       = DUMP / "channels_v2.csv"

SRC_FLAT = DUMP / "channel_cand_rows_flat.npy"
SRC_OFFS = DUMP / "channel_cand_rows_offsets.npy"

OUT_FLAT = OUT_DIR / "v2_channel_cand_rows_flat.npy"
OUT_OFFS = OUT_DIR / "v2_channel_cand_rows_offsets.npy"
OUT_CSV  = OUT_DIR / "v2_channels_with_n_cand.csv"


def main():
    for p in (V2_CSV, SRC_FLAT, SRC_OFFS, Path(FILTERED_CSV)):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Source CSR arrays ...")
    src_flat = np.load(str(SRC_FLAT), mmap_mode="r")
    src_offs = np.load(str(SRC_OFFS), mmap_mode="r")
    print(f"  flat={src_flat.shape}  offsets={src_offs.shape}  (N_v2={len(src_offs)-1:,})")

    print("[LOAD] channels_v2.csv (full) ...")
    df_v2 = pd.read_csv(V2_CSV, usecols=["sample_path", "channel_idx"])
    key_to_v2_idx = {
        f"{sp}|{ci}": i
        for i, (sp, ci) in enumerate(zip(df_v2["sample_path"], df_v2["channel_idx"]))
    }
    print(f"  {len(df_v2):,} rows indexed")

    print("[LOAD] channels_v2_filtered.csv ...")
    df_flt = pd.read_csv(FILTERED_CSV)
    N = len(df_flt)
    print(f"  {N:,} filtered channels")

    # Map each filtered row to its index in channels_v2.csv
    print("[MAP] Building filtered → v2 index mapping ...")
    v2_indices = np.full(N, -1, dtype=np.int64)
    for i, (sp, ci) in enumerate(tqdm(
        zip(df_flt["sample_path"], df_flt["channel_idx"]), total=N
    )):
        v2_indices[i] = key_to_v2_idx.get(f"{sp}|{ci}", -1)

    missing = int((v2_indices == -1).sum())
    if missing:
        print(f"  [WARN] {missing:,} filtered rows not found in channels_v2.csv (will get 0 candidates)")

    # Build output CSR
    print("[BUILD] Filtered CSR arrays ...")
    counts = np.zeros(N, dtype=np.int32)
    for i, v2i in enumerate(v2_indices):
        if v2i < 0:
            continue
        a, b = int(src_offs[v2i]), int(src_offs[v2i + 1])
        counts[i] = b - a

    total = int(counts.sum())
    print(f"  total links: {total:,}  zero-candidate channels: {int((counts == 0).sum()):,}")

    out_offs = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=out_offs[1:])
    out_flat = np.empty(total, dtype=np.int32)

    cursor = 0
    for i, v2i in enumerate(tqdm(v2_indices, desc="filling flat")):
        if v2i < 0 or counts[i] == 0:
            continue
        a, b = int(src_offs[v2i]), int(src_offs[v2i + 1])
        chunk = src_flat[a:b].astype(np.int32, copy=False)
        out_flat[cursor:cursor + len(chunk)] = chunk
        cursor += len(chunk)

    np.save(str(OUT_FLAT), out_flat)
    np.save(str(OUT_OFFS), out_offs)

    df_flt["n_cand_molformer"] = counts
    df_flt.to_csv(str(OUT_CSV), index=False)

    print(f"\n[OK] flat    : {OUT_FLAT}  shape={out_flat.shape}")
    print(f"[OK] offsets : {OUT_OFFS}  shape={out_offs.shape}")
    print(f"[OK] csv     : {OUT_CSV}")
    print(f"  channels with ≥1 candidate: {int((counts > 0).sum()):,} / {N:,}")


if __name__ == "__main__":
    main()
