"""
build_channel_csv.py
--------------------
Expand the sample-level metadata CSV (df_meta_REBUILT2.csv) into a
channel-level CSV where each row represents one (MSI sample, channel_index) pair.

Input
-----
  TILE_CSV   : sample-level CSV (one row per MSI sample)
               columns: sample_path, dataset_id, img_r, img_c, img_h,
                        img_w, channels, split, organism, polarity,
                        Organism_Part, Condition, analyzerType, ionisationSource

  DATA_ROOT  : root directory for resolving relative sample_path values
               (see MSI_RAW_DIR in metabofm_paths.py).

Output
------
  OUT_CSV    : channel-level CSV (one row per channel image)
               All sample-level columns are copied to each channel row, plus:
                 channel_idx   : integer index within the MSI sample (0-based)
                 mz            : m/z value from the NPZ (float)
                 manifest_row  : manifest row index from the NPZ (int, -1 if absent)
                 n_channels    : total channels in this MSI sample (same for all rows of a sample)

The output CSV is ready to be passed to dataset_v2.IonImageDataset.

Usage
-----
  python build_channel_csv.py

Notes
-----
  - MSI samples whose NPZ cannot be loaded are skipped (warning printed).
  - Progress is shown with tqdm.
  - Takes ~1-3 minutes for ~5000 samples on a network drive.
"""

from __future__ import annotations

from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

TILE_CSV   = str(METABOFM_ROOT / "metaspace_images_dump/df_meta_REBUILT2.csv")
DATA_ROOT  = MSI_RAW_DIR
OUT_CSV    = str(METABOFM_ROOT / "metaspace_images_dump/channels_v2.csv")

# Columns from the sample CSV to carry over to each channel row
TILE_COLS = [
    "sample_path", "dataset_id", "img_r", "img_c",
    "img_h", "img_w", "channels", "split",
    "organism", "polarity", "Organism_Part", "Condition",
    "analyzerType", "ionisationSource",
]

# ============================================================
# HELPERS
# ============================================================

def resolve_path(sample_path: str) -> Path:
    p = Path(sample_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = DATA_ROOT / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve: {sample_path}")


def expand_sample(sample_row: pd.Series) -> list[dict] | None:
    """
    Load the NPZ for one MSI sample and return a list of channel-level dicts.
    Returns None if the file cannot be loaded.
    """
    try:
        npz_path = resolve_path(str(sample_row["sample_path"]))
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "patch" not in z:
                print(f"  [SKIP] no 'patch' key: {npz_path.name}")
                return None
            C           = int(z["patch"].shape[0])
            mz_arr      = np.asarray(z["mz"],          dtype=np.float32) if "mz"          in z else np.zeros(C, dtype=np.float32)
            manifest_arr= np.asarray(z["manifest_row"], dtype=np.int64)  if "manifest_row" in z else np.full(C, -1, dtype=np.int64)
    except Exception as e:
        print(f"  [SKIP] load error {npz_path.name}: {e}")
        return None

    # Base dict from sample-level columns
    base = {col: sample_row.get(col) for col in TILE_COLS if col in sample_row.index}

    rows = []
    for ch in range(C):
        row = dict(base)
        row["channel_idx"]  = int(ch)
        row["mz"]           = float(mz_arr[ch])      if ch < len(mz_arr)       else 0.0
        row["manifest_row"] = int(manifest_arr[ch])  if ch < len(manifest_arr) else -1
        row["n_channels"]   = C
        rows.append(row)

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"[INFO] Reading sample CSV: {TILE_CSV}")
    sample_df = pd.read_csv(TILE_CSV, sep=None, engine="python")
    print(f"[INFO] MSI samples: {len(sample_df)}")

    # Remap old tile_* column names to img_* if present (backward compat with old CSVs)
    col_remap = {
        "tile_r": "img_r",
        "tile_c": "img_c",
        "tile_h": "img_h",
        "tile_w": "img_w",
    }
    sample_df = sample_df.rename(columns={k: v for k, v in col_remap.items() if k in sample_df.columns})

    # Keep only columns that exist in the CSV
    keep = [c for c in TILE_COLS if c in sample_df.columns]
    missing = [c for c in TILE_COLS if c not in sample_df.columns]
    if missing:
        print(f"  [WARN] Columns not found in sample CSV (will be skipped): {missing}")
    sample_df = sample_df[keep].copy()

    all_rows: list[dict] = []
    n_skipped = 0

    for _, sample_row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Expanding MSI samples"):
        result = expand_sample(sample_row)
        if result is None:
            n_skipped += 1
        else:
            all_rows.extend(result)

    if not all_rows:
        raise RuntimeError("No channel rows produced. Check DATA_ROOT and TILE_CSV paths.")

    out_df = pd.DataFrame(all_rows)

    # Reorder columns sensibly
    front = ["sample_path", "channel_idx", "mz", "manifest_row", "n_channels", "split", "dataset_id"]
    rest  = [c for c in out_df.columns if c not in front]
    out_df = out_df[front + rest]

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\n[DONE]")
    print(f"  Samples processed : {len(sample_df) - n_skipped} / {len(sample_df)}")
    print(f"  Samples skipped   : {n_skipped}")
    print(f"  Channel rows      : {len(out_df)}")
    print(f"  Output            : {out_path}")
    if "split" in out_df.columns:
        print(f"  Split breakdown   : {out_df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
