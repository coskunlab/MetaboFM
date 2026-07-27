"""
filter_samples.py
-----------------
Automatic quality filtering of MSI samples — no annotations required.

Filters applied (all configurable)
------------------------------------
  1. min_pixels_h / min_pixels_w
     Minimum spatial dimensions in pixels (original resolution, before resize).
     Removes thin strip-like samples that are too narrow to have meaningful
     spatial structure after patch embedding (e.g. 14×3 pixels → meaningless).

  2. max_aspect_ratio
     max(H,W) / min(H,W). Very elongated samples are tissue arrays or strips.

  3. min_fg_fraction / max_fg_fraction
     Fraction of pixels above the background threshold (mean composite).
     Too sparse  → mostly empty slide.
     Too dense   → stitched multi-sample image.

  4. max_components
     Number of large connected components in the foreground mask.
     Multi-organ / tissue-array slides have many disconnected blobs.

Behaviour
---------
- Processes samples one at a time.
- As each flagged sample is found, its mean composite PNG is saved immediately
  to `OUT_FLAGGED_DIR` — you can watch the folder live while the script runs.
- At the end, a sample of kept samples is saved to `OUT_KEPT_DIR` for comparison.
- Writes `sample_quality.csv` and `channels_v2_filtered.csv`.

Usage
-----
  python filter_samples.py
"""

from __future__ import annotations

from pathlib import Path

from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy import ndimage
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    print("[WARN] scipy not installed — n_components check disabled.  "
          "pip install scipy")

# ============================================================
# CONFIG
# ============================================================

SAMPLE_CSV   = str(METABOFM_ROOT / "metaspace_images_dump/df_meta_REBUILT2.csv")
CHANNEL_CSV  = str(METABOFM_ROOT / "metaspace_images_dump/channels_v2.csv")
DATA_ROOT    = MSI_RAW_DIR

OUT_QUALITY     = METABOFM_ROOT / "outputs/filtering/sample_quality.csv"
OUT_FILTERED    = METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv"
OUT_FLAGGED_DIR = METABOFM_ROOT / "outputs/filtering/previews/flagged"
OUT_KEPT_DIR    = METABOFM_ROOT / "outputs/filtering/previews/kept"

# ---- Spatial size thresholds (original pixel resolution) ----
MIN_PIXELS_H    = 16    # minimum height in pixels
MIN_PIXELS_W    = 16    # minimum width in pixels

# ---- Foreground thresholds ----
MIN_FG_FRACTION  = 0.03   # below → mostly empty
MAX_FG_FRACTION  = 0.97   # above → likely non-tissue / all-signal artifact
MIN_FG_PIXELS    = 64     # minimum absolute foreground pixel count

# ---- Shape thresholds ----
MAX_ASPECT_RATIO = 8.0    # max(H,W) / min(H,W) — covers intestine, spinal cord etc.

# ---- Connectivity threshold ----
MAX_COMPONENTS   = 8      # more than this → likely multi-organ / tissue array

# ---- Background estimation ----
BG_PERCENTILE    = 5      # percentile of nonzero pixels for threshold (lower = less fragmentation)

# ---- Minimum fraction of channels with nonzero signal ----
MIN_CHANNEL_FILL = 0.25   # at least 25% of channels must have some signal

# ---- How many channels to average for the composite (efficiency) ----
MAX_CHANNELS_COMPOSITE = 8

# ---- Kept previews (just for reference) ----
N_KEPT_PREVIEWS  = 30

# ============================================================
# HELPERS
# ============================================================

def resolve_path(sample_path: str) -> Path:
    p = Path(sample_path)
    if p.is_absolute() and p.exists():
        return p
    c = DATA_ROOT / p
    if c.exists():
        return c
    raise FileNotFoundError(f"Cannot resolve: {sample_path}")


def make_composite(patch: np.ndarray) -> np.ndarray:
    """
    Mean composite of up to MAX_CHANNELS_COMPOSITE channels, percentile-clipped
    to [0, 1]. Returns (H, W) float32.
    """
    C = patch.shape[0]
    ch_idx = np.linspace(0, C - 1, min(MAX_CHANNELS_COMPOSITE, C), dtype=int)
    mean_img = patch[ch_idx].mean(axis=0).astype(np.float32)
    lo, hi   = np.percentile(mean_img, 1), np.percentile(mean_img, 99)
    if hi <= lo:
        return np.zeros_like(mean_img)
    return np.clip((mean_img - lo) / (hi - lo), 0.0, 1.0)


def foreground_mask(mean_img: np.ndarray) -> np.ndarray:
    """Binary foreground mask using BG_PERCENTILE of nonzero pixels."""
    nonzero = mean_img[mean_img > 0]
    if nonzero.size < 4:
        return np.zeros_like(mean_img, dtype=bool)
    thresh = float(np.percentile(nonzero, BG_PERCENTILE))
    return mean_img > thresh


def count_components(fg: np.ndarray, H: int, W: int) -> int:
    """Number of connected components larger than 0.5% of image area."""
    if not _HAVE_SCIPY or not fg.any():
        return 0
    labeled, n = ndimage.label(fg)
    min_size    = max(4, int(0.005 * H * W))
    sizes       = np.bincount(labeled.ravel())[1:]
    return int((sizes >= min_size).sum())


def compute_metrics(npz_path: Path) -> dict:
    """Return quality metrics dict for one sample. Sets error=True on failure."""
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "patch" not in z:
                return {"error": True, "error_msg": "no patch key"}
            patch = np.asarray(z["patch"], dtype=np.float32)
    except Exception as e:
        return {"error": True, "error_msg": str(e)}

    C, H, W      = patch.shape
    aspect_ratio = max(H, W) / max(min(H, W), 1)
    comp         = make_composite(patch)
    fg           = foreground_mask(comp)
    fg_frac      = float(fg.mean())
    fg_pixels    = int(fg.sum())
    n_comp       = count_components(fg, H, W)

    # Channel fill: fraction of channels with any nonzero pixel
    channel_fill = float((patch.max(axis=(1, 2)) > 0).mean())

    return {
        "error":               False,
        "H":                   H,
        "W":                   W,
        "aspect_ratio":        round(aspect_ratio, 3),
        "foreground_fraction": round(fg_frac, 4),
        "fg_pixels":           fg_pixels,
        "n_components":        n_comp,
        "channel_fill":        round(channel_fill, 4),
        "mean_intensity":      round(float(comp.mean()), 4),
    }


def apply_filters(m: dict) -> str:
    """
    Return semicolon-separated reason string if sample should be removed,
    empty string if it passes all filters.
    """
    if m.get("error", False):
        return f"load_error: {m.get('error_msg','')}"

    reasons = []
    H, W = m.get("H", 0), m.get("W", 0)

    if H < MIN_PIXELS_H:
        reasons.append(f"thin_h={H}px")
    if W < MIN_PIXELS_W:
        reasons.append(f"thin_w={W}px")
    if m["aspect_ratio"] > MAX_ASPECT_RATIO:
        reasons.append(f"aspect={m['aspect_ratio']:.1f}")
    if m["foreground_fraction"] < MIN_FG_FRACTION:
        reasons.append(f"sparse_fg={m['foreground_fraction']:.3f}")
    if m["foreground_fraction"] > MAX_FG_FRACTION:
        reasons.append(f"dense_fg={m['foreground_fraction']:.3f}")
    if m.get("fg_pixels", 999) < MIN_FG_PIXELS:
        reasons.append(f"few_fg_pixels={m.get('fg_pixels',0)}")
    if m.get("channel_fill", 1.0) < MIN_CHANNEL_FILL:
        reasons.append(f"low_channel_fill={m.get('channel_fill',0):.2f}")
    if _HAVE_SCIPY and m["n_components"] > MAX_COMPONENTS:
        reasons.append(f"n_comp={m['n_components']}")

    return "; ".join(reasons)


def save_preview(npz_path: Path, metrics: dict, flag: str, out_path: Path) -> None:
    """
    Save a 2-panel PNG:
      Left  — mean composite (viridis)
      Right — foreground component overlay on grayscale composite
    Title shows metrics and flag reason.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            patch = np.asarray(z["patch"], dtype=np.float32)
    except Exception:
        return

    C, H, W = patch.shape
    comp    = make_composite(patch)
    fg      = foreground_mask(comp)

    # Component overlay
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    if _HAVE_SCIPY and fg.any():
        labeled, _ = ndimage.label(fg)
        min_size   = max(4, int(0.005 * H * W))
        cmap_comp  = plt.get_cmap("tab20")
        sizes      = np.bincount(labeled.ravel())[1:]
        for ci, sz in enumerate(sizes, 1):
            if sz >= min_size:
                col = cmap_comp((ci - 1) % 20)
                overlay[labeled == ci] = (*col[:3], 0.55)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    axes[0].imshow(comp, cmap="viridis", origin="upper", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].set_title("Mean composite", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(comp, cmap="gray", origin="upper", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[1].imshow(overlay, origin="upper", interpolation="nearest")
    axes[1].set_title("Foreground components", fontsize=8)
    axes[1].axis("off")

    status = f"FILTERED: {flag}" if flag else "KEPT"
    title  = (
        f"{npz_path.stem}  |  {H}×{W}px  "
        f"n_comp={metrics.get('n_components','?')}  "
        f"fg={metrics.get('foreground_fraction', 0):.2f}  "
        f"aspect={metrics.get('aspect_ratio', 0):.1f}\n"
        f"[{status}]"
    )
    fig.suptitle(title, fontsize=7, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_FLAGGED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_KEPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading sample CSV: {SAMPLE_CSV}")
    sample_df = pd.read_csv(SAMPLE_CSV, sep=None, engine="python")
    print(f"[INFO] Total MSI samples: {len(sample_df)}")
    print(f"\n[INFO] Filters active:")
    print(f"  min size         : H≥{MIN_PIXELS_H}px, W≥{MIN_PIXELS_W}px")
    print(f"  aspect ratio     : ≤{MAX_ASPECT_RATIO}")
    print(f"  foreground frac  : {MIN_FG_FRACTION:.2f} – {MAX_FG_FRACTION:.2f}")
    if _HAVE_SCIPY:
        print(f"  n_components     : ≤{MAX_COMPONENTS}")
    print()

    metric_rows  = []
    flagged_idx  = []   # for keeping order of flagged for preview naming
    n_flagged    = 0

    for i, (_, row) in enumerate(tqdm(sample_df.iterrows(),
                                       total=len(sample_df),
                                       desc="Filtering")):
        sp = str(row["sample_path"])
        try:
            npz_path = resolve_path(sp)
            metrics  = compute_metrics(npz_path)
        except FileNotFoundError as e:
            metrics  = {"error": True, "error_msg": str(e)}
            npz_path = None

        flag = apply_filters(metrics)
        metrics["sample_path"]  = sp
        metrics["quality_flag"] = flag
        metrics["keep"]         = flag == ""
        metric_rows.append(metrics)

        # Save PNG immediately when flagged (live tracking)
        if flag and npz_path is not None:
            stem     = Path(sp).stem
            flag_tag = flag[:50].replace("; ", "_").replace(" ", "").replace("=", "")
            out_png  = OUT_FLAGGED_DIR / f"{n_flagged:04d}__{stem}__{flag_tag}.png"
            save_preview(npz_path, metrics, flag, out_png)
            n_flagged += 1

    # ---- Build quality dataframe ----
    metrics_df = pd.DataFrame(metric_rows)
    keep_cols  = ["sample_path", "aspect_ratio", "foreground_fraction",
                  "fg_pixels", "n_components", "channel_fill",
                  "mean_intensity", "error", "quality_flag", "keep"]
    keep_cols  = [c for c in keep_cols if c in metrics_df.columns]
    sample_out = sample_df.merge(metrics_df[keep_cols], on="sample_path", how="left")

    n_total   = len(sample_out)
    n_kept    = int(sample_out["keep"].sum())
    n_removed = n_total - n_kept

    print(f"\n[RESULTS]")
    print(f"  Total   : {n_total}")
    print(f"  Kept    : {n_kept}  ({100*n_kept/n_total:.1f}%)")
    print(f"  Removed : {n_removed}  ({100*n_removed/n_total:.1f}%)")

    # Breakdown by flag reason
    if n_removed > 0:
        print(f"\n  Removal reasons:")
        for reason, cnt in (
            sample_out[~sample_out["keep"]]["quality_flag"]
            .str.split(";").explode().str.strip()
            .value_counts().items()
        ):
            print(f"    {reason}: {cnt}")

    # ---- Save quality CSV ----
    sample_out.to_csv(OUT_QUALITY, index=False)
    print(f"\n[OK] Quality CSV: {OUT_QUALITY}")

    # ---- Filter channels CSV ----
    print(f"[INFO] Filtering channel CSV: {CHANNEL_CSV}")
    chan_df      = pd.read_csv(CHANNEL_CSV, sep=None, engine="python")
    kept_paths   = set(sample_out[sample_out["keep"]]["sample_path"].tolist())
    chan_filtered = chan_df[chan_df["sample_path"].isin(kept_paths)].reset_index(drop=True)
    chan_filtered.to_csv(OUT_FILTERED, index=False)

    print(f"[OK] Filtered channel CSV: {OUT_FILTERED}")
    print(f"     {len(chan_df):,} → {len(chan_filtered):,} rows  "
          f"({100*len(chan_filtered)/len(chan_df):.1f}% retained)")

    # ---- Metric distributions ----
    kept = sample_out[sample_out["keep"]]
    print(f"\n[DISTRIBUTIONS — kept samples]")
    for col in ["H", "W", "n_components", "foreground_fraction", "fg_pixels", "channel_fill", "aspect_ratio"]:
        if col in kept.columns and kept[col].notna().any():
            v = kept[col].dropna()
            print(f"  {col:22s}: min={v.min():.1f}  "
                  f"median={v.median():.1f}  "
                  f"p95={v.quantile(0.95):.1f}  "
                  f"max={v.max():.1f}")

    # ---- Kept sample previews ----
    kept_sample = sample_out[sample_out["keep"]].sample(
        n=min(N_KEPT_PREVIEWS, n_kept), random_state=42
    ).reset_index(drop=True)

    print(f"\n[PREVIEWS] Saving {len(kept_sample)} kept sample previews…")
    for j, (_, row) in enumerate(tqdm(kept_sample.iterrows(),
                                       total=len(kept_sample),
                                       desc="Kept previews")):
        try:
            npz_path = resolve_path(str(row["sample_path"]))
            m        = {k: row[k] for k in ["n_components","foreground_fraction",
                                             "aspect_ratio","H","W"]
                        if k in row.index}
            stem     = Path(str(row["sample_path"])).stem
            save_preview(npz_path, m, "", OUT_KEPT_DIR / f"{j:04d}__{stem}.png")
        except Exception as e:
            tqdm.write(f"  [SKIP] {row['sample_path']}: {e}")

    print(f"\n[DONE]")
    print(f"  Flagged previews : {OUT_FLAGGED_DIR}  ({n_removed} PNGs)")
    print(f"  Kept previews    : {OUT_KEPT_DIR}  ({len(kept_sample)} PNGs)")


if __name__ == "__main__":
    main()
