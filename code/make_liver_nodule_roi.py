"""
make_liver_nodule_roi.py
------------------------
Threshold a single channel of the metastatic liver TIFF to create a
per-blob nodule mask. Each connected component blob becomes its own
ImageJ ROI file (<stem>_1.roi, _2.roi, ...) so probe_resnet_umap.py
unions them into a binary positive mask (nodules vs background).

Tune these parameters:
  CHANNEL     : which channel to threshold (1 = ch 1, bright nodules)
  PERCENTILE  : within-tissue percentile cutoff (higher = fewer pixels)
  CLOSE_ITER  : morphological closing iterations (fills holes inside blobs)
  OPEN_ITER   : morphological opening iterations (removes small noise)
  TISSUE_MIN  : minimum intensity to be considered tissue (not background)
  MIN_BLOB_PX : minimum blob size in pixels to keep (filters tiny noise)

Usage
-----
  python make_liver_nodule_roi.py
"""

from pathlib import Path
from metabofm_paths import METABOFM_ROOT
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from scipy.ndimage import binary_closing, binary_opening, label as ndlabel
from scipy.spatial import ConvexHull
from roifile import ImagejRoi

# ── CONFIG ────────────────────────────────────────────────────────────────────

TIFF_PATH  = METABOFM_ROOT / "outputs/tiff_stacks_by_condition/Metastatic/2025-12-05_00h57m15s__Liver.tif"

CHANNEL    = 1      # 0-indexed channel with bright nodules
PERCENTILE = 50     # within-tissue intensity percentile cutoff (0–100)
CLOSE_ITER = 2      # morphological closing (fills holes)
OPEN_ITER  = 2      # morphological opening (removes noise)
TISSUE_MIN = 0.02   # pixels below this are considered background
MIN_BLOB_PX = 50    # minimum blob size to keep (pixels)

# ── LOAD ──────────────────────────────────────────────────────────────────────

roi_dir = TIFF_PATH.parent

with tifffile.TiffFile(str(TIFF_PATH)) as tf:
    stack = np.stack([p.asarray() for p in tf.pages], axis=0).astype(np.float32)

ch = stack[CHANNEL]
H, W = ch.shape
print(f"[INFO] Loaded ch {CHANNEL}  shape={H}x{W}  range=[{ch.min():.3f}, {ch.max():.3f}]")

# ── MASK + CONNECTED COMPONENTS ───────────────────────────────────────────────

tissue = ch > TISSUE_MIN
thresh = np.percentile(ch[tissue], PERCENTILE)
mask   = tissue & (ch > thresh)
mask   = binary_closing(mask, iterations=CLOSE_ITER)
mask   = binary_opening(mask, iterations=OPEN_ITER)

labeled, n_blobs = ndlabel(mask)
print(f"[INFO] Threshold p{PERCENTILE} = {thresh:.3f}")
print(f"[INFO] Raw blobs: {n_blobs}  total pixels: {mask.sum()}")

# filter small blobs
kept, sizes = [], []
for i in range(1, n_blobs + 1):
    sz = (labeled == i).sum()
    if sz >= MIN_BLOB_PX:
        kept.append(i)
        sizes.append(sz)
print(f"[INFO] Kept blobs >= {MIN_BLOB_PX}px: {len(kept)}  sizes: {sorted(sizes, reverse=True)[:10]}")

# ── BUILD UNION MASK + SAVE AS .npy ──────────────────────────────────────────

stem = TIFF_PATH.stem
# remove any stale ROI files for this sample
for old in sorted(roi_dir.glob(f"{stem}*.roi")):
    old.unlink()
    print(f"  Deleted old ROI: {old.name}")

union_mask = np.zeros((H, W), dtype=bool)
for blob_id in kept:
    union_mask |= (labeled == blob_id)

out_npy = roi_dir / f"{stem}_mask.npy"
np.save(str(out_npy), union_mask)
print(f"[OK] Mask saved: {out_npy.name}  ({union_mask.sum()} positive pixels, {len(kept)} blobs)")

# ── PNG OVERLAY ───────────────────────────────────────────────────────────────

from matplotlib.patches import Polygon as MplPoly
import matplotlib.cm as cm

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(ch, cmap="hot")
axes[0].set_title(f"Ch {CHANNEL} ion image")
axes[0].axis("off")

axes[1].imshow(ch, cmap="gray")
ov = np.zeros((*ch.shape, 4), dtype=np.float32)
ov[union_mask] = [1, 0.15, 0, 0.6]
axes[1].imshow(ov)
axes[1].set_title(
    f"Nodule blobs  ch{CHANNEL}  p{PERCENTILE}={thresh:.2f}  "
    f"close={CLOSE_ITER}  open={OPEN_ITER}  min={MIN_BLOB_PX}px\n"
    f"{len(kept)} blobs kept"
)
axes[1].axis("off")

plt.tight_layout()
out_png = roi_dir / "liver_nodule_mask_overlay.png"
plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Overlay saved: {out_png.name}")
