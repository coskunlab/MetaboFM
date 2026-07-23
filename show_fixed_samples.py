"""
show_fixed_samples.py
---------------------
Shows the exact 5 fixed ion images used for reconstruction tracking.
Saves a grid PNG so you can verify they look right before training.

Usage:  python show_fixed_samples.py
"""

import sys
from pathlib import Path
from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dataset import IonImageDataset, set_data_root

CSV_PATH  = str(METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv")
DATA_ROOT = str(MSI_RAW_DIR)
OUT_PATH  = METABOFM_ROOT / "outputs/fixed_samples_preview.png"

FIXED_INDICES = [0, 39601, 79202, 118803, 158404]

set_data_root(DATA_ROOT)
df = pd.read_csv(CSV_PATH, sep=None, engine="python")
ds = IonImageDataset(df, base_dir=DATA_ROOT, img_size=224, norm="tile_max", augment=False)

fig, axes = plt.subplots(1, len(FIXED_INDICES), figsize=(4 * len(FIXED_INDICES), 4.5))
fig.suptitle("Fixed samples used for reconstruction tracking", fontsize=10)

for ax, idx in zip(axes, FIXED_INDICES):
    row  = df.iloc[idx]
    item = ds[idx]
    img  = item["pixel_values"][0].numpy()  # (H, W) in [0,1]

    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    disp = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    ax.imshow(disp, cmap="viridis", origin="upper", interpolation="nearest")
    sp   = Path(str(row["sample_path"])).stem
    ch   = int(row["channel_idx"])
    mz   = float(row.get("mz", 0))
    ax.set_title(f"idx={idx}\n{sp[:20]}\nch={ch}  m/z={mz:.2f}", fontsize=6)
    ax.axis("off")
    print(f"  idx={idx:6d}  ch={ch:3d}  m/z={mz:.4f}  file={Path(str(row['sample_path'])).name}")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[OK] Saved: {OUT_PATH}")

