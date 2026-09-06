"""
plot_top_loading_channels.py
------------------------------
Renders the top PC1-loading metabolite ion images (from
identify_top_loading_channels.py) alongside the registered H&E crop, for
each organ — direct visual check of whether individual metabolite spatial
patterns show structure invisible in H&E (the manuscript's H&E-comparison analysis).

Must run under the base conda env (torch_gpu's matplotlib crashes on savefig
on this machine).

Usage
-----
  python plot_top_loading_channels.py
"""

from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from optical_alignment import ion_to_native_optical_crop, native_optical_crop

HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
REG_DIR = METABOFM_ROOT / "outputs/optical_images/registration"
NATIVE_OPTICAL_DIR = HIST_DIR / "native_optical_panels"
NATIVE_OPTICAL_DIR.mkdir(parents=True, exist_ok=True)


def process_one(images_path: Path):
    stem = images_path.stem.replace("_top_loading_images", "")
    organ, dataset_id = stem.split("_", 1)

    images = np.load(images_path)
    ident_path = HIST_DIR / f"{stem}_top_loading_identities.csv"
    ident = pd.read_csv(ident_path) if ident_path.exists() else None

    reg_path = REG_DIR / f"{stem}_registration_data.npz"
    optical_crop = None
    affine = None
    if reg_path.exists():
        rd = np.load(reg_path, allow_pickle=False)
        if "affine_ion_to_optical" in rd:
            first_image = images[list(images.keys())[0]]
            affine = rd["affine_ion_to_optical"]
            optical_crop = native_optical_crop(rd["optical"], affine, first_image.shape)

    keys = list(images.keys())
    n_panels = len(keys) + (1 if optical_crop is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    i = 0
    if optical_crop is not None:
        # Preserve an exact, lossless native-pixel asset for final figure
        # assembly; the multi-panel matplotlib image is only a review preview.
        Image.fromarray(optical_crop.image).save(
            NATIVE_OPTICAL_DIR / f"{stem}_HE_native.png"
        )
        axes[i].imshow(optical_crop.image)
        axes[i].set_title(f"{organ} optical (H&E)\nnative optical resolution")
        axes[i].axis("off")
        i += 1

    ion_cmap = plt.get_cmap("viridis").copy()
    ion_cmap.set_bad(color="white", alpha=0)
    for key in keys:
        display_image = images[key]
        if optical_crop is not None and affine is not None:
            display_image, valid = ion_to_native_optical_crop(
                display_image, affine, optical_crop
            )
            display_image = np.ma.masked_where(~valid, display_image)
        axes[i].imshow(display_image, cmap=ion_cmap)
        title = key
        if ident is not None:
            row = ident[ident["image_key"] == key]
            if not row.empty:
                names = str(row.iloc[0]["top_names"])
                mz = row.iloc[0]["mz"]
                title = f"mz={mz:.2f}\n{names[:40]}"
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")
        i += 1

    fig.suptitle(
        f"Top PC1-loading channels — {organ} ({dataset_id})\n"
        "ion images transformed to native optical coordinates"
    )
    fig.tight_layout()
    out_path = HIST_DIR / f"{stem}_top_loading_channels.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] saved -> {out_path}")


def main():
    files = sorted(HIST_DIR.glob("*_top_loading_images.npz"))
    if not files:
        raise SystemExit(f"No *_top_loading_images.npz files found in {HIST_DIR}")
    for p in files:
        process_one(p)


if __name__ == "__main__":
    main()
