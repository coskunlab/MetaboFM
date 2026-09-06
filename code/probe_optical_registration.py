"""
probe_optical_registration.py
------------------------------
Registration overlay: warps a trained sample's ion images onto its
METASPACE optical (H&E) image using the dataset's stored affine transform,
to check the optical image is usable for the manuscript's H&E-comparison analysis.

METASPACE's `rawOpticalImage.transform` maps ION-IMAGE pixel coordinates
(col, row in the MSI acquisition grid) to OPTICAL-IMAGE pixel coordinates:
    [x_opt, y_opt, 1]^T = T @ [col_ion, row_ion, 1]^T
This script uses that transform to find where the ion-image grid falls in
the optical image (crop bbox + polygon overlay), and builds a summed-
intensity image across all annotated ion channels as a tissue silhouette —
a single channel is too sparse/noisy to visually confirm registration, but
the summed image reliably shows gross tissue structure (see Lung pilot).

Usage
-----
  python probe_optical_registration.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import numpy as np
import requests
from metaspace import SMInstance
from PIL import Image

OUT_DIR = METABOFM_ROOT / "outputs/optical_images/registration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: this machine's torch_gpu conda env (the one with `metaspace` installed)
# crashes silently (exit 127) inside matplotlib's savefig — a BLAS/backend
# conflict unrelated to this script's logic. So this script only fetches data
# and saves raw arrays; plotting is done by plot_optical_registration.py run
# under the base conda env, which has a working matplotlib but no `metaspace`.

# Confirmed true-H&E candidates from manual spot-check review (see
# outputs/optical_images/he_classification.csv + samples/ for the full triage).
CANDIDATES = {
    "Lung": "2023-06-27_22h58m39s",
    "Placenta": "2024-02-26_22h36m04s",
    "Pancreas": "2023-07-05_22h26m29s",
    "Brain": "2019-11-25_17h14m31s",  # borderline — organ label vs. morphology unconfirmed
}

# The training corpus's local .npz cache (C:\Users\eozturk7\data\msi_fm_samples5) is
# not present on this machine, so ion images are pulled live from METASPACE's API
# instead (same underlying pixel grid — image_size matches img_h/img_w in channels_v2.csv).


def affine_forward(T: list[list[float]], col, row):
    """Ion-grid (col, row) -> optical-image (x, y), per METASPACE convention.

    Implemented with plain Python arithmetic (not np.matmul/np.linalg) because
    this machine's torch_gpu conda env has a BLAS conflict that silently
    crashes the interpreter (exit 127, no traceback) on numpy linear-algebra
    calls.
    """
    a, b, tx = T[0]
    c, d, ty = T[1]
    ox = [a * cx + b * ry + tx for cx, ry in zip(col, row)]
    oy = [c * cx + d * ry + ty for cx, ry in zip(col, row)]
    return ox, oy


def process_one(sm: SMInstance, organ: str, dataset_id: str) -> None:
    print(f"\n=== {organ} ({dataset_id}) ===")
    ds = sm.dataset(id=dataset_id)
    print(f"[INFO] METASPACE image_size={ds.image_size}")

    imgs = ds.all_annotation_images(fdr=0.5)
    if not imgs:
        print(f"[SKIP] no annotated ion images for {dataset_id}")
        return
    print(f"[INFO] {len(imgs)} annotated ion images available")

    stack = np.array([np.nan_to_num(np.asarray(im._images[0])) for im in imgs])
    summed = stack.sum(axis=0)

    totals = stack.reshape(stack.shape[0], -1).sum(axis=1)
    med_idx = int(np.argsort(totals)[len(totals) // 2])
    ion_img = stack[med_idx]
    formula = imgs[med_idx].formula
    print(f"[INFO] representative channel idx={med_idx} formula={formula} shape={ion_img.shape}")

    raw = sm._gqclient.getRawOpticalImage(dataset_id)
    raw_im = raw.get("rawOpticalImage")
    if not raw_im:
        print(f"[SKIP] no optical image for {dataset_id}")
        return
    T = raw_im["transform"]
    r = requests.get(raw_im["url"], timeout=30)
    optical = np.asarray(Image.open(BytesIO(r.content)).convert("RGB"))
    print(f"[INFO] optical image shape={optical.shape}, transform={T}")

    h, w = ion_img.shape
    corners_col = [0, w, w, 0]
    corners_row = [0, 0, h, h]
    ox, oy = affine_forward(T, corners_col, corners_row)
    print(f"[INFO] ion-grid corners -> optical (x,y): "
          f"{list(zip([round(v, 1) for v in ox], [round(v, 1) for v in oy]))}")

    x0, x1 = max(0, int(min(ox))), min(optical.shape[1], int(np.ceil(max(ox))))
    y0, y1 = max(0, int(min(oy))), min(optical.shape[0], int(np.ceil(max(oy))))
    in_bounds = (x0 < x1) and (y0 < y1)
    print(f"[CHECK] ion-grid footprint falls inside optical canvas: {in_bounds}")

    optical_crop = optical[y0:y1, x0:x1] if in_bounds else optical

    # optical_crop above is an axis-aligned bbox crop -- misleading whenever T
    # has real rotation (it includes extra background and a different framing
    # than the ion images). optical_warped resamples the optical image
    # directly onto the ion grid's own (H, W) frame using PIL's affine
    # transform, which is pixel-aligned and the correct panel to compare
    # against ion images -- see fix_optical_warp.py for the full story.
    from PIL import Image as PILImage
    a, b, tx = T[0]
    c, d_, ty = T[1]
    optical_warped = np.array(PILImage.fromarray(optical).transform(
        (w, h), PILImage.AFFINE, (a, b, tx, c, d_, ty), resample=PILImage.BILINEAR))

    npz_path = OUT_DIR / f"{organ}_{dataset_id}_registration_data.npz"
    np.savez(
        npz_path,
        optical=optical,
        optical_crop=optical_crop,
        optical_warped=optical_warped,
        ion_img=ion_img,
        summed=summed,
        ox=np.array(ox, dtype=np.float64),
        oy=np.array(oy, dtype=np.float64),
        affine_ion_to_optical=np.asarray(T, dtype=np.float64),
        organ=organ,
        dataset_id=dataset_id,
        formula=str(formula),
        med_idx=med_idx,
    )
    print(f"[DONE] saved arrays -> {npz_path}")


def main():
    sm = SMInstance()
    for organ, dataset_id in CANDIDATES.items():
        try:
            process_one(sm, organ, dataset_id)
        except Exception as e:
            print(f"[ERROR] {organ} ({dataset_id}): {type(e).__name__} {str(e)[:200]}")
    print("\n[NEXT] run plot_optical_registration.py (base conda env) to render the figures")


if __name__ == "__main__":
    main()
