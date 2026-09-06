"""
plot_optical_registration.py
------------------------------
Renders registration-check figures from arrays saved by
probe_optical_registration.py. Split into a separate script because this
machine's torch_gpu conda env (needed for the `metaspace` package) crashes
inside matplotlib's savefig; this must be run under the base conda env
instead, which has a working matplotlib but no `metaspace`.

Usage
-----
  python plot_optical_registration.py
"""

from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import matplotlib.pyplot as plt
import numpy as np

from optical_alignment import ion_to_native_optical_crop, native_optical_crop

OUT_DIR = METABOFM_ROOT / "outputs/optical_images/registration"


def plot_one(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)

    organ = str(d["organ"])
    dataset_id = str(d["dataset_id"])
    optical = d["optical"]
    ion_img = d["ion_img"]
    summed = d["summed"] if "summed" in d else None
    ox, oy = d["ox"], d["oy"]
    formula = str(d["formula"])
    med_idx = int(d["med_idx"])
    if "affine_ion_to_optical" not in d:
        raise ValueError(f"{npz_path.name} lacks affine_ion_to_optical; rerun fix_optical_warp.py")
    affine = d["affine_ion_to_optical"]
    optical_crop = native_optical_crop(optical, affine, ion_img.shape)
    ion_display, ion_valid = ion_to_native_optical_crop(ion_img, affine, optical_crop)
    ion_display = np.ma.masked_where(~ion_valid, ion_display)
    summed_display = None
    if summed is not None:
        summed_display, summed_valid = ion_to_native_optical_crop(summed, affine, optical_crop)
        summed_display = np.ma.masked_where(~summed_valid, summed_display)

    n_panels = 4 if summed is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(optical)
    axes[0].plot(np.append(ox, ox[0]), np.append(oy, oy[0]), "r-", linewidth=2)
    axes[0].set_title(f"{organ} optical (H&E) — full frame\nred = ion-grid footprint")
    axes[0].axis("off")

    axes[1].imshow(optical_crop.image)
    axes[1].set_title(f"Native-resolution optical crop\n({dataset_id})")
    axes[1].axis("off")

    axes[2].imshow(ion_display, cmap="inferno")
    axes[2].set_title(
        f"Single ion channel transformed to optical coordinates\n"
        f"idx={med_idx}, formula={formula}"
    )
    axes[2].axis("off")

    if summed is not None:
        axes[3].imshow(summed_display, cmap="gray")
        axes[3].set_title(
            "Summed intensity transformed to optical coordinates\n(tissue silhouette)"
        )
        axes[3].axis("off")

    fig.suptitle(f"Registration check — {organ} ({dataset_id})")
    fig.tight_layout()
    out_path = OUT_DIR / f"{organ}_{dataset_id}_registration_check.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] saved -> {out_path}")


def main():
    npz_files = sorted(OUT_DIR.glob("*_registration_data.npz"))
    if not npz_files:
        raise SystemExit(f"No *_registration_data.npz files found in {OUT_DIR}")
    for npz_path in npz_files:
        plot_one(npz_path)


if __name__ == "__main__":
    main()
