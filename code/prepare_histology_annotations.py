"""Prepare a blinded, native-resolution H&E annotation bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from metabofm_paths import METABOFM_ROOT

ROOT = METABOFM_ROOT
REG_DIR = ROOT / "outputs" / "optical_images" / "registration"
IMAGE_DIR = (
    ROOT / "outputs" / "optical_images" / "histology_comparison"
    / "native_optical_panels"
)
BUNDLE_DIR = ROOT / "outputs" / "optical_images" / "annotations"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_SCHEMA = {
    "coordinate_system": {
        "space": "native_optical_crop_pixels",
        "origin": "top-left",
        "x_direction": "right",
        "y_direction": "down",
        "units": "pixels",
    },
    "required_classes": {
        "tissue": "Complete tissue footprint visible in the H&E crop.",
        "anatomical_region": (
            "A recognizable H&E-defined compartment. Set region_name to a "
            "specific blinded morphology label."
        ),
        "uncertain_exclude": (
            "Fold, tear, debris, staining artifact, ambiguous morphology, or "
            "any area that should not enter quantitative evaluation."
        ),
    },
    "optional_classes": {
        "lumen": "Clearly visible lumen or empty anatomical cavity.",
        "vessel": "Morphologically recognizable blood vessel.",
    },
    "required_feature_properties": [
        "classification", "region_name", "confidence", "annotator", "notes"
    ],
    "allowed_confidence": ["high", "medium", "low"],
}


def _parse_stem(registration_path: Path) -> tuple[str, str]:
    stem = registration_path.stem.removesuffix("_registration_data")
    return stem.split("_", 1)


def _blank_geojson(metadata: dict) -> dict:
    return {
        "type": "FeatureCollection",
        "name": metadata["annotation_stem"],
        "features": [],
        "metabofm_annotation_metadata": {
            "organ_label": metadata["organ_label"],
            "dataset_id": metadata["dataset_id"],
            "image_file": metadata["image_file"],
            "image_width_px": metadata["crop_width_px"],
            "image_height_px": metadata["crop_height_px"],
            "coordinate_space": "native_optical_crop_pixels_top_left_origin",
            "blinded_to_msi": True,
        },
    }


def main() -> None:
    rows = []
    for registration_path in sorted(REG_DIR.glob("*_registration_data.npz")):
        with np.load(registration_path, allow_pickle=False) as data:
            if "affine_ion_to_optical" not in data:
                raise KeyError(
                    f"{registration_path.name} lacks affine_ion_to_optical; "
                    "rerun fix_optical_warp.py"
                )
            affine = data["affine_ion_to_optical"].copy()
            ion_shape = tuple(int(v) for v in data["ion_img"].shape)
        organ, dataset_id = _parse_stem(registration_path)
        image_name = f"{organ}_{dataset_id}_HE_native.png"
        image_path = IMAGE_DIR / image_name
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing native optical asset {image_path}; rerun "
                "plot_top_loading_channels.py"
            )
        with Image.open(image_path) as image:
            crop_width, crop_height = image.size

        h, w = ion_shape
        corners = np.array(
            [[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float
        )
        footprint = (affine @ corners.T).T[:, :2]
        span = max(float(np.ptp(footprint[:, 0])), float(np.ptp(footprint[:, 1])))
        padding = max(2, int(round(span * 0.02)))
        crop_x0 = max(0, int(np.floor(footprint[:, 0].min())) - padding)
        crop_y0 = max(0, int(np.floor(footprint[:, 1].min())) - padding)

        annotation_stem = f"{organ}_{dataset_id}_regions"
        row = {
            "annotation_stem": annotation_stem,
            "organ_label": organ,
            "dataset_id": dataset_id,
            "image_file": str(image_path.relative_to(ROOT)),
            "crop_width_px": crop_width,
            "crop_height_px": crop_height,
            "crop_x0_in_original_optical_px": crop_x0,
            "crop_y0_in_original_optical_px": crop_y0,
            "ion_grid_width_px": ion_shape[1],
            "ion_grid_height_px": ion_shape[0],
            "annotation_coordinate_space": "native_optical_crop_pixels",
        }
        rows.append(row)

        # Never overwrite human work. These are starting templates only.
        geojson_path = BUNDLE_DIR / f"{annotation_stem}.geojson"
        if not geojson_path.exists():
            geojson_path.write_text(
                json.dumps(_blank_geojson(row), indent=2), encoding="utf-8"
            )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(BUNDLE_DIR / "annotation_metadata.csv", index=False)

    qc_path = BUNDLE_DIR / "blinded_qc.csv"
    if not qc_path.exists():
        qc = metadata[["organ_label", "dataset_id", "image_file"]].copy()
        qc["genuine_he"] = ""
        qc["organ_plausible"] = ""
        qc["section_adequate"] = ""
        qc["recognizable_compartments"] = ""
        qc["include_for_analysis"] = ""
        qc["reviewer_or_annotator"] = ""
        qc["notes"] = ""
        qc.to_csv(qc_path, index=False)

    (BUNDLE_DIR / "label_schema.json").write_text(
        json.dumps(LABEL_SCHEMA, indent=2), encoding="utf-8"
    )
    (BUNDLE_DIR / "README.md").write_text(
        """# Blinded H&E annotation bundle

## Images

Open the lossless native-resolution PNG listed in `annotation_metadata.csv`.
Do not inspect MSI, ion-channel, PCA, UMAP, or MetaboFM-derived panels while
annotating. Complete `blinded_qc.csv` before drawing regions.

## Required annotations

1. Draw the complete visible tissue footprint as `tissue`.
2. Draw each recognizable H&E compartment as `anatomical_region` and give it
   a morphology-based `region_name`.
3. Mark folds, tears, debris, ambiguous tissue, and other artifacts as
   `uncertain_exclude`.
4. Use `high`, `medium`, or `low` confidence and record the annotator.

Do not force an organ-specific label when morphology is uncertain. In
particular, the acquisition currently labelled Brain requires independent
morphological confirmation.

## Coordinates and export

GeoJSON coordinates must remain in the native crop's pixel coordinate system:
origin at top-left, x increasing right, y increasing down. Do not resize,
rotate, or re-export the image before annotation. Export one GeoJSON per image
using the provided filename. QuPath polygon annotations exported as GeoJSON
are suitable; ensure class and region name are included in feature properties.

The blank GeoJSON files are templates and may be replaced by annotation-tool
exports. Rerunning `prepare_histology_annotations.py` will not overwrite an
existing GeoJSON or `blinded_qc.csv`.
""",
        encoding="utf-8",
    )
    print(f"[DONE] prepared {len(metadata)} annotation cases -> {BUNDLE_DIR}")


if __name__ == "__main__":
    main()
