"""Validate tissue segmentation and interior-token selection for H&E analyses."""

from __future__ import annotations

from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from histology_masks import (
    PATCH_GRID,
    RECEPTIVE_FIELD_RADIUS,
    build_tissue_masks,
)
from optical_alignment import ion_to_native_optical_crop, native_optical_crop


HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
REG_DIR = METABOFM_ROOT / "outputs/optical_images/registration"
OUT_DIR = METABOFM_ROOT / "outputs/optical_images/mask_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_SCALES = (0.8, 1.0, 1.2)
INTERIOR_MARGINS = (16.0, 24.0, 32.0, 40.0, RECEPTIVE_FIELD_RADIUS)
DEFAULT_THRESHOLD_SCALE = 1.0
DEFAULT_MARGIN = RECEPTIVE_FIELD_RADIUS


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _patch_means(native: np.ndarray) -> np.ndarray:
    """Map a native-grid scalar image to the encoder's 28x28 token grid."""
    from PIL import Image

    h, w = native.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    padded = np.zeros((side, side), dtype=np.float32)
    padded[top:top + h, left:left + w] = native
    model = np.asarray(
        Image.fromarray(padded).resize((224, 224), Image.Resampling.BOX),
        dtype=np.float32,
    )
    return model.reshape(PATCH_GRID, 8, PATCH_GRID, 8).mean(axis=(1, 3))


def evaluate_setting(tokens: np.ndarray, score: np.ndarray, threshold_scale: float,
                     margin: float) -> tuple[dict, object]:
    masks = build_tissue_masks(
        score,
        threshold_scale=threshold_scale,
        interior_margin_px=margin,
    )
    selected = masks.patch_interior.ravel()
    row = {
        "threshold_scale": threshold_scale,
        "interior_margin_px": margin,
        "threshold": masks.threshold,
        "native_tissue_fraction": float(masks.native_tissue.mean()),
        "n_tissue_tokens": int(masks.patch_tissue.sum()),
        "n_boundary_tokens": int(masks.patch_boundary.sum()),
        "n_interior_tokens": int(selected.sum()),
        "pc1_variance": np.nan,
        "pc1_score_correlation": np.nan,
        "pc1_boundary_distance_correlation": np.nan,
    }
    if selected.sum() >= 20:
        x = tokens[selected]
        pc1 = PCA(n_components=1, random_state=42).fit_transform(x)[:, 0]
        pca = PCA(n_components=1, random_state=42).fit(x)
        patch_score = _patch_means(score).ravel()[selected]
        centers = np.arange(4, 224, 8)
        center_distance = masks.distance_to_background[np.ix_(centers, centers)].ravel()[selected]
        row.update({
            "pc1_variance": float(pca.explained_variance_ratio_[0]),
            "pc1_score_correlation": _correlation(pc1, np.log1p(patch_score)),
            "pc1_boundary_distance_correlation": _correlation(pc1, center_distance),
        })
    return row, masks


def render_diagnostic(organ: str, dataset_id: str, score: np.ndarray,
                      masks, registration_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(score, cmap="magma")
    axes[0, 0].set_title("Robust normalized MSI tissue score")

    axes[0, 1].imshow(masks.score_smoothed, cmap="magma")
    axes[0, 1].contour(masks.native_tissue, levels=[0.5], colors="cyan", linewidths=1)
    axes[0, 1].set_title(f"Cleaned Otsu mask (threshold={masks.threshold:.4g})")

    axes[0, 2].imshow(masks.native_tissue, cmap="gray")
    axes[0, 2].set_title(f"Native tissue mask ({masks.native_tissue.mean():.1%} of grid)")

    token_classes = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.uint8)
    token_classes[masks.patch_boundary] = 1
    token_classes[masks.patch_interior] = 2
    axes[1, 0].imshow(token_classes, cmap="viridis", vmin=0, vmax=2,
                      interpolation="nearest")
    axes[1, 0].set_title(
        f"Token classes: background / boundary / interior\n"
        f"{masks.patch_interior.sum()} interior at {DEFAULT_MARGIN:.0f}px margin"
    )

    axes[1, 1].imshow(masks.distance_to_background, cmap="cividis")
    axes[1, 1].contour(
        masks.distance_to_background,
        levels=[DEFAULT_MARGIN], colors="red", linewidths=1,
    )
    axes[1, 1].set_title("Distance to background in 224x224 model frame")

    if registration_path.exists():
        registration = np.load(registration_path, allow_pickle=False)
        if "affine_ion_to_optical" in registration:
            affine = registration["affine_ion_to_optical"]
            optical_crop = native_optical_crop(
                registration["optical"], affine, masks.native_tissue.shape
            )
            mapped_mask, mapped_valid = ion_to_native_optical_crop(
                masks.native_tissue.astype(np.float32),
                affine,
                optical_crop,
                resample=Image.Resampling.NEAREST,
            )
            axes[1, 2].imshow(optical_crop.image)
            contour_mask = np.where(mapped_valid, mapped_mask, 0)
            axes[1, 2].contour(
                contour_mask, levels=[0.5], colors="yellow", linewidths=1
            )
            axes[1, 2].set_title("MSI tissue boundary on native-resolution H&E")
        else:
            axes[1, 2].text(
                0.5, 0.5, "Registration lacks affine matrix", ha="center", va="center"
            )
    else:
        axes[1, 2].text(0.5, 0.5, "No registration data", ha="center", va="center")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Histology mask validation — {organ} ({dataset_id})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{organ}_{dataset_id}_mask_validation.png", dpi=180)
    plt.close(fig)


def process_one(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=False)
    if "robust_tissue_score" not in data:
        raise KeyError(f"{path.name} lacks robust_tissue_score; rerun probe_histology_comparison.py")
    organ, dataset_id = str(data["organ"]), str(data["dataset_id"])
    score, tokens = data["robust_tissue_score"], data["concat_tokens"]
    rows = []
    default_masks = None
    for threshold_scale in THRESHOLD_SCALES:
        for margin in INTERIOR_MARGINS:
            row, masks = evaluate_setting(tokens, score, threshold_scale, margin)
            row.update({"organ": organ, "dataset_id": dataset_id})
            rows.append(row)
            if threshold_scale == DEFAULT_THRESHOLD_SCALE and margin == DEFAULT_MARGIN:
                default_masks = masks
    render_diagnostic(
        organ,
        dataset_id,
        score,
        default_masks,
        REG_DIR / f"{organ}_{dataset_id}_registration_data.npz",
    )
    return rows


def main() -> None:
    paths = sorted(HIST_DIR.glob("*_tokens_data.npz"))
    if not paths:
        raise SystemExit(f"No token files found in {HIST_DIR}")
    rows = []
    for path in paths:
        rows.extend(process_one(path))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "mask_sensitivity.csv", index=False)
    default = table[
        (table.threshold_scale == DEFAULT_THRESHOLD_SCALE)
        & np.isclose(table.interior_margin_px, DEFAULT_MARGIN)
    ]
    print(default[[
        "organ", "native_tissue_fraction", "n_tissue_tokens",
        "n_boundary_tokens", "n_interior_tokens", "pc1_variance",
        "pc1_score_correlation", "pc1_boundary_distance_correlation",
    ]].to_string(index=False))
    print(f"\n[DONE] diagnostics and sensitivity table -> {OUT_DIR}")


if __name__ == "__main__":
    main()
