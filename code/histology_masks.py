"""Tissue and receptive-field-aware interior masks for the H&E experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    gaussian_filter,
    label,
    distance_transform_edt,
)


IMG_SIZE = 224
PATCH_GRID = 28
PATCH_STRIDE = IMG_SIZE // PATCH_GRID
# The theoretical receptive field of a ResNet-18 layer2 output is 99x99 input
# pixels, so a token center needs about 49 pixels of tissue on each side to be
# independent of zero padding/background.
LAYER2_RECEPTIVE_FIELD = 99
RECEPTIVE_FIELD_RADIUS = (LAYER2_RECEPTIVE_FIELD - 1) / 2


@dataclass(frozen=True)
class TissueMasks:
    threshold: float
    score_smoothed: np.ndarray
    native_tissue: np.ndarray
    model_tissue: np.ndarray
    patch_occupancy: np.ndarray
    patch_tissue: np.ndarray
    patch_interior: np.ndarray
    patch_boundary: np.ndarray
    distance_to_background: np.ndarray


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Compute Otsu's threshold without adding a scikit-image dependency."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or float(x.max()) <= float(x.min()):
        return float(x.min()) if x.size else 0.0
    hist, edges = np.histogram(x, bins=bins, range=(float(x.min()), float(x.max())))
    centers = (edges[:-1] + edges[1:]) / 2
    weight_left = np.cumsum(hist)
    weight_right = x.size - weight_left
    mean_left = np.cumsum(hist * centers) / np.maximum(weight_left, 1)
    sum_right = (hist * centers).sum() - np.cumsum(hist * centers)
    mean_right = sum_right / np.maximum(weight_right, 1)
    between = weight_left * weight_right * (mean_left - mean_right) ** 2
    valid = (weight_left > 0) & (weight_right > 0)
    between[~valid] = -1
    return float(centers[int(np.argmax(between))])


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, max_size: int) -> np.ndarray:
    holes = ~mask
    labels, n_labels = label(holes)
    if n_labels == 0:
        return mask
    counts = np.bincount(labels.ravel())
    touches_edge = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    fill = counts <= max_size
    fill[touches_edge] = False
    fill[0] = False
    return mask | fill[labels]


def _pad_to_square(array: np.ndarray, value=0) -> np.ndarray:
    h, w = array.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    out = np.full((side, side), value, dtype=array.dtype)
    out[top:top + h, left:left + w] = array
    return out


def build_tissue_masks(
    tissue_score: np.ndarray,
    *,
    threshold_scale: float = 1.0,
    patch_occupancy_min: float = 0.80,
    interior_margin_px: float = RECEPTIVE_FIELD_RADIUS,
) -> TissueMasks:
    """Segment tissue and classify token locations as boundary or interior.

    ``interior_margin_px`` is measured in the model's 224x224 input frame,
    making it comparable across acquisitions with different native sizes.
    """
    score = np.asarray(tissue_score, dtype=np.float32)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("tissue_score must be a finite 2D array")
    if float(score.max()) <= 0:
        raise ValueError("tissue_score contains no positive signal")

    sigma = max(0.75, min(score.shape) / 150.0)
    smoothed = gaussian_filter(score, sigma=sigma)
    base_threshold = otsu_threshold(smoothed)
    threshold = float(base_threshold * threshold_scale)
    native = smoothed > threshold

    radius = max(1, int(round(min(score.shape) / 200.0)))
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    native = binary_opening(native, structure=structure)
    native = binary_closing(native, structure=structure)
    area = native.size
    native = _remove_small_components(native, max(8, int(round(area * 0.002))))
    native = _fill_small_holes(native, max(8, int(round(area * 0.001))))

    padded = _pad_to_square(native.astype(np.uint8))
    model_tissue = np.asarray(
        Image.fromarray(padded * 255).resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.NEAREST
        )
    ) > 127
    distance = distance_transform_edt(model_tissue)

    occupancy = model_tissue.reshape(
        PATCH_GRID, PATCH_STRIDE, PATCH_GRID, PATCH_STRIDE
    ).mean(axis=(1, 3))
    patch_tissue = occupancy >= patch_occupancy_min
    centers = np.arange(PATCH_STRIDE // 2, IMG_SIZE, PATCH_STRIDE)
    center_distance = distance[np.ix_(centers, centers)]
    patch_interior = patch_tissue & (center_distance >= interior_margin_px)
    patch_boundary = patch_tissue & ~patch_interior

    return TissueMasks(
        threshold=threshold,
        score_smoothed=smoothed,
        native_tissue=native,
        model_tissue=model_tissue,
        patch_occupancy=occupancy,
        patch_tissue=patch_tissue,
        patch_interior=patch_interior,
        patch_boundary=patch_boundary,
        distance_to_background=distance,
    )
