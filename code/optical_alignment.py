"""Native-resolution optical crops and ion-to-optical visualization warps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OpticalCrop:
    image: np.ndarray
    x0: int
    y0: int
    ion_footprint_xy: np.ndarray


def native_optical_crop(
    optical: np.ndarray,
    affine_ion_to_optical: np.ndarray,
    ion_shape: tuple[int, int],
    padding_fraction: float = 0.02,
) -> OpticalCrop:
    """Crop native-resolution optical data around the transformed ion grid."""
    h, w = ion_shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
    footprint = (np.asarray(affine_ion_to_optical, dtype=float) @ corners.T).T[:, :2]
    span = max(float(np.ptp(footprint[:, 0])), float(np.ptp(footprint[:, 1])))
    padding = max(2, int(round(span * padding_fraction)))
    x0 = max(0, int(np.floor(footprint[:, 0].min())) - padding)
    x1 = min(optical.shape[1], int(np.ceil(footprint[:, 0].max())) + padding)
    y0 = max(0, int(np.floor(footprint[:, 1].min())) - padding)
    y1 = min(optical.shape[0], int(np.ceil(footprint[:, 1].max())) + padding)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Transformed ion footprint does not intersect the optical image")
    return OpticalCrop(
        image=optical[y0:y1, x0:x1],
        x0=x0,
        y0=y0,
        ion_footprint_xy=footprint - np.array([x0, y0]),
    )


def ion_to_native_optical_crop(
    ion_image: np.ndarray,
    affine_ion_to_optical: np.ndarray,
    crop: OpticalCrop,
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample/rotate an ion-grid image into a native optical crop.

    Returns the transformed scalar image and a validity mask. The optical
    image is never resampled or downsampled.
    """
    ion = np.asarray(ion_image, dtype=np.float32)
    if ion.ndim != 2:
        raise ValueError(f"Expected a 2D ion-grid image, got {ion.shape}")
    transform = np.asarray(affine_ion_to_optical, dtype=float)
    if transform.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, got {transform.shape}")
    inverse = np.linalg.inv(transform)
    a, b, tx = inverse[0]
    c, d, ty = inverse[1]
    # PIL output coordinates are crop-local. Shift them into full optical
    # coordinates before applying optical -> ion-grid inverse mapping.
    coefficients = (
        a,
        b,
        a * crop.x0 + b * crop.y0 + tx,
        c,
        d,
        c * crop.x0 + d * crop.y0 + ty,
    )
    size = (crop.image.shape[1], crop.image.shape[0])
    transformed = np.asarray(
        Image.fromarray(ion, mode="F").transform(
            size, Image.Transform.AFFINE, coefficients, resample=resample, fillcolor=0
        ),
        dtype=np.float32,
    )
    valid = np.asarray(
        Image.fromarray(np.ones(ion.shape, dtype=np.uint8) * 255).transform(
            size,
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
    ) > 127
    return transformed, valid


def close_footprint(footprint_xy: np.ndarray) -> np.ndarray:
    """Return a closed polygon suitable for plotting."""
    points = np.asarray(footprint_xy)
    return np.vstack((points, points[0]))
