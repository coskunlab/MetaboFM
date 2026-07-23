"""
plot_utils.py
-------------
Shared helpers for Nature-quality figure generation across MetaboFM figure scripts.

Provides:
  set_nature_style()          — global rcParams for journal-quality output
  load_model_view_channel()   — pad+resize channel exactly as the model sees it
  load_model_view_specific()  — same for an indexed channel
  pick_median_sample()        — quality-filter then pick median-variance sample
  find_channel_for_mz()       — quality-filter + median for a specific m/z
  draw_pipeline_diagram()     — box-and-arrow pipeline wireframe for Panel A
"""

from __future__ import annotations
from pathlib import Path
from metabofm_paths import METABOFM_ROOT, MSI_RAW_DIR
from typing import Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── DATA PATH ──────────────────────────────────────────────────────────────────
MSI_DATA = MSI_RAW_DIR

# ── NATURE rcPARAMS ────────────────────────────────────────────────────────────
NATURE_RC = {
    "font.family":          "Arial",
    "font.size":             8,
    "axes.titlesize":        9,
    "axes.titleweight":      "bold",
    "axes.labelsize":        8,
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "legend.fontsize":       7,
    "legend.frameon":        False,
    "legend.handlelength":   1.2,
    "axes.linewidth":        0.7,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
    "xtick.minor.width":     0.5,
    "ytick.minor.width":     0.5,
    "xtick.major.size":      3.0,
    "ytick.major.size":      3.0,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "lines.linewidth":       1.2,
    "lines.markersize":      4,
    "patch.linewidth":       0.7,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
}


def set_nature_style() -> None:
    """Apply Nature-journal-quality rcParams. Call once at module level."""
    matplotlib.rcParams.update(NATURE_RC)


# ── ION IMAGE HELPERS ──────────────────────────────────────────────────────────

def _load_npz(sample_path: str | Path) -> np.ndarray | None:
    """Load patch array (C, H, W) float32; returns None if file missing."""
    p = MSI_DATA / Path(sample_path).name
    if not p.exists():
        return None
    return np.load(str(p))["patch"].astype(np.float32)


def _pad_and_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Replicate the exact dataset preprocessing the model receives:
      1. Zero-pad to square (centred)
      2. Bilinear resize to (size, size)
    Returns float32 array in [0, 1] (tile_max normalised for display).
    """
    from PIL import Image

    H, W = img.shape
    S = max(H, W)
    # centre-pad
    padded = np.zeros((S, S), dtype=np.float32)
    top  = (S - H) // 2
    left = (S - W) // 2
    padded[top:top + H, left:left + W] = img

    # tile-max normalise before resizing so PIL sees [0, 255]
    vmax = padded.max()
    if vmax > 0:
        padded = padded / vmax

    pil = Image.fromarray((padded * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.NEAREST)
    return np.array(pil, dtype=np.float32) / 255.0


def _channel_quality(img: np.ndarray) -> tuple[float, float]:
    """Return (spatial_std, nonzero_fraction) for a single (H, W) channel."""
    flat = img.flatten()
    nonzero_frac = float((flat > 0).mean())
    std = float(img.std())
    return std, nonzero_frac


def _interior_signal_frac(img: np.ndarray, border_frac: float = 0.15) -> float:
    """
    Fraction of the channel's nonzero signal that falls within the central
    (1 - 2*border_frac) region of the image, excluding a border_frac-wide
    rim on each side. Low values indicate signal concentrated at the tissue
    boundary (a thin outline) rather than distributed within the organ
    interior -- exactly the "boundary only, no signal inside" pattern that
    passes a plain nonzero-fraction/std check but looks empty/noisy when
    displayed.
    """
    H, W = img.shape
    bh, bw = int(H * border_frac), int(W * border_frac)
    total_nonzero = int((img > 0).sum())
    if total_nonzero == 0:
        return 0.0
    interior = img[bh:H - bh, bw:W - bw]
    interior_nonzero = int((interior > 0).sum())
    return interior_nonzero / total_nonzero


def _is_clean(img: np.ndarray, min_nonzero: float = 0.10, min_std: float = 0.02,
              min_interior_frac: float = 0.45) -> bool:
    """
    True if the channel image passes minimum quality thresholds.
    Rejects mostly-empty images, flat/noisy channels, and channels whose
    signal is mostly a thin boundary rim rather than distributed within the
    tissue interior (min_interior_frac).
    """
    std, nzf = _channel_quality(img)
    if nzf < min_nonzero or std < min_std:
        return False
    return _interior_signal_frac(img) >= min_interior_frac


def load_model_view_channel(
    sample_path: str | Path,
    size: int = 224,
    min_nonzero: float = 0.10,
    min_std: float = 0.02,
) -> np.ndarray | None:
    """
    Return the median-variance channel for this sample, processed exactly as
    the model sees it (pad-to-square → resize to `size` × `size`, tile_max 0-1).

    Selects the channel at the 50th percentile of spatial std among channels
    that pass quality filtering (non-zero fraction ≥ min_nonzero, std ≥ min_std).
    Falls back to the median of all channels if none pass.
    """
    patch = _load_npz(sample_path)
    if patch is None:
        return None
    C = patch.shape[0]

    stds = patch.reshape(C, -1).std(axis=1)
    clean_mask = np.array([_is_clean(patch[c], min_nonzero, min_std) for c in range(C)])

    candidates = np.where(clean_mask)[0]
    if len(candidates) == 0:
        candidates = np.arange(C)   # fallback: all channels

    # pick median-variance channel among candidates
    cand_stds = stds[candidates]
    median_idx = candidates[np.argsort(cand_stds)[len(cand_stds) // 2]]

    return _pad_and_resize(patch[median_idx], size)


def load_model_view_specific(
    sample_path: str | Path,
    channel_idx: int,
    size: int = 224,
) -> np.ndarray | None:
    """
    Load one indexed channel, processed as the model sees it
    (pad-to-square → resize to size×size, tile_max 0-1).
    """
    patch = _load_npz(sample_path)
    if patch is None:
        return None
    img = patch[int(channel_idx)]
    return _pad_and_resize(img, size)


def pick_median_sample(
    sample_paths: Sequence[str],
    n_candidates: int = 60,
    min_nonzero: float = 0.10,
    min_std: float = 0.02,
) -> str | None:
    """
    Return the sample at the median spatial variance among candidates that pass
    quality filtering.  This avoids cherry-picking (not best, not worst) while
    excluding truly empty or noisy images.

    Scans up to n_candidates paths; returns None if all fail to load.
    """
    scored: list[tuple[float, str]] = []
    for sp in list(sample_paths)[:n_candidates]:
        patch = _load_npz(sp)
        if patch is None:
            continue
        C = patch.shape[0]
        stds = patch.reshape(C, -1).std(axis=1)
        clean = [c for c in range(C) if _is_clean(patch[c], min_nonzero, min_std)]
        if not clean:
            continue
        best_std = float(stds[clean].max())
        scored.append((best_std, sp))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    return scored[len(scored) // 2][1]   # median


def find_channel_for_mz(
    ch_meta,
    mz_target: float,
    n_candidates: int = 200,
    min_nonzero: float = 0.10,
    min_std: float = 0.02,
) -> tuple[str | None, int | None]:
    """
    Find the median-variance clean channel for a given m/z.
    Matches within +/-0.001 of the target (equivalent to 3 d.p. precision),
    using an absolute-difference tolerance rather than rounding both sides
    and requiring exact equality — the latter can miss real matches that
    straddle a rounding boundary (e.g. target 329.2485 rounds to 329.248 or
    329.249 depending on floating-point representation, while actual channel
    values in the 329.2485-329.2487 range split across both buckets).

    Quality-filtered then median-selected (not cherry-picked). Filtering is
    applied at progressively relaxed levels only when the stricter level
    yields zero candidates, rather than dropping straight to no filtering at
    all -- the previous behaviour silently allowed boundary-only or empty
    channels through whenever fewer than n_candidates samples existed for a
    given m/z (common for rare/specific metabolites, e.g. drug-matched hits
    or single high-AP class representatives), which is exactly the failure
    mode reported for Figs. 6d and 7e.
    """
    cm = ch_meta.copy()
    matches = cm[(cm["mz"] - mz_target).abs() < 0.001].reset_index(drop=True)
    candidates = list(matches.head(n_candidates).iterrows())

    loaded: list[tuple[np.ndarray, str, int]] = []
    for _, row in candidates:
        patch = _load_npz(row["sample_path"])
        if patch is None:
            continue
        ci = int(row["channel_idx"])
        if ci >= patch.shape[0]:
            continue
        loaded.append((patch[ci], row["sample_path"], ci))

    if not loaded:
        return None, None

    # Progressively relax the quality bar until at least one candidate passes.
    levels = [
        dict(min_nonzero=min_nonzero, min_std=min_std, min_interior_frac=0.45),
        dict(min_nonzero=min_nonzero, min_std=min_std, min_interior_frac=0.25),
        dict(min_nonzero=min_nonzero * 0.5, min_std=min_std * 0.5, min_interior_frac=0.0),
    ]
    scored: list[tuple[float, str, int]] = []
    level_used = None
    for level_i, kwargs in enumerate(levels):
        scored = [(float(img.std()), sp, ci) for img, sp, ci in loaded if _is_clean(img, **kwargs)]
        if scored:
            level_used = level_i
            break

    if not scored:
        # Absolute last resort: no filtering at all.
        scored = [(float(img.std()), sp, ci) for img, sp, ci in loaded]
        level_used = "unfiltered"

    if level_used != 0:
        print(f"  [find_channel_for_mz] mz={mz_target:.4f}: only found clean candidates at "
              f"relaxed level {level_used} ({len(loaded)} loaded, {len(scored)} kept)")

    scored.sort(key=lambda x: x[0])
    _, best_sp, best_ci = scored[len(scored) // 2]
    return best_sp, best_ci


# ── BACKWARD-COMPATIBLE ALIASES ────────────────────────────────────────────────

def load_best_channel(sample_path: str | Path, size: int = 224) -> np.ndarray | None:
    """Alias for load_model_view_channel (median-quality, model-view)."""
    return load_model_view_channel(sample_path, size=size)


def load_specific_channel(
    sample_path: str | Path, channel_idx: int, size: int = 224
) -> np.ndarray | None:
    """Alias for load_model_view_specific (model-view, pad+resize)."""
    return load_model_view_specific(sample_path, channel_idx, size=size)


def pick_best_sample(
    sample_paths: Sequence[str], n_candidates: int = 60
) -> str | None:
    """Alias for pick_median_sample."""
    return pick_median_sample(sample_paths, n_candidates=n_candidates)

# ── PIPELINE DIAGRAM  (grid-based, illustrated) ───────────────────────────────

_BOX_COLORS = {
    "data":    ("#ddeeff", "#2166ac"),
    "model":   ("#e8f4e8", "#2ca02c"),
    "output":  ("#fff0e0", "#e08214"),
    "eval":    ("#f5e8f5", "#9467bd"),
    "default": ("#f0f0f0", "#666666"),
}

def _bc(kind):
    return _BOX_COLORS.get(kind, _BOX_COLORS["default"])

# ── ICON DRAWING FUNCTIONS ────────────────────────────────────────────────────
# Each takes (ax, cx, cy, bw, bh) in transAxes coordinates.
# cx, cy = box centre;  bw, bh = box width/height.
# Icon sits in the upper ~45 % of the box.

def _icon_msi(ax, cx, cy, bw, bh):
    """Tissue ion image - 5x5 pixel grid."""
    cmap = plt.get_cmap("viridis")
    pattern = np.array([
        [0.15, 0.45, 0.85, 0.65, 0.25],
        [0.35, 0.75, 0.95, 0.80, 0.40],
        [0.30, 0.70, 0.90, 0.70, 0.35],
        [0.15, 0.45, 0.65, 0.50, 0.20],
        [0.05, 0.15, 0.35, 0.20, 0.10],
    ])
    n = 5
    iw, ih = bw * 0.52, bh * 0.38
    x0 = cx - iw / 2
    y0 = cy + bh * 0.06
    pw, ph = iw / n, ih / n
    for r in range(n):
        for c in range(n):
            ax.add_patch(mpatches.Rectangle(
                (x0 + c * pw, y0 + (n - 1 - r) * ph), pw * 0.88, ph * 0.88,
                facecolor=cmap(pattern[r, c]), edgecolor="none",
                transform=ax.transAxes, zorder=5))
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), iw, ih, facecolor="none",
        edgecolor="#555", linewidth=0.6, transform=ax.transAxes, zorder=6))

def _icon_patches(ax, cx, cy, bw, bh):
    """28x28 patch grid - 3x3 coloured tiles."""
    iw, ih = bw * 0.50, bh * 0.38
    x0 = cx - iw / 2; y0 = cy + bh * 0.06
    colors = ["#2166ac","#d6604d","#4dac26",
              "#fdae6b","#9970ab","#74add1",
              "#e08214","#1b7837","#aaaaaa"]
    n = 3; pw = iw / n; ph = ih / n
    for r in range(n):
        for c in range(n):
            col = colors[r * n + c]
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0 + c * pw + 0.001, y0 + r * ph + 0.001), pw - 0.002, ph - 0.002,
                boxstyle="round,pad=0.003",
                facecolor=col, edgecolor="white", linewidth=0.4, alpha=0.85,
                transform=ax.transAxes, zorder=5))
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), iw, ih, facecolor="none",
        edgecolor="#555", linewidth=0.6, transform=ax.transAxes, zorder=6))

def _icon_resnet(ax, cx, cy, bw, bh):
    """ResNet - stacked conv-layer rectangles with activation lines."""
    iw, ih = bw * 0.46, bh * 0.38
    x0 = cx; y0 = cy + bh * 0.07
    layer_cols = ["#aec6cf", "#78a8c0", "#4d8fa8"]
    widths  = [iw * 0.80, iw * 0.62, iw * 0.44]
    offsets = [(-0.007, 0.007), (0.0, 0.0), (0.007, -0.007)]
    lh = ih * 0.48
    for (dx, dy), w, col in zip(offsets, widths, layer_cols):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 - w / 2 + dx, y0 - lh / 2 + dy), w, lh,
            boxstyle="round,pad=0.006",
            facecolor=col, edgecolor="#2166ac", linewidth=0.6,
            transform=ax.transAxes, zorder=5))
    # ReLU activation stripes
    for k in range(4):
        yy = y0 - lh * 0.18 + k * lh * 0.12
        ax.plot([x0 - iw * 0.18, x0 + iw * 0.18], [yy, yy],
                color="#e08214", lw=0.5, transform=ax.transAxes, zorder=7)

def _icon_transformer(ax, cx, cy, bw, bh):
    """Transformer - attention heatmap + Q/K/V labels."""
    n = 4
    iw, ih = bw * 0.46, bh * 0.36
    x0 = cx - iw / 2; y0 = cy + bh * 0.07
    cmap = plt.get_cmap("Blues")
    rng = np.random.RandomState(0)
    attn = np.eye(n) * 0.65 + rng.rand(n, n) * 0.35
    attn /= attn.sum(axis=1, keepdims=True)
    pw, ph = iw / n, ih / n
    for r in range(n):
        for c in range(n):
            ax.add_patch(mpatches.Rectangle(
                (x0 + c * pw + 0.0005, y0 + r * ph + 0.0005),
                pw * 0.94, ph * 0.94,
                facecolor=cmap(attn[r, c]), edgecolor="none",
                transform=ax.transAxes, zorder=5))
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), iw, ih, facecolor="none",
        edgecolor="#9467bd", linewidth=0.5, transform=ax.transAxes, zorder=6))

def _icon_embedding(ax, cx, cy, bw, bh):
    """Embedding vector - horizontal coloured bars."""
    iw, ih = bw * 0.52, bh * 0.18
    x0 = cx - iw / 2; y0 = cy + bh * 0.16
    cmap = plt.get_cmap("RdYlBu_r")
    vals = np.array([0.85, 0.32, 0.71, 0.48, 0.90, 0.15, 0.63, 0.44])
    pw = iw / len(vals)
    for i, v in enumerate(vals):
        ax.add_patch(mpatches.Rectangle(
            (x0 + i * pw, y0), pw * 0.85, ih,
            facecolor=cmap(v), edgecolor="none",
            transform=ax.transAxes, zorder=5))
    ax.add_patch(mpatches.Rectangle(
        (x0, y0), iw, ih, facecolor="none",
        edgecolor="#666", linewidth=0.4, transform=ax.transAxes, zorder=6))
    ax.text(cx, y0 + ih + bh * 0.01, "512-d", transform=ax.transAxes,
            fontsize=4.5, ha="center", va="bottom", color="#555", zorder=6)

def _icon_umap(ax, cx, cy, bw, bh):
    """UMAP / PCA - coloured scatter clusters."""
    iw, ih = bw * 0.50, bh * 0.36
    x0 = cx; y0 = cy + bh * 0.08
    clusters = [
        ([-0.22, -0.18, -0.20, -0.24], [0.10, 0.14, 0.08, 0.12], "#d62728"),
        ([0.06,  0.11,  0.08,  0.04],  [0.10, 0.06, 0.14, 0.08], "#2166ac"),
        ([-0.06, 0.00, -0.08, -0.02],  [-0.12, -0.08, -0.14, -0.10], "#2ca02c"),
    ]
    r_dot = bw * 0.020
    for xs, ys, col in clusters:
        for x, y in zip(xs, ys):
            ax.add_patch(mpatches.Circle(
                (x0 + x * iw, y0 + y * ih), radius=r_dot,
                facecolor=col, edgecolor="white", linewidth=0.3, alpha=0.85,
                transform=ax.transAxes, zorder=5))

def _icon_smiles(ax, cx, cy, bw, bh):
    """Molecular graph - benzene ring with substituent."""
    iw, ih = bw * 0.40, bh * 0.38
    x0 = cx; y0 = cy + bh * 0.08
    n = 6; r = min(iw, ih) * 0.40
    angles = [np.pi / 2 + k * 2 * np.pi / n for k in range(n)]
    xs = [x0 + r * np.cos(a) for a in angles]
    ys = [y0 + r * np.sin(a) for a in angles]
    for k in range(n):
        ax.plot([xs[k], xs[(k + 1) % n]], [ys[k], ys[(k + 1) % n]],
                color="#444", lw=0.9, transform=ax.transAxes, zorder=4, solid_capstyle="round")
    r_a = bw * 0.022
    for x, y in zip(xs, ys):
        ax.add_patch(mpatches.Circle(
            (x, y), radius=r_a,
            facecolor="#e8c14c", edgecolor="#777", linewidth=0.3,
            transform=ax.transAxes, zorder=5))
    # side chain
    sx2, sy2 = xs[1] + r * 0.55, ys[1] + r * 0.35
    ax.plot([xs[1], sx2], [ys[1], sy2],
            color="#444", lw=0.9, transform=ax.transAxes, zorder=4)
    ax.add_patch(mpatches.Circle(
        (sx2, sy2), radius=r_a,
        facecolor="#e06060", edgecolor="#777", linewidth=0.3,
        transform=ax.transAxes, zorder=5))

def _icon_scoring(ax, cx, cy, bw, bh):
    """Bar chart (scoring)."""
    iw, ih = bw * 0.48, bh * 0.34
    x0 = cx - iw / 2; y0 = cy + bh * 0.08
    hs = [0.40, 0.65, 1.00, 0.82, 0.53]
    cs = ["#aaa", "#aaa", "#e08214", "#bbb", "#aaa"]
    pw = iw / len(hs)
    for i, (h, c) in enumerate(zip(hs, cs)):
        ax.add_patch(mpatches.Rectangle(
            (x0 + i * pw + pw * 0.10, y0), pw * 0.80, ih * h,
            facecolor=c, edgecolor="none",
            transform=ax.transAxes, zorder=5))
    ax.plot([x0, x0 + iw], [y0, y0],
            color="#666", lw=0.5, transform=ax.transAxes, zorder=6)

def _icon_retrieval(ax, cx, cy, bw, bh):
    """Ranked result list."""
    iw, ih = bw * 0.50, bh * 0.34
    x0 = cx - iw / 2; y0 = cy + bh * 0.08
    items = [("#2ca02c", 0.92), ("#2ca02c", 0.74), ("#d62728", 0.43)]
    ph_gap = ih / len(items)
    ph = ph_gap * 0.78
    for i, (col, score) in enumerate(items):
        ry = y0 + (len(items) - 1 - i) * ph_gap
        ax.add_patch(mpatches.Rectangle(
            (x0, ry), iw * 0.16, ph,
            facecolor="#ddd", edgecolor="none",
            transform=ax.transAxes, zorder=5))
        ax.add_patch(mpatches.Rectangle(
            (x0 + iw * 0.20, ry), iw * 0.80 * score, ph,
            facecolor=col, edgecolor="none", alpha=0.78,
            transform=ax.transAxes, zorder=5))

def _icon_drug(ax, cx, cy, bw, bh):
    """Drug database - stacked pages + Rx symbol."""
    iw, ih = bw * 0.42, bh * 0.34
    x0 = cx; y0 = cy + bh * 0.09
    for k in range(3):
        off = k * 0.006
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 - iw / 2 + off, y0 - ih / 2 + off), iw, ih,
            boxstyle="round,pad=0.006",
            facecolor="#f5e8f5", edgecolor="#9467bd", linewidth=0.5,
            transform=ax.transAxes, zorder=5 - k))
    ax.text(x0, y0 + 0.002, "Rx", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", color="#9467bd",
            ha="center", va="center", zorder=7)

def _icon_fusion(ax, cx, cy, bw, bh):
    """Two-stream merge circle."""
    x0 = cx; y0 = cy + bh * 0.12
    r = bw * 0.13
    ax.add_patch(mpatches.Circle(
        (x0, y0), radius=r,
        facecolor="#fff0e0", edgecolor="#e08214", linewidth=1.0,
        transform=ax.transAxes, zorder=5))
    ax.text(x0, y0, "+", transform=ax.transAxes,
            fontsize=8, fontweight="bold", color="#e08214",
            ha="center", va="center", zorder=6)

def _icon_spatial_map(ax, cx, cy, bw, bh):
    """Spatial colour map - blended gradient overlay."""
    iw, ih = bw * 0.50, bh * 0.36
    x0 = cx - iw / 2; y0 = cy + bh * 0.07
    # Gradient image: warm left, cool right
    grad = np.linspace(0, 1, 20).reshape(1, -1) * np.linspace(0.3, 1, 10).reshape(-1, 1)
    ax_in = ax.inset_axes(
        [x0, y0, iw, ih], transform=ax.transAxes)
    ax_in.imshow(grad, cmap="RdYlBu_r", aspect="auto", origin="lower")
    ax_in.axis("off")

_ICONS = {
    "msi":         _icon_msi,
    "patches":     _icon_patches,
    "resnet":      _icon_resnet,
    "transformer": _icon_transformer,
    "embedding":   _icon_embedding,
    "umap":        _icon_umap,
    "smiles":      _icon_smiles,
    "scoring":     _icon_scoring,
    "retrieval":   _icon_retrieval,
    "drug":        _icon_drug,
    "fusion":      _icon_fusion,
    "spatial_map": _icon_spatial_map,
}


# ── LAYOUT ENGINE ─────────────────────────────────────────────────────────────

def _grid_layout(steps):
    """
    Compute (cx, cy) in [0,1] axes coords for each step.
    Each step may have pos=(row, col); default is auto row-0.
    Returns dict: step_idx -> (cx, cy, bw) where bw is that row's box width.
    """
    assigned = [s.get("pos", (0, i)) for i, s in enumerate(steps)]
    rows = sorted(set(r for r, _ in assigned))
    n_rows = len(rows)

    BOX_GAP = 0.10   # vertical gap between rows (fraction of axes height)
    if n_rows == 1:
        row_cy = {rows[0]: 0.50}
    else:
        total_h = n_rows * BOX_H + (n_rows - 1) * BOX_GAP
        y_start = 0.50 + total_h / 2 - BOX_H / 2
        row_cy = {r: y_start - i * (BOX_H + BOX_GAP)
                  for i, r in enumerate(rows)}

    row_items = {}
    for idx, (r, c) in enumerate(assigned):
        row_items.setdefault(r, []).append((c, idx))

    X_MAR = 0.04; ARROW_GAP = 0.042
    layout = {}
    for r in rows:
        items = sorted(row_items[r])
        n_cols = len(items)
        avail  = 1.0 - 2 * X_MAR
        bw     = min((avail - ARROW_GAP * (n_cols - 1)) / n_cols, 0.20)
        total  = bw * n_cols + ARROW_GAP * (n_cols - 1)
        x0     = (1.0 - total) / 2
        cy     = row_cy[r]
        for j, (_, idx) in enumerate(items):
            cx = x0 + bw / 2 + j * (bw + ARROW_GAP)
            layout[idx] = (cx, cy, bw)
    return layout


# ── ARROW ROUTING ─────────────────────────────────────────────────────────────

def _draw_arrow(ax, sx, sy, dx, dy, sbw, dbw, bh):
    """Route and draw a single arrow between two boxes."""
    same_row = abs(sy - dy) < 0.04

    if same_row:
        if dx >= sx:
            xs, ys = sx + sbw / 2 + 0.003, sy
            xe, ye = dx - dbw / 2 - 0.003, dy
        else:
            xs, ys = sx - sbw / 2 - 0.003, sy
            xe, ye = dx + dbw / 2 + 0.003, dy
        conn = "arc3,rad=0"
    elif dy < sy:          # going down
        if abs(dx - sx) < 0.03:   # same column
            xs, ys = sx, sy - bh / 2 - 0.006
            xe, ye = dx, dy + bh / 2 + 0.006
            conn = "arc3,rad=0"
        elif dx > sx:              # down-right
            xs, ys = sx, sy - bh / 2 - 0.006
            xe, ye = dx - dbw / 2 - 0.003, dy
            conn = "angle,angleA=90,angleB=0,rad=0.04"
        else:                      # down-left
            xs, ys = sx, sy - bh / 2 - 0.006
            xe, ye = dx + dbw / 2 + 0.003, dy
            conn = "angle,angleA=90,angleB=180,rad=0.04"
    else:                  # going up
        if abs(dx - sx) < 0.03:
            xs, ys = sx, sy + bh / 2 + 0.006
            xe, ye = dx, dy - bh / 2 - 0.006
            conn = "arc3,rad=0"
        elif dx > sx:              # up-right
            xs, ys = sx, sy + bh / 2 + 0.006
            xe, ye = dx - dbw / 2 - 0.003, dy
            conn = "angle,angleA=-90,angleB=0,rad=0.04"
        else:                      # up-left
            xs, ys = sx, sy + bh / 2 + 0.006
            xe, ye = dx + dbw / 2 + 0.003, dy
            conn = "angle,angleA=-90,angleB=180,rad=0.04"

    ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(
                    arrowstyle="-|>", color="#4477aa",
                    lw=1.6, mutation_scale=13,
                    connectionstyle=conn,
                ), zorder=2)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

BOX_H = 0.24   # fixed box height in axes fraction

def draw_pipeline_diagram(ax, steps, title="", connections=None, **_kwargs):
    """
    Draw an illustrated, grid-based pipeline diagram.

    Parameters
    ----------
    steps : list[dict]  each with:
        label   str          — bold title text inside box
        sub     str          — small italic subtitle (optional)
        kind    str          — "data"|"model"|"output"|"eval"|"default"
        icon    str          — icon key from _ICONS (optional)
        pos     (row, col)   — grid position; default auto row-0
    title : str — panel label at top-left (e.g. "A")
    connections : list[(from_idx, to_idx)] — default: sequential
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    if title:
        ax.text(-0.01, 1.02, title, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left", color="#111")

    n = len(steps)
    if n == 0:
        return

    layout = _grid_layout(steps)

    if connections is None:
        connections = [(i, i + 1) for i in range(n - 1)]

    # ── draw arrows first (so boxes sit on top)
    for src, dst in connections:
        sx, sy, sbw = layout[src]
        dx, dy, dbw = layout[dst]
        _draw_arrow(ax, sx, sy, dx, dy, sbw, dbw, BOX_H)

    # ── draw boxes + icons
    for i, s in enumerate(steps):
        cx, cy, bw = layout[i]
        kind = s.get("kind", "default")
        fill, edge = _bc(kind)
        label = s.get("label", "")
        sub   = s.get("sub", "")
        icon  = s.get("icon")

        # Subtle drop shadow
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - bw / 2 + 0.004, cy - BOX_H / 2 - 0.005), bw, BOX_H,
            boxstyle="round,pad=0.016",
            facecolor="#bbbbbb", edgecolor="none", alpha=0.25,
            transform=ax.transAxes, zorder=2))

        # Main box
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - bw / 2, cy - BOX_H / 2), bw, BOX_H,
            boxstyle="round,pad=0.016",
            facecolor=fill, edgecolor=edge, linewidth=1.3,
            transform=ax.transAxes, zorder=3))

        # Icon
        if icon and icon in _ICONS:
            _ICONS[icon](ax, cx, cy, bw, BOX_H)
            label_y = cy - BOX_H * 0.10
            sub_y   = cy - BOX_H * 0.30
        else:
            label_y = cy + (BOX_H * 0.08 if sub else 0.0)
            sub_y   = cy - BOX_H * 0.22

        ax.text(cx, label_y, label, transform=ax.transAxes,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#1a2a3a",
                zorder=6, multialignment="center")

        if sub:
            ax.text(cx, sub_y, sub, transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=5.8, color="#445566", style="italic",
                    zorder=6, multialignment="center")
