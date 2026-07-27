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

# ── DATASETS WITHOUT A RESOLVABLE PIXEL SIZE ───────────────────────────────────
# METASPACE submitter never filled in Pixel_Size (older 2017-2018 datasets), or
# the dataset has since been removed/renamed on METASPACE. Excluded from
# representative-sample selection so every displayed panel can carry an
# accurate scale bar -- an image we can't scale is a worse choice than one we
# can, all else equal.
EXCLUDED_DATASET_IDS = {
    "2017-03-01_11h13m38s",  # Brain -- no Pixel_Size in METASPACE metadata
    "2017-02-27_15h21m19s",  # Brain -- no Pixel_Size in METASPACE metadata
    "2017-02-24_15h04m10s",  # Brain -- no Pixel_Size in METASPACE metadata
    "2018-09-04_00h52m04s",  # Kidney -- no Pixel_Size in METASPACE metadata
    "2023-06-15_18h29m50s",  # Kidney -- dataset no longer found on METASPACE
    # All 16 "Cervix | Muscle" datasets: a single 2018-05-29 HeLa/NIH3T3
    # coculture submission batch, none of which have Pixel_Size in METASPACE
    # (verified for all 16, not just a sample). No representative image for
    # this organ can ever carry a scale bar -- listed explicitly so pool
    # construction drops this organ to zero candidates and callers fall
    # through to the next-best organ instead of silently showing an
    # unscaled panel.
    "2018-05-29_11h21m59s", "2018-05-29_11h22m28s", "2018-05-29_11h22m45s",
    "2018-05-29_11h22m56s", "2018-05-29_11h23m11s", "2018-05-29_11h23m20s",
    "2018-05-29_11h23m47s", "2018-05-29_11h24m00s", "2018-05-29_11h24m22s",
    "2018-05-29_11h25m02s", "2018-05-29_11h25m30s", "2018-05-29_11h25m58s",
    "2018-05-29_11h27m02s", "2018-05-29_11h27m19s", "2018-05-29_11h27m32s",
    "2018-05-29_11h27m48s",
    # All 24 "whole body" datasets: entirely removed/renamed on METASPACE
    # (verified for all 24 -- none resolve at all, not just missing
    # Pixel_Size). Same wholesale-gap situation as Cervix | Muscle above.
    "2024-04-10_18h09m46s", "2024-04-10_18h20m52s", "2024-04-10_18h24m30s",
    "2024-04-10_18h26m43s", "2024-04-10_18h29m55s", "2024-04-10_18h31m51s",
    "2024-04-10_18h32m21s", "2024-04-10_18h33m02s", "2024-04-10_18h35m23s",
    "2024-04-10_18h36m11s", "2024-04-10_18h37m47s", "2024-04-10_18h49m34s",
    "2024-04-12_08h53m57s", "2024-04-12_08h55m06s", "2024-04-12_08h55m13s",
    "2024-04-12_09h03m46s", "2024-04-12_09h41m36s", "2024-04-12_09h51m19s",
    "2024-04-12_09h54m11s", "2024-04-12_09h56m37s", "2024-04-12_09h58m46s",
    "2024-04-12_10h02m39s", "2024-04-12_10h02m46s", "2024-04-12_10h06m04s",
}


def _excluded(sample_path: str | Path) -> bool:
    return any(ds in str(sample_path) for ds in EXCLUDED_DATASET_IDS)


# ── REPRESENTATIVE-SAMPLE LOGGING (for scale-bar pixel-size lookup) ────────────
_REPR_SAMPLE_LOG = METABOFM_ROOT / "outputs/scale_bar_manifest.csv"

def _log_repr_sample(sample_path: str | Path) -> None:
    """Append a spatial sample used for figure display to a manifest CSV,
    so pixel-size lookups (for scale bars) only need to cover samples that
    are actually shown, not the full corpus. Best-effort; never raises."""
    try:
        _REPR_SAMPLE_LOG.parent.mkdir(parents=True, exist_ok=True)
        line = f"{sample_path}\n"
        existing = set()
        if _REPR_SAMPLE_LOG.exists():
            existing = set(_REPR_SAMPLE_LOG.read_text(encoding="utf-8").splitlines())
        if str(sample_path) not in existing:
            with open(_REPR_SAMPLE_LOG, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass

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


def _touches_border(img: np.ndarray, margin: int = 3, max_border_frac: float = 0.03) -> bool:
    """
    True if tissue signal reaches the outer `margin`-pixel frame of the image,
    i.e. the organ is cropped/cut off rather than fully contained with a clean
    zero-intensity margin on all sides.
    """
    H, W = img.shape
    m = min(margin, H // 2, W // 2)
    if m <= 0:
        return False
    border_vals = np.concatenate([
        img[:m, :].flatten(), img[-m:, :].flatten(),
        img[:, :m].flatten(), img[:, -m:].flatten(),
    ])
    if border_vals.size == 0:
        return False
    return float((border_vals > 0).mean()) > max_border_frac


def _gradient_strength(img: np.ndarray) -> float:
    """
    Max |Pearson r| between row-index/row-mean and column-index/column-mean.
    High values indicate a dominant linear intensity gradient across the
    whole image (acquisition drift / ion-suppression trend) rather than
    organ-shaped spatial structure.
    """
    H, W = img.shape
    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)
    with np.errstate(invalid="ignore"):
        r_row = np.corrcoef(np.arange(H), row_means)[0, 1]
        r_col = np.corrcoef(np.arange(W), col_means)[0, 1]
    r_row = 0.0 if np.isnan(r_row) else abs(float(r_row))
    r_col = 0.0 if np.isnan(r_col) else abs(float(r_col))
    return max(r_row, r_col)


def _n_significant_components(img: np.ndarray, min_component_frac: float = 0.05) -> int:
    """
    Number of spatially disconnected tissue pieces in the image, ignoring
    components smaller than min_component_frac of the total nonzero area
    (noise specks). Used to reject frames showing multiple separate tissue
    sections (e.g. two brain slices mounted side by side) rather than one
    single, whole organ section.
    """
    from scipy.ndimage import label
    mask = img > 0
    total = int(mask.sum())
    if total == 0:
        return 0
    labeled, n = label(mask)
    if n <= 1:
        return n
    sizes = np.bincount(labeled.ravel())[1:]  # skip background label 0
    return int(((sizes / total) >= min_component_frac).sum())


def _is_clean(img: np.ndarray, min_nonzero: float = 0.10, min_std: float = 0.02,
              min_interior_frac: float = 0.45, border_margin: int = 3,
              max_border_frac: float = 0.03, max_gradient: float = 0.5,
              max_components: int = 1) -> bool:
    """
    True if the channel image passes minimum quality thresholds.
    Rejects mostly-empty images, flat/noisy channels, channels whose signal
    is mostly a thin boundary rim rather than distributed within the tissue
    interior (min_interior_frac), channels where the organ is cropped/cut
    off at the image edge rather than fully visible with a clean background
    margin (border_margin/max_border_frac), channels dominated by a linear
    intensity gradient rather than real spatial structure (max_gradient),
    and channels showing more than max_components separate disconnected
    tissue pieces in the same frame.
    """
    std, nzf = _channel_quality(img)
    if nzf < min_nonzero or std < min_std:
        return False
    if _interior_signal_frac(img) < min_interior_frac:
        return False
    if _touches_border(img, margin=border_margin, max_border_frac=max_border_frac):
        return False
    if _gradient_strength(img) > max_gradient:
        return False
    if _n_significant_components(img) > max_components:
        return False
    return True


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
    _log_repr_sample(sample_path)
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
    _log_repr_sample(sample_path)
    img = patch[int(channel_idx)]
    return _pad_and_resize(img, size)


def pick_median_sample(
    sample_paths: Sequence[str],
    n_candidates: int = 60,
    min_nonzero: float = 0.10,
    min_std: float = 0.02,
    seed: int = 42,
) -> str | None:
    """
    Return the sample at the median spatial variance among candidates that pass
    quality filtering.  This avoids cherry-picking (not best, not worst) while
    excluding truly empty or noisy images.

    Candidates are a seeded random sample of up to n_candidates paths drawn from
    the full pool (not the first n_candidates in file order) -- sample_paths is
    typically grouped by dataset/acquisition order in the source CSV, so taking
    a plain prefix systematically under-represents platforms/analyzers that
    appear later in the file. Returns None if all fail to load.
    """
    all_paths = [sp for sp in sample_paths if not _excluded(sp)]
    if len(all_paths) > n_candidates:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_paths), size=n_candidates, replace=False)
        candidate_paths = [all_paths[i] for i in idx]
    else:
        candidate_paths = all_paths

    scored: list[tuple[float, str]] = []
    for sp in candidate_paths:
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
    picked = scored[len(scored) // 2][1]   # median
    _log_repr_sample(picked)
    return picked


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
    matches = matches[~matches["sample_path"].apply(_excluded)].reset_index(drop=True)
    if len(matches) > n_candidates:
        rng = np.random.default_rng(42)
        matches = matches.iloc[rng.choice(len(matches), size=n_candidates, replace=False)]
    candidates = list(matches.iterrows())

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
    _log_repr_sample(best_sp)
    return best_sp, best_ci


# ── SCALE BARS ──────────────────────────────────────────────────────────────────

_PIXEL_SIZE_CSV = METABOFM_ROOT / "outputs/scale_bar_pixel_sizes.csv"
_CHANNEL_DIMS_CSV = METABOFM_ROOT / "outputs/filtering/channels_v2_filtered.csv"

_pixel_size_lookup: dict | None = None
_raw_dims_lookup: dict | None = None


def _load_scale_bar_lookups() -> None:
    global _pixel_size_lookup, _raw_dims_lookup
    if _pixel_size_lookup is not None:
        return
    import pandas as pd

    _pixel_size_lookup = {}
    if _PIXEL_SIZE_CSV.exists():
        px = pd.read_csv(_PIXEL_SIZE_CSV)
        for _, row in px.iterrows():
            x, y = row.get("pixel_size_x_um"), row.get("pixel_size_y_um")
            if pd.notna(x) and pd.notna(y):
                _pixel_size_lookup[str(row["sample_path"])] = (float(x), float(y))

    _raw_dims_lookup = {}
    if _CHANNEL_DIMS_CSV.exists():
        dims = pd.read_csv(_CHANNEL_DIMS_CSV, usecols=["sample_path", "img_h", "img_w"])
        dims = dims.drop_duplicates("sample_path")
        for _, row in dims.iterrows():
            _raw_dims_lookup[str(row["sample_path"])] = (int(row["img_h"]), int(row["img_w"]))


def _nice_scalebar_length(value_um: float) -> float:
    """Snap to a visually clean 1/2/5 x 10^n length, at or just below value_um."""
    import math
    if value_um <= 0:
        return 0.0
    exp = math.floor(math.log10(value_um))
    for mult in (5, 2, 1):
        candidate = mult * (10 ** exp)
        if candidate <= value_um:
            return float(candidate)
    return float(10 ** exp)


def _draw_scale_bar_artist(
    ax, um_per_display_px: float, display_width_px: float,
    loc: str = "lower right", color: str = "white", fontsize: int = 7,
) -> bool:
    """Low-level draw given an already-computed um-per-displayed-pixel scale."""
    if um_per_display_px is None or um_per_display_px <= 0:
        return False
    field_of_view_um = um_per_display_px * display_width_px
    bar_um = _nice_scalebar_length(field_of_view_um * 0.2)
    if bar_um <= 0:
        return False
    bar_px = bar_um / um_per_display_px

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    label = f"{bar_um:g} µm"
    scalebar = AnchoredSizeBar(
        ax.transData, bar_px, label, loc,
        pad=0.4, color=color, frameon=False,
        size_vertical=display_width_px * 0.012,
        fontproperties=fm.FontProperties(size=fontsize),
    )
    ax.add_artist(scalebar)
    return True


def add_scale_bar(
    ax, sample_path: str | Path, display_size: int = 224,
    loc: str = "lower right", color: str = "white", fontsize: int = 7,
) -> bool:
    """
    Draw a physically accurate scale bar on a panel showing the standard
    model-view image (`_pad_and_resize`: zero-pad to a square of side
    `S = max(img_h, img_w)`, then resize to `display_size` x `display_size`).
    The effective pixel size of the *displayed* image is
    `pixel_size_um * (S / display_size)`, not the raw acquisition's pixel size.

    Silently does nothing (returns False) if pixel size or raw dimensions
    are unavailable for this sample -- an omitted scale bar is preferable to
    a wrong one.
    """
    _load_scale_bar_lookups()
    sp = str(sample_path)

    px = _pixel_size_lookup.get(sp)
    dims = _raw_dims_lookup.get(sp)
    if px is None or dims is None:
        return False

    px_x, px_y = px
    raw_h, raw_w = dims
    S = max(raw_h, raw_w)
    um_per_display_px = ((px_x + px_y) / 2.0) * (S / display_size)
    return _draw_scale_bar_artist(ax, um_per_display_px, display_size,
                                   loc=loc, color=color, fontsize=fontsize)


def add_scale_bar_stretched(
    ax, sample_path: str | Path, display_width: int, display_height: int,
    loc: str = "lower right", color: str = "white", fontsize: int = 7,
) -> bool:
    """
    Like `add_scale_bar`, but for panels that resize the raw image directly
    to (display_height, display_width) without square-padding first (i.e. an
    anisotropic stretch, not the standard model-view pipeline). Uses the
    horizontal axis's effective pixel size for the bar.

    Silently does nothing (returns False) if pixel size or raw dimensions
    are unavailable for this sample.
    """
    _load_scale_bar_lookups()
    sp = str(sample_path)

    px = _pixel_size_lookup.get(sp)
    dims = _raw_dims_lookup.get(sp)
    if px is None or dims is None:
        return False

    px_x, _px_y = px
    _raw_h, raw_w = dims
    um_per_display_px = px_x * (raw_w / display_width)
    return _draw_scale_bar_artist(ax, um_per_display_px, display_width,
                                   loc=loc, color=color, fontsize=fontsize)


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
