"""
embed_ihc_histology_comparison.py
------------------------------
PCA on the concatenated Stage 1 tokens saved by probe_ihc_histology_comparison.py
(the MALDI-IHC mouse brain dataset), restricted to the hand-drawn tissue
boundary (tissue_border.csv), then a figure comparing MetaboFM's learned
spatial structure against the registered H&E image and reporting which
protein markers drive PC1 by name (no METASPACE lookup needed here).

Registration is a plain uniform scale between the MALDI-IHC grid (H, W) and
the full-resolution H&E frame (he_height_px, he_width_px) -- MAGIC's own
affine step already resolved rotation/translation before export.

Must run under the base conda env (matplotlib/sklearn; this machine's
torch_gpu env crashes on those calls).

Usage
-----
  python embed_ihc_histology_comparison.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from metabofm_paths import METABOFM_ROOT, IHC_RAW_DIR

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import roifile
import tifffile
from PIL import Image, ImageDraw
from scipy import stats
from sklearn.decomposition import PCA

from embed_histology_comparison import PATCH_GRID, IMG_SIZE, pad_to_square, _raster_to_patch_mask

DATA_ROOT = IHC_RAW_DIR
OUT_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
CONDITIONS = ["alz", "wt"]
MIN_INTERIOR_TOKENS = 20


def build_tissue_patch_mask(tissue_border_he_yx: np.ndarray, he_shape: tuple[int, int],
                             ion_shape: tuple[int, int]) -> np.ndarray:
    """Rasterizes the H&E-pixel-space tissue polygon down into ion-grid
    space (uniform scale, no rotation) and then into the (PATCH_GRID,
    PATCH_GRID) token grid."""
    Hh, Wh = he_shape
    H, W = ion_shape
    scale_y, scale_x = H / Hh, W / Wh
    ion_yx = tissue_border_he_yx * np.array([scale_y, scale_x])
    ion_xy = ion_yx[:, ::-1]  # PIL wants (x, y)

    mask_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask_img).polygon([tuple(pt) for pt in ion_xy], outline=1, fill=1)
    ion_mask = np.array(mask_img, dtype=bool)
    return _raster_to_patch_mask(ion_mask)


def _channel_patch_means(img_hw: np.ndarray) -> np.ndarray:
    """Mean raw intensity per (PATCH_GRID, PATCH_GRID) token, using the same
    pad-to-square + resize-to-224 geometry as encoding (BOX resize = an
    actual local average, appropriate for a raw-intensity summary)."""
    padded = pad_to_square(img_hw.astype(np.float32))
    resized = np.asarray(
        Image.fromarray(padded).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BOX),
        dtype=np.float32,
    )
    block = IMG_SIZE // PATCH_GRID
    return resized.reshape(PATCH_GRID, block, PATCH_GRID, block).mean(axis=(1, 3))


def compare_pc_to_raw_channels(pc_scores: np.ndarray, interior_flat: np.ndarray,
                                channel_images: np.ndarray, channel_names: list[str],
                                n_pcs: int) -> pd.DataFrame:
    """Automated PC-vs-raw-marker comparison: Spearman correlation between
    each PC's interior-token scores and each channel's own patch-mean
    intensity over the same tokens. Answers "does PC1 (and later PCs)
    actually track this channel's raw pattern, or has it been blended away
    by the other channels in the concatenation?" without eyeballing."""
    rows = []
    for ch_idx, name in enumerate(channel_names):
        patch_means = _channel_patch_means(channel_images[ch_idx]).flatten()[interior_flat]
        for k in range(n_pcs):
            rho, p = stats.spearmanr(pc_scores[:, k], patch_means)
            rows.append({
                "channel_name": name,
                "pc": f"PC{k+1}",
                "spearman_rho": float(rho),
                "p_value": float(p),
            })
    table = pd.DataFrame(rows)
    # for each channel, which PC (if any) best tracks its raw pattern
    best = table.loc[table.groupby("channel_name")["spearman_rho"].apply(lambda s: s.abs().idxmax())]
    best = best.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    return table, best


def load_region_patch_masks(condition: str, he_shape: tuple[int, int],
                             ion_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Rasterizes each hand-drawn blind-H&E {condition}_roi*.roi (full-res H&E
    pixel coordinates, drawn independently of any MALDI-IHC channel) down into
    the (PATCH_GRID, PATCH_GRID) token grid, keyed by ROI filename stem."""
    Hh, Wh = he_shape
    H, W = ion_shape
    scale_y, scale_x = H / Hh, W / Wh

    masks = {}
    for roi_path in sorted((DATA_ROOT / condition).glob(f"{condition}_roi*.roi")):
        roi = roifile.ImagejRoi.fromfile(str(roi_path))
        he_xy = roi.coordinates().astype(np.float64)  # (x, y) = (col, row), H&E pixels
        ion_xy = he_xy * np.array([scale_x, scale_y])
        mask_img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask_img).polygon([tuple(pt) for pt in ion_xy], outline=1, fill=1)
        # min_overlap=0: any nonzero overlap counts a patch as "in the region".
        # The default >50% threshold (right for the tissue/interior boundary
        # mask) silently zeroes out small hand-drawn regions that touch many
        # patches without covering >50% of any single one.
        masks[roi_path.stem] = _raster_to_patch_mask(np.array(mask_img, dtype=bool), min_overlap=0.0)
    return masks


def process_one(tokens_path: Path):
    d = np.load(tokens_path, allow_pickle=False)
    condition = str(d["dataset_id"])
    tokens = d["concat_tokens"]
    H, W = int(d["H"]), int(d["W"])
    channel_names = [str(n) for n in d["channel_names"]]
    n_ch = len(channel_names)

    print(f"\n=== Brain IHC ({condition}) — {n_ch} channels, tokens {tokens.shape} ===")

    he_shape = (int(d["he_height_px"]), int(d["he_width_px"]))
    interior_mask = build_tissue_patch_mask(d["tissue_border_he_yx"], he_shape, (H, W))
    n_interior = int(interior_mask.sum())
    print(f"[INFO] hand-annotated tissue boundary: interior={n_interior} patches")
    if n_interior < MIN_INTERIOR_TOKENS:
        print(f"[SKIP] too few interior patches ({n_interior})")
        return

    interior_flat = interior_mask.flatten()
    tokens_interior = tokens[interior_flat]

    pca = PCA(n_components=5, random_state=42)
    pc_interior = pca.fit_transform(tokens_interior)
    print(f"[INFO] interior-only PCA explained variance (top 5): {pca.explained_variance_ratio_}")

    # ── automated PC-vs-raw-marker comparison: does PC1 (and PC2-5) actually
    # track each channel's own raw spatial pattern, or has it been blended
    # away by concatenation across the other 18 channels? ─────────────────
    channel_images = d["channel_images"]  # (n_ch, H, W)
    corr_table, corr_best = compare_pc_to_raw_channels(
        pc_interior, interior_flat, channel_images, channel_names, n_pcs=pc_interior.shape[1]
    )
    corr_path = OUT_DIR / f"BrainIHC_{condition}_pc_vs_raw_channel_correlation.csv"
    corr_table.to_csv(corr_path, index=False)
    print(f"[INFO] each channel's best-matching PC (Spearman |rho|, sorted):")
    for _, row in corr_best.iterrows():
        flag = "" if abs(row["spearman_rho"]) >= 0.3 else "  <- weakly tracked by any PC1-5"
        print(f"    {row['channel_name']:<28s} best={row['pc']} rho={row['spearman_rho']:+.3f}{flag}")
    print(f"[INFO] full PC x channel correlation table saved -> {corr_path}")

    embed_dim = tokens.shape[1] // n_ch
    loadings = pca.components_[0].reshape(n_ch, embed_dim)
    loading_norm = np.linalg.norm(loadings, axis=1)
    order = np.argsort(loading_norm)[::-1]
    print(f"[INFO] channels driving PC1 (by loading norm, all {n_ch}):")
    loadings_path = OUT_DIR / f"BrainIHC_{condition}_pc1_channel_loadings.csv"
    with open(loadings_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "channel_name", "loading_norm"])
        for rank, idx in enumerate(order):
            w.writerow([rank, channel_names[idx], float(loading_norm[idx])])
            if rank < 8:
                print(f"    #{rank}: {channel_names[idx]} loading_norm={float(loading_norm[idx]):.4f}")
    print(f"[INFO] full ranking saved -> {loadings_path}")

    pc1_full = np.full(PATCH_GRID * PATCH_GRID, np.nan, dtype=np.float32)
    pc1_full[interior_flat] = pc_interior[:, 0]
    pc1_grid = pc1_full.reshape(PATCH_GRID, PATCH_GRID)

    # ── blind-H&E anatomical regions vs. rest-of-tissue: token-level test ──
    region_masks = load_region_patch_masks(condition, he_shape, (H, W))
    region_rows = []
    if region_masks:
        valid = ~np.isnan(pc1_grid) & interior_mask
        in_any_region = np.zeros_like(valid)
        for rmask in region_masks.values():
            in_any_region |= rmask
        outside_mask = valid & ~in_any_region
        outside_vals = pc1_grid[outside_mask]
        for name, rmask in region_masks.items():
            region_valid = rmask & valid
            n_valid = int(region_valid.sum())
            region_vals = pc1_grid[region_valid]
            row = {
                "condition": condition, "region_name": name,
                "n_region_tokens_total": int(rmask.sum()),
                "n_region_tokens_valid": n_valid,
                "n_outside_tokens_valid": int(outside_vals.size),
            }
            if n_valid >= 2 and outside_vals.size >= 2:
                _, p_value = stats.mannwhitneyu(region_vals, outside_vals, alternative="two-sided")
                row.update({
                    "region_pc1_mean": float(region_vals.mean()),
                    "region_pc1_median": float(np.median(region_vals)),
                    "outside_pc1_mean": float(outside_vals.mean()),
                    "outside_pc1_median": float(np.median(outside_vals)),
                    "mannwhitney_p": float(p_value),
                })
            else:
                print(f"[WARN] {name}: too few valid tokens (region={n_valid}, outside={outside_vals.size})")
            region_rows.append(row)
        region_table = pd.DataFrame(region_rows)
        region_csv = OUT_DIR / f"BrainIHC_{condition}_annotation_vs_pc1.csv"
        region_table.to_csv(region_csv, index=False)
        print(f"[INFO] {len(region_masks)} blind-H&E region(s) vs. PC1:")
        print(region_table.to_string(index=False))
        print(f"[DONE] saved -> {region_csv}")
    else:
        print(f"[INFO] no {condition}_roi*.roi files found; skipping region-vs-PC1 test")

    # ── save token-level data for downstream statistics ───────────────────
    token_path = OUT_DIR / f"BrainIHC_{condition}_pc1_token_level.npz"
    np.savez(
        token_path,
        organ="BrainIHC", dataset_id=condition,
        pc1_grid=pc1_grid.astype(np.float32),
        interior_mask=interior_mask,
        region_names=np.array(list(region_masks.keys())),
        region_masks=(
            np.stack(list(region_masks.values())) if region_masks
            else np.zeros((0, PATCH_GRID, PATCH_GRID), dtype=bool)
        ),
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )
    print(f"[DONE] saved token-level PC1 grid -> {token_path}")

    # ── figure: H&E (downsized) + PC1 map upsampled to the same frame ─────
    he_path = str(d["he_path"])
    he_image = tifffile.imread(he_path)  # (Hh, Wh, 3)
    # Aspect-ratio-preserving thumbnail: capping each dimension to 1600
    # independently (the old behaviour) forces a square canvas regardless of
    # the source aspect ratio, visibly stretching/squishing every non-square
    # H&E image (both alz and wt are wider than tall).
    _thumb_scale = min(1.0, 1600 / max(he_shape))
    he_thumb = np.array(Image.fromarray(he_image).resize(
        (max(1, int(round(he_shape[1] * _thumb_scale))), max(1, int(round(he_shape[0] * _thumb_scale)))),
        Image.BILINEAR))

    def _norm_and_upscale(grid, out_hw):
        valid = grid[~np.isnan(grid)]
        vmin, vmax = valid.min(), valid.max()
        norm = (grid - vmin) / (vmax - vmin + 1e-8)
        # Undo pad_to_square's centered padding before resizing to the
        # (non-square) display frame -- resizing the padded-square patch
        # grid directly to (H, W)-derived dimensions silently squishes it.
        from embed_histology_comparison import patch_grid_to_ion_resolution
        ion_res = patch_grid_to_ion_resolution(norm, H, W)
        ion_valid = patch_grid_to_ion_resolution((~np.isnan(grid)).astype(np.float32), H, W) > 0.5
        up = np.array(Image.fromarray(ion_res).resize((out_hw[1], out_hw[0]), Image.NEAREST))
        mask_up = np.array(Image.fromarray(ion_valid.astype(np.uint8) * 255).resize(
            (out_hw[1], out_hw[0]), Image.NEAREST)) > 127
        return np.ma.masked_where(~mask_up, up)

    pc1_display = _norm_and_upscale(pc1_grid, he_thumb.shape[:2])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(he_thumb)
    axes[0].set_title(f"Brain ({condition}) H&E")
    axes[0].axis("off")

    im = axes[1].imshow(pc1_display, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f"MetaboFM Stage 1 PC1 (MALDI-IHC, hand-annotated tissue)\n"
                       f"var={pca.explained_variance_ratio_[0]*100:.1f}%")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    for rmask in region_masks.values():
        axes[1].contour(rmask, levels=[0.5], colors="lime", linewidths=1.2)

    fig.suptitle(f"MetaboFM vs. H&E — Brain IHC ({condition})")
    fig.tight_layout()
    out_path = OUT_DIR / f"BrainIHC_{condition}_histology_comparison_interior.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] saved -> {out_path}")

    # ── PC2-PC5: does the joint PC1 (dominated by whichever channels covary
    # most) bury other channels' distinct spatial patterns? Later components
    # can surface structure PC1 doesn't capture ────────────────────────────
    n_pcs = min(5, pc_interior.shape[1])
    fig2, axes2 = plt.subplots(1, n_pcs, figsize=(4.2 * n_pcs, 4.5))
    if n_pcs == 1:
        axes2 = [axes2]
    for k in range(n_pcs):
        pck_full = np.full(PATCH_GRID * PATCH_GRID, np.nan, dtype=np.float32)
        pck_full[interior_flat] = pc_interior[:, k]
        pck_grid = pck_full.reshape(PATCH_GRID, PATCH_GRID)
        valid_k = ~np.isnan(pck_grid)
        vmin_k, vmax_k = pck_grid[valid_k].min(), pck_grid[valid_k].max()
        pck_norm = (pck_grid - vmin_k) / (vmax_k - vmin_k + 1e-8)
        pck_display = np.ma.masked_where(~valid_k, pck_norm)
        im_k = axes2[k].imshow(pck_display, cmap=cmap, vmin=0, vmax=1)
        axes2[k].set_title(f"PC{k+1} (var={pca.explained_variance_ratio_[k]*100:.1f}%)")
        axes2[k].axis("off")
        plt.colorbar(im_k, ax=axes2[k], fraction=0.046, pad=0.04)
    fig2.suptitle(f"MetaboFM Stage 1 PC1-PC{n_pcs} — Brain IHC ({condition})")
    fig2.tight_layout()
    pcs_path = OUT_DIR / f"BrainIHC_{condition}_pc1_to_pc{n_pcs}.png"
    fig2.savefig(pcs_path, dpi=150)
    plt.close(fig2)
    print(f"[DONE] saved -> {pcs_path}")

    # ── raw per-channel marker maps: the unmixed ground truth each channel
    # actually shows, for comparison against the blended joint PC1 ────────
    n_cols = 5
    n_rows = int(np.ceil(n_ch / n_cols))
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    axes3 = np.atleast_2d(axes3)
    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad(color="white")
    for i, name in enumerate(channel_names):
        r, c = divmod(i, n_cols)
        img = channel_images[i].astype(np.float32)
        img = np.ma.masked_where(~np.array(Image.fromarray(interior_mask.astype(np.uint8) * 255).resize(
            (W, H), Image.NEAREST)).astype(bool), img)
        vmax = np.percentile(channel_images[i][channel_images[i] > 0], 99) if (channel_images[i] > 0).any() else 1
        axes3[r, c].imshow(img, cmap=viridis, vmin=0, vmax=vmax)
        axes3[r, c].set_title(name, fontsize=9)
        axes3[r, c].axis("off")
    for i in range(n_ch, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes3[r, c].axis("off")
    fig3.suptitle(f"Raw per-channel marker maps (interior tissue only) — Brain IHC ({condition})")
    fig3.tight_layout()
    channels_path = OUT_DIR / f"BrainIHC_{condition}_raw_channel_maps.png"
    fig3.savefig(channels_path, dpi=150)
    plt.close(fig3)
    print(f"[DONE] saved -> {channels_path}")


def main():
    for condition in CONDITIONS:
        tokens_path = OUT_DIR / f"BrainIHC_{condition}_tokens_data.npz"
        if not tokens_path.exists():
            print(f"[SKIP] {condition}: no tokens data, run probe_ihc_histology_comparison.py first")
            continue
        process_one(tokens_path)


if __name__ == "__main__":
    main()
