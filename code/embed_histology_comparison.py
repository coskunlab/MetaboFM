"""
embed_histology_comparison.py
------------------------------
Second stage of the histology-comparison experiment (part of the manuscript's
H&E-comparison analysis): runs PCA/UMAP on the concatenated Stage 1 patch tokens saved by
probe_histology_comparison.py, then renders a figure comparing MetaboFM's
learned spatial structure against the registered H&E image.

IMPORTANT: a first pass (mean-pooling tokens across channels, PCA/UMAP on all
784 patches) found the dominant axis (PC1 ~73-90% variance in all 4 organs)
was a generic tissue-boundary-vs-interior gradient, not real intra-tissue
substructure -- the same "rim vs core" pattern appeared in every organ
regardless of shape, which points to a systematic artifact (CNN
padding/receptive-field effect at the tissue edge, or MALDI ionization edge
effect) rather than four independent biological findings. This version masks
out boundary-adjacent patches (eroded tissue mask) before PCA/UMAP so the
dominant axis can't just be edge-vs-core.

Must run under the base conda env — the torch_gpu env (needed for
`metaspace`/GPU inference) has a BLAS conflict that crashes matplotlib's
savefig and sklearn/numpy.linalg calls silently (exit 127, no traceback).

Usage
-----
  python embed_histology_comparison.py
"""

import json
from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats
from scipy.ndimage import binary_erosion
from sklearn.decomposition import PCA

from optical_alignment import ion_to_native_optical_crop, native_optical_crop

OUT_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
REG_DIR = METABOFM_ROOT / "outputs/optical_images/registration"  # source of optical crops
ANNOT_DIR = METABOFM_ROOT / "outputs/optical_images/annotations"
PATCH_GRID = 28
IMG_SIZE = 224
EROSION_ITERATIONS = 2  # patches to strip off the tissue boundary, automated-mask fallback only


def upscale(grid_2d: np.ndarray, H: int, W: int) -> np.ndarray:
    return np.array(Image.fromarray(grid_2d.astype(np.float32)).resize((W, H), Image.NEAREST))


def patch_grid_to_ion_resolution(grid: np.ndarray, H: int, W: int) -> np.ndarray:
    """Correctly maps a (PATCH_GRID, PATCH_GRID) grid -- which represents the
    *padded-to-square* encoder frame (side S = max(H, W)), not the raw
    (H, W) ion array -- back to true ion-grid resolution: upscale to the
    full padded square, then crop out the centered (H, W) region
    pad_to_square() added. Resizing a non-square grid's patch map directly
    to (H, W) (skipping this crop) silently squishes and misaligns the real
    content -- this is the correct inverse of pad_to_square()+resize."""
    S = max(H, W)
    up = np.array(Image.fromarray(grid.astype(np.float32)).resize((S, S), Image.NEAREST))
    top, left = (S - H) // 2, (S - W) // 2
    return up[top:top + H, left:left + W]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    H, W = img.shape
    S = max(H, W)
    top, left = (S - H) // 2, (S - W) // 2
    out = np.zeros((S, S), dtype=img.dtype)
    out[top:top + H, left:left + W] = img
    return out


def _raster_to_patch_mask(binary_ion_grid: np.ndarray, min_overlap: float = 0.5) -> np.ndarray:
    """Boolean (PATCH_GRID, PATCH_GRID) mask from a boolean/0-1 ion-grid
    array, using the same pad-to-square + resize geometry as encoding.

    ``min_overlap`` is the fraction of a patch's area the input must cover
    for that patch to count. The default (>50%) is intentionally strict for
    the tissue/interior boundary mask, where it protects against
    boundary-adjacent patches leaking in. For a small hand-drawn anatomical
    region, that same strict threshold can silently zero out every patch the
    polygon touches without ever crossing 50% in any single one -- pass a
    lower ``min_overlap`` (e.g. >0, any nonzero overlap) when rasterizing a
    region you want to score, not a boundary you want to exclude from."""
    padded = pad_to_square(binary_ion_grid.astype(np.float32))
    S = padded.shape[0]
    resized = np.array(Image.fromarray((padded * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST)) / 255.0
    block = IMG_SIZE // PATCH_GRID
    patch_mask = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)
    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            blk = resized[r * block:(r + 1) * block, c * block:(c + 1) * block]
            patch_mask[r, c] = blk.mean() > min_overlap
    return patch_mask


def build_patch_tissue_mask(summed_ion: np.ndarray) -> np.ndarray:
    """Automated fallback: threshold summed ion intensity as a tissue proxy."""
    return _raster_to_patch_mask(summed_ion > 0)


def load_annotated_tissue_patch_mask(
    organ: str, dataset_id: str, affine: np.ndarray, optical_crop, ion_shape: tuple[int, int],
) -> np.ndarray | None:
    """Rasterizes the hand-drawn `tissue` GeoJSON polygon (native-optical-crop
    pixel space) back into ion-grid space via the inverse affine, then into
    the (PATCH_GRID, PATCH_GRID) token grid. Returns None if no annotation
    exists yet for this organ/dataset (falls back to the automated mask)."""
    geojson_path = ANNOT_DIR / f"{organ}_{dataset_id}_regions.geojson"
    if not geojson_path.exists():
        return None
    fc = json.loads(geojson_path.read_text(encoding="utf-8"))
    tissue_feats = [f for f in fc["features"] if f["properties"]["classification"] == "tissue"]
    if not tissue_feats:
        return None

    inv_affine = np.linalg.inv(np.asarray(affine, dtype=float))
    h, w = ion_shape
    ion_mask = np.zeros((h, w), dtype=np.uint8)
    for feat in tissue_feats:
        crop_local = np.asarray(feat["geometry"]["coordinates"][0], dtype=float)
        global_optical = crop_local + np.array([optical_crop.x0, optical_crop.y0])
        homogeneous = np.column_stack([global_optical, np.ones(len(global_optical))])
        ion_pts = (inv_affine @ homogeneous.T).T[:, :2]
        mask_img = Image.fromarray(ion_mask, mode="L")
        ImageDraw.Draw(mask_img).polygon([tuple(pt) for pt in ion_pts], outline=1, fill=1)
        ion_mask = np.array(mask_img, dtype=np.uint8)

    exclude_feats = [f for f in fc["features"] if f["properties"]["classification"] == "uncertain_exclude"]
    for feat in exclude_feats:
        crop_local = np.asarray(feat["geometry"]["coordinates"][0], dtype=float)
        global_optical = crop_local + np.array([optical_crop.x0, optical_crop.y0])
        homogeneous = np.column_stack([global_optical, np.ones(len(global_optical))])
        ion_pts = (inv_affine @ homogeneous.T).T[:, :2]
        mask_img = Image.fromarray(ion_mask, mode="L")
        ImageDraw.Draw(mask_img).polygon([tuple(pt) for pt in ion_pts], outline=0, fill=0)
        ion_mask = np.array(mask_img, dtype=np.uint8)

    return _raster_to_patch_mask(ion_mask.astype(bool))


def load_annotated_region_patch_masks(
    organ: str, dataset_id: str, affine: np.ndarray, optical_crop, ion_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Rasterizes each `anatomical_region` GeoJSON polygon into a
    (PATCH_GRID, PATCH_GRID) token mask, keyed by region_name. Empty dict if
    no annotation / no regions exist yet."""
    geojson_path = ANNOT_DIR / f"{organ}_{dataset_id}_regions.geojson"
    if not geojson_path.exists():
        return {}
    fc = json.loads(geojson_path.read_text(encoding="utf-8"))
    region_feats = [f for f in fc["features"] if f["properties"]["classification"] == "anatomical_region"]
    if not region_feats:
        return {}

    inv_affine = np.linalg.inv(np.asarray(affine, dtype=float))
    h, w = ion_shape
    masks: dict[str, np.ndarray] = {}
    for feat in region_feats:
        name = feat["properties"]["region_name"] or "region"
        crop_local = np.asarray(feat["geometry"]["coordinates"][0], dtype=float)
        global_optical = crop_local + np.array([optical_crop.x0, optical_crop.y0])
        homogeneous = np.column_stack([global_optical, np.ones(len(global_optical))])
        ion_pts = (inv_affine @ homogeneous.T).T[:, :2]
        ion_mask = np.zeros((h, w), dtype=np.uint8)
        mask_img = Image.fromarray(ion_mask, mode="L")
        ImageDraw.Draw(mask_img).polygon([tuple(pt) for pt in ion_pts], outline=1, fill=1)
        existing = masks.get(name)
        # min_overlap=0: see _raster_to_patch_mask docstring -- a strict >50%
        # threshold can silently zero out a small annotated region.
        new_mask = _raster_to_patch_mask(np.array(mask_img, dtype=bool), min_overlap=0.0)
        masks[name] = (existing | new_mask) if existing is not None else new_mask
    return masks


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
                                channel_images: np.ndarray, channel_labels: list[str],
                                n_pcs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automated PC-vs-raw-marker comparison (see embed_ihc_histology_comparison.py
    for the full rationale): Spearman correlation between each PC's interior-token
    scores and each channel's own patch-mean intensity over the same tokens."""
    rows = []
    for ch_idx, label in enumerate(channel_labels):
        patch_means = _channel_patch_means(channel_images[ch_idx]).flatten()[interior_flat]
        for k in range(n_pcs):
            rho, p = stats.spearmanr(pc_scores[:, k], patch_means)
            rows.append({"channel_label": label, "pc": f"PC{k+1}", "spearman_rho": float(rho), "p_value": float(p)})
    table = pd.DataFrame(rows)
    best = table.loc[table.groupby("channel_label")["spearman_rho"].apply(lambda s: s.abs().idxmax())]
    best = best.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    return table, best


def process_one(tokens_path: Path):
    d = np.load(tokens_path, allow_pickle=False)
    organ = str(d["organ"])
    dataset_id = str(d["dataset_id"])
    tokens = d["concat_tokens"] if "concat_tokens" in d else d["mean_tokens"]
    H, W = int(d["H"]), int(d["W"])
    summed_ion = d["summed_ion"]
    n_matched = int(d["n_channels_matched"])

    print(f"\n=== {organ} ({dataset_id}) — {n_matched} channels, tokens {tokens.shape} ===")

    # ── load registration data (needed both for the optical panel and to map
    # hand-drawn annotations, in native-optical pixel space, back to the
    # ion/patch grid) ──────────────────────────────────────────────────────
    reg_path = REG_DIR / f"{organ}_{dataset_id}_registration_data.npz"
    optical_crop = None
    affine = None
    if reg_path.exists():
        rd = np.load(reg_path, allow_pickle=False)
        if "affine_ion_to_optical" in rd:
            affine = rd["affine_ion_to_optical"]
            optical_crop = native_optical_crop(rd["optical"], affine, (H, W))
        else:
            print("[WARN] registration data lacks affine_ion_to_optical; "
                  "rerun fix_optical_warp.py before making publication figures")
    else:
        print(f"[WARN] no registration data at {reg_path}, skipping optical panel")

    annotated_mask = None
    if optical_crop is not None and affine is not None:
        annotated_mask = load_annotated_tissue_patch_mask(organ, dataset_id, affine, optical_crop, (H, W))

    if annotated_mask is not None:
        interior_mask = annotated_mask
        n_interior = int(interior_mask.sum())
        print(f"[INFO] using hand-annotated tissue boundary: interior={n_interior} patches")
    else:
        tissue_mask = build_patch_tissue_mask(summed_ion)
        interior_mask = binary_erosion(tissue_mask, iterations=EROSION_ITERATIONS)
        n_tissue, n_interior = int(tissue_mask.sum()), int(interior_mask.sum())
        print(f"[INFO] no annotation found, falling back to automated mask: "
              f"tissue patches={n_tissue}/{PATCH_GRID**2}, "
              f"interior (eroded x{EROSION_ITERATIONS})={n_interior}")
    if n_interior < 20:
        print(f"[SKIP] too few interior patches ({n_interior})")
        return

    interior_flat = interior_mask.flatten()
    tokens_interior = tokens[interior_flat]

    pca = PCA(n_components=5, random_state=42)
    pc_interior = pca.fit_transform(tokens_interior)
    print(f"[INFO] interior-only PCA explained variance (top 5): {pca.explained_variance_ratio_}")

    # ── automated PC-vs-raw-marker comparison: does PC1 (and PC2-5) actually
    # track each channel's own raw spatial pattern, or has it been blended
    # away by concatenation across the other channels? ────────────────────
    if "channel_images" in d and "matched_mz" in d:
        channel_labels = [f"mz={mz:.4f}" for mz in d["matched_mz"]]
        corr_table, corr_best = compare_pc_to_raw_channels(
            pc_interior, interior_flat, d["channel_images"], channel_labels, n_pcs=pc_interior.shape[1]
        )
        corr_path = OUT_DIR / f"{organ}_{dataset_id}_pc_vs_raw_channel_correlation.csv"
        corr_table.to_csv(corr_path, index=False)
        print(f"[INFO] each channel's best-matching PC (Spearman |rho|, sorted):")
        for _, row in corr_best.iterrows():
            flag = "" if abs(row["spearman_rho"]) >= 0.3 else "  <- weakly tracked by any PC1-5"
            print(f"    {row['channel_label']:<14s} best={row['pc']} rho={row['spearman_rho']:+.3f}{flag}")
        print(f"[INFO] full PC x channel correlation table saved -> {corr_path}")

    # ── which channels/metabolites drive PC1? ─────────────────────────────
    # tokens are channel-blocks concatenated in matched_channel_idx order
    # (see probe_histology_comparison.py::match_trained_channels), each block
    # 256-dim (Stage 1 embed_dim) -- rank channels by their PC1 loading norm.
    if "matched_channel_idx" in d and "matched_mz" in d:
        matched_channel_idx = d["matched_channel_idx"]
        matched_mz = d["matched_mz"]
        n_ch = len(matched_channel_idx)
        embed_dim = tokens.shape[1] // n_ch
        loadings = pca.components_[0].reshape(n_ch, embed_dim)
        loading_norm = np.linalg.norm(loadings, axis=1)
        order = np.argsort(loading_norm)[::-1]
        top_k = min(8, n_ch)
        print(f"[INFO] top {top_k} channels driving PC1 (by loading norm):")
        loadings_path = OUT_DIR / f"{organ}_{dataset_id}_pc1_channel_loadings.csv"
        import csv
        with open(loadings_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "channel_idx", "mz", "loading_norm"])
            for rank, idx in enumerate(order):
                w.writerow([rank, int(matched_channel_idx[idx]), float(matched_mz[idx]), float(loading_norm[idx])])
                if rank < top_k:
                    print(f"    #{rank}: channel_idx={int(matched_channel_idx[idx])} "
                          f"mz={float(matched_mz[idx]):.4f} loading_norm={float(loading_norm[idx]):.4f}")
        print(f"[INFO] full ranking saved -> {loadings_path}")

    pc1_full = np.full(PATCH_GRID * PATCH_GRID, np.nan, dtype=np.float32)
    pc1_full[interior_flat] = pc_interior[:, 0]
    pc1_grid = pc1_full.reshape(PATCH_GRID, PATCH_GRID)

    um1_grid = None
    try:
        import umap as umap_lib
        um_interior = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                     metric="cosine", random_state=42).fit_transform(tokens_interior)
        um1_full = np.full(PATCH_GRID * PATCH_GRID, np.nan, dtype=np.float32)
        um1_full[interior_flat] = um_interior[:, 0]
        um1_grid = um1_full.reshape(PATCH_GRID, PATCH_GRID)
    except ImportError:
        print("[WARN] umap-learn not available, skipping UMAP map")

    def _norm_and_upscale(grid):
        valid = grid[~np.isnan(grid)]
        vmin, vmax = valid.min(), valid.max()
        norm = (grid - vmin) / (vmax - vmin + 1e-8)
        up = patch_grid_to_ion_resolution(norm, H, W)
        mask_up = patch_grid_to_ion_resolution((~np.isnan(grid)).astype(np.float32), H, W) > 0.5
        return np.ma.masked_where(~mask_up, up)

    pc1_masked = _norm_and_upscale(pc1_grid)
    um1_masked = _norm_and_upscale(um1_grid) if um1_grid is not None else None

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")

    # ── figure ─────────────────────────────────────────────────────────────
    n_panels = 2 + (1 if optical_crop is not None else 0) + (1 if um1_masked is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    i = 0
    if optical_crop is not None:
        axes[i].imshow(optical_crop.image)
        axes[i].set_title(f"{organ} optical (H&E)\nnative optical resolution")
        axes[i].axis("off")
        i += 1

    summed_display = summed_ion
    if optical_crop is not None:
        summed_display, summed_valid = ion_to_native_optical_crop(
            summed_ion, affine, optical_crop
        )
        summed_display = np.ma.masked_where(~summed_valid, summed_display)
    axes[i].imshow(summed_display, cmap="viridis")
    axes[i].set_title(f"Summed ion intensity\n({n_matched} MSM-ranked channels)")
    axes[i].axis("off")
    i += 1

    pc1_display = pc1_masked
    if optical_crop is not None:
        pc1_values, pc1_valid = ion_to_native_optical_crop(
            pc1_masked.filled(0), affine, optical_crop,
            resample=Image.Resampling.NEAREST,
        )
        pc1_support, support_valid = ion_to_native_optical_crop(
            (~np.ma.getmaskarray(pc1_masked)).astype(np.float32), affine, optical_crop,
            resample=Image.Resampling.NEAREST,
        )
        pc1_display = np.ma.masked_where(
            ~(pc1_valid & support_valid & (pc1_support > 0.5)), pc1_values
        )
    im = axes[i].imshow(pc1_display, cmap=cmap, vmin=0, vmax=1)
    mask_source = "hand-annotated tissue" if annotated_mask is not None else "automated eroded mask"
    axes[i].set_title(f"MetaboFM Stage 1 PC1 ({mask_source})\n"
                       f"var={pca.explained_variance_ratio_[0]*100:.1f}%")
    axes[i].axis("off")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    i += 1

    if um1_masked is not None:
        um1_display = um1_masked
        if optical_crop is not None:
            um1_values, um1_valid = ion_to_native_optical_crop(
                um1_masked.filled(0), affine, optical_crop,
                resample=Image.Resampling.NEAREST,
            )
            um1_support, support_valid = ion_to_native_optical_crop(
                (~np.ma.getmaskarray(um1_masked)).astype(np.float32), affine, optical_crop,
                resample=Image.Resampling.NEAREST,
            )
            um1_display = np.ma.masked_where(
                ~(um1_valid & support_valid & (um1_support > 0.5)), um1_values
            )
        im2 = axes[i].imshow(um1_display, cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(f"MetaboFM Stage 1 UMAP-1 (interior only)\nboundary patches excluded")
        axes[i].axis("off")
        plt.colorbar(im2, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(f"MetaboFM vs. H&E — {organ} ({dataset_id}) — interior patches only")
    fig.tight_layout()
    out_path = OUT_DIR / f"{organ}_{dataset_id}_histology_comparison_interior.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DONE] saved -> {out_path}")

    # ── save the raw PC1 map in native-optical-crop pixel space, for
    # downstream quantitative comparison against pathologist annotations ──
    if optical_crop is not None:
        pc1_values, pc1_valid = ion_to_native_optical_crop(
            pc1_masked.filled(0), affine, optical_crop,
            resample=Image.Resampling.NEAREST,
        )
        pc1_support, support_valid = ion_to_native_optical_crop(
            (~np.ma.getmaskarray(pc1_masked)).astype(np.float32), affine, optical_crop,
            resample=Image.Resampling.NEAREST,
        )
        pc1_native_valid = pc1_valid & support_valid & (pc1_support > 0.5)
        map_path = OUT_DIR / f"{organ}_{dataset_id}_pc1_native_map.npz"
        np.savez(
            map_path,
            organ=organ,
            dataset_id=dataset_id,
            pc1=pc1_values.astype(np.float32),
            valid=pc1_native_valid,
            crop_height_px=optical_crop.image.shape[0],
            crop_width_px=optical_crop.image.shape[1],
            pca_explained_variance_ratio=pca.explained_variance_ratio_,
        )
        print(f"[DONE] saved raw PC1 map -> {map_path}")

        # ── token-level (patch-grid) data for statistics: the actual PCA
        # sample unit is one 8x8 patch, not an upsampled native pixel, so
        # any region-vs-outside test must run at this resolution ──────────
        region_patch_masks = load_annotated_region_patch_masks(organ, dataset_id, affine, optical_crop, (H, W))
        token_path = OUT_DIR / f"{organ}_{dataset_id}_pc1_token_level.npz"
        np.savez(
            token_path,
            organ=organ,
            dataset_id=dataset_id,
            pc1_grid=pc1_grid.astype(np.float32),  # (PATCH_GRID, PATCH_GRID), NaN outside interior
            interior_mask=interior_mask,
            region_names=np.array(list(region_patch_masks.keys())),
            region_masks=(
                np.stack(list(region_patch_masks.values()))
                if region_patch_masks else np.zeros((0, PATCH_GRID, PATCH_GRID), dtype=bool)
            ),
            pca_explained_variance_ratio=pca.explained_variance_ratio_,
        )
        print(f"[DONE] saved token-level PC1 grid ({len(region_patch_masks)} region(s)) -> {token_path}")
    else:
        print(f"[WARN] no optical crop for {organ} ({dataset_id}); skipping PC1 map export")

    # ── PC2-PC5 + raw per-channel marker maps: same diagnostic as
    # embed_ihc_histology_comparison.py -- does the joint PC1 (dominated by
    # whichever channels covary most) bury other channels' distinct spatial
    # patterns, and what does each channel's own unmixed pattern look like? ─
    if "channel_images" in d:
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
        fig2.suptitle(f"MetaboFM Stage 1 PC1-PC{n_pcs} — {organ} ({dataset_id})")
        fig2.tight_layout()
        pcs_path = OUT_DIR / f"{organ}_{dataset_id}_pc1_to_pc{n_pcs}.png"
        fig2.savefig(pcs_path, dpi=150)
        plt.close(fig2)
        print(f"[DONE] saved -> {pcs_path}")

        channel_images = d["channel_images"]
        channel_labels = [f"mz={mz:.4f}" for mz in d["matched_mz"]] if "matched_mz" in d else [
            f"ch{i}" for i in range(len(channel_images))
        ]
        n_cols = 6
        n_rows = int(np.ceil(len(channel_images) / n_cols))
        fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))
        axes3 = np.atleast_2d(axes3)
        viridis = plt.get_cmap("viridis").copy()
        viridis.set_bad(color="white")
        for i, label in enumerate(channel_labels):
            r, c = divmod(i, n_cols)
            img = channel_images[i].astype(np.float32)
            vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1
            axes3[r, c].imshow(img, cmap=viridis, vmin=0, vmax=vmax)
            axes3[r, c].set_title(label, fontsize=9)
            axes3[r, c].axis("off")
        for i in range(len(channel_labels), n_rows * n_cols):
            r, c = divmod(i, n_cols)
            axes3[r, c].axis("off")
        fig3.suptitle(f"Raw per-channel maps — {organ} ({dataset_id})")
        fig3.tight_layout()
        channels_path = OUT_DIR / f"{organ}_{dataset_id}_raw_channel_maps.png"
        fig3.savefig(channels_path, dpi=150)
        plt.close(fig3)
        print(f"[DONE] saved -> {channels_path}")


def main():
    # BrainIHC_* tokens belong to the separate MALDI-IHC pipeline (simple-scale
    # registration, own annotation format) -- handled by
    # embed_ihc_histology_comparison.py, not here; skip to avoid overwriting
    # its correctly-annotated outputs with this script's automated-mask fallback.
    tokens_files = sorted(
        p for p in OUT_DIR.glob("*_tokens_data.npz") if not p.name.startswith("BrainIHC_")
    )
    if not tokens_files:
        raise SystemExit(f"No *_tokens_data.npz files found in {OUT_DIR}")
    for p in tokens_files:
        process_one(p)


if __name__ == "__main__":
    main()
