"""
probe_resnet_umap.py
--------------------
Run UMAP on ResNet-18 patch tokens for any annotated MSI TIFF.

ROI discovery (looks in same folder as TIFF):
  Single ROI : <tiff_stem>.roi
  Multi ROI  : <tiff_stem>_1.roi, <tiff_stem>_2.roi, ... (all unioned → one mask)

Labels: annotated pixels = 1, everything else = 0.
Silhouette score measures whether annotated-region patch tokens are
more similar to each other than to background tokens.

Usage
-----
  python probe_resnet_umap.py --checkpoint <path> --tiff <path>
  python probe_resnet_umap.py --checkpoint <path>   # uses built-in lymph node default
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import torch
import tifffile
from roifile import ImagejRoi
from skimage.draw import polygon

sys.path.insert(0, str(Path(__file__).parent))
from models.resnet_encoder import build_ion_encoder_for_inference
from plot_utils import add_scale_bar_stretched

# Maps each sample's dataset date-time stamp (tiff_path.stem before "__") to
# its corpus sample_path, for scale-bar pixel-size lookup.
_SAMPLE_PATH_BY_DATASET_PREFIX = {
    "2021-06-30_20h06m19s": r"metaspace_images_dump\msi_fm_samples5\2021-06-30_20h06m19s_r0_c0_C22.npz",
    "2023-10-02_17h16m22s": r"metaspace_images_dump\msi_fm_samples5\2023-10-02_17h16m22s_r0_c0_C32.npz",
    "2025-12-05_00h57m15s": r"metaspace_images_dump\msi_fm_samples5\2025-12-05_00h57m15s_r0_c0_C32.npz",
    "2023-11-27_04h09m07s": r"metaspace_images_dump\msi_fm_samples5\2023-11-27_04h09m07s_r0_c0_C32.npz",
}

# ── default sample (lymph node) ───────────────────────────────────────────────
DEFAULT_TIFF = METABOFM_ROOT / "outputs/tiff_stacks_by_condition/Tumor/2021-06-30_20h06m19s__Lymph_node.tif"

IMG_SIZE   = 224
PATCH_GRID = 28
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── ROI helpers ───────────────────────────────────────────────────────────────

def discover_rois(tiff_path: Path):
    """
    Find annotation for this TIFF. Checks in order:
      1. <stem>_mask.npy  — pre-computed binary numpy mask (multi-blob support)
      2. <stem>.roi       — single ImageJ polygon ROI
      3. <stem>_1.roi … <stem>_N.roi — multiple ImageJ polygon ROIs

    Returns either:
      ("npy",  Path)        — numpy mask file
      ("roi",  list[Path])  — list of ROI files (may be empty)
    """
    d    = tiff_path.parent
    stem = tiff_path.stem
    npy = d / f"{stem}_mask.npy"
    if npy.exists():
        return ("npy", npy)
    single = d / f"{stem}.roi"
    if single.exists():
        return ("roi", [single])
    multi = sorted(d.glob(f"{stem}_*.roi"))
    return ("roi", multi)


def build_union_mask(annotation, H: int, W: int) -> np.ndarray:
    """
    Build binary mask (H, W).
    annotation: tuple returned by discover_rois — ("npy", path) or ("roi", [paths])
    """
    kind, payload = annotation
    if kind == "npy":
        mask = np.load(str(payload)).astype(bool)
        if mask.shape != (H, W):
            raise ValueError(f"Mask shape {mask.shape} != image shape ({H},{W})")
        return mask
    # ROI polygon(s)
    mask = np.zeros((H, W), dtype=bool)
    for rp in payload:
        roi = ImagejRoi.fromfile(str(rp))
        try:
            coords = roi.coordinates()
            rr, cc = polygon(coords[:, 1], coords[:, 0], shape=(H, W))
        except Exception:
            t, l, b, r = roi.top, roi.left, roi.bottom, roi.right
            rr, cc = np.mgrid[t:b, l:r]
            rr, cc = rr.ravel(), cc.ravel()
            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rr, cc = rr[valid], cc[valid]
        mask[rr, cc] = True
    return mask


# ── encoder helpers ───────────────────────────────────────────────────────────

def load_tiff_channels(tiff_path: Path) -> np.ndarray:
    """Returns (C, H, W) float32, tile_max normalised per channel."""
    with tifffile.TiffFile(str(tiff_path)) as tf:
        stack = np.stack([p.asarray() for p in tf.pages], axis=0).astype(np.float32)
    mx = stack.max(axis=(1, 2), keepdims=True)
    mx = np.where(mx == 0, 1.0, mx)
    return stack / mx


@torch.no_grad()
def extract_patch_tokens(encoder, img_224: np.ndarray) -> np.ndarray:
    """img_224: (H, W) float32 → (784, D)"""
    x = torch.from_numpy(img_224)[None, None].to(DEVICE)
    _, patches = encoder(x)
    return patches[0].cpu().numpy()


def patch_label_from_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Assigns each of the 784 patches label 1 if majority annotated, else 0."""
    ph = H / PATCH_GRID
    pw = W / PATCH_GRID
    labels = np.zeros(PATCH_GRID * PATCH_GRID, dtype=int)
    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            r0, r1 = int(r * ph), int((r + 1) * ph)
            c0, c1 = int(c * pw), int((c + 1) * pw)
            labels[r * PATCH_GRID + c] = int(mask[r0:r1, c0:c1].mean() > 0.5)
    return labels


def upscale(grid_2d: np.ndarray, H: int, W: int) -> np.ndarray:
    from PIL import Image as PILImage
    return np.array(PILImage.fromarray(grid_2d.astype(np.float32)).resize(
        (W, H), PILImage.NEAREST))


def pad_offsets(H: int, W: int) -> tuple[int, int, int]:
    """Centered pad-to-square offsets, matching dataset.py::_pad_to_square."""
    S = max(H, W)
    pad_h, pad_w = S - H, S - W
    top, left = pad_h // 2, pad_w // 2
    return S, top, left


def pad_to_square(img_2d: np.ndarray, S: int, top: int, left: int,
                   pad_value: float = 0.0) -> np.ndarray:
    """Centered zero-pad (H, W) -> (S, S), matching dataset.py::_pad_to_square."""
    H, W = img_2d.shape
    out = np.full((S, S), pad_value, dtype=img_2d.dtype)
    out[top:top + H, left:left + W] = img_2d
    return out


# ── main ──────────────────────────────────────────────────────────────────────

EXTRA_TIFFS = [
    METABOFM_ROOT / "outputs/tiff_stacks_by_condition/Cancer/2023-10-02_17h16m22s__Brain.tif",
    METABOFM_ROOT / "outputs/tiff_stacks_by_condition/Metastatic/2025-12-05_00h57m15s__Liver.tif",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint_epochXXX.pt or encoder_final.pt")
    parser.add_argument("--tiff", default=None,
                        help="Path to a specific TIFF (default: run all built-in samples)")
    parser.add_argument("--label", default=None,
                        help="Short label for plot titles (auto-derived from filename if omitted)")
    args = parser.parse_args()

    tiff_list = [Path(args.tiff)] if args.tiff else [DEFAULT_TIFF] + EXTRA_TIFFS
    for tiff_path in tiff_list:
        print(f"\n{'='*60}")
        print(f"[SAMPLE] {tiff_path.name}")
        run_sample(args.checkpoint, tiff_path, args.label)


def run_sample(checkpoint: str, tiff_path: Path, label_override: str | None = None):

    ckpt_path = Path(checkpoint)
    label     = label_override or tiff_path.stem.replace("_", " ")
    epoch_tag = ckpt_path.stem

    print(f"[INFO] Encoder  : {ckpt_path.name}")
    print(f"[INFO] TIFF     : {tiff_path.name}")

    # discover annotation
    annotation = discover_rois(tiff_path)
    kind, payload = annotation
    if kind == "npy":
        print(f"[INFO] Mask: {payload.name}  (numpy binary mask)")
    elif payload:
        print(f"[INFO] ROIs ({len(payload)}): {[r.name for r in payload]}")
    else:
        print("[WARN] No annotation found — UMAP will be unsupervised only")

    encoder = build_ion_encoder_for_inference(str(ckpt_path)).to(DEVICE)
    encoder.eval()

    stack = load_tiff_channels(tiff_path)   # (C, H, W)
    C, H, W = stack.shape
    print(f"[INFO] Stack: {C} channels × {H}×{W}")

    # Pad-to-square offsets, matching dataset.py::_pad_to_square (the actual
    # Stage 1 training/inference preprocessing) — avoids the aspect-ratio
    # distortion of a naive stretch-resize straight to 224x224.
    S, pad_top, pad_left = pad_offsets(H, W)
    print(f"[INFO] Pad-to-square: {H}x{W} -> {S}x{S} (top={pad_top}, left={pad_left})")

    # build union mask
    has_annotation = kind == "npy" or (kind == "roi" and len(payload) > 0)
    if has_annotation:
        ann_mask = build_union_mask(annotation, H, W)
        print(f"[INFO] Annotated pixels: {ann_mask.sum()} / {H*W}  ({100*ann_mask.mean():.1f}%)")
    else:
        ann_mask = None

    # extract tokens
    import torch.nn.functional as F
    from PIL import Image

    all_tokens, all_labels = [], []
    print(f"[INFO] Extracting patch tokens for {C} channels …")
    for ch_idx in range(C):
        ch_img   = stack[ch_idx]
        ch_img_S = pad_to_square(ch_img, S, pad_top, pad_left)
        pil      = Image.fromarray(ch_img_S).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img224   = np.array(pil)
        tokens = extract_patch_tokens(encoder, img224)   # (784, D)
        all_tokens.append(tokens)
        if ann_mask is not None:
            ann_mask_S = pad_to_square(ann_mask.astype(np.float32), S, pad_top, pad_left) > 0.5
            all_labels.append(patch_label_from_mask(ann_mask_S, S, S))
        if (ch_idx + 1) % 5 == 0:
            print(f"  channel {ch_idx+1}/{C}", flush=True)

    tokens_all = np.concatenate(all_tokens, axis=0)   # (C*784, D)
    if ann_mask is not None:
        labels_all = np.concatenate(all_labels, axis=0)
        n_ann = (labels_all == 1).sum()
        n_bg  = (labels_all == 0).sum()
        print(f"[INFO] Total tokens: {tokens_all.shape[0]}  annotated={n_ann}  background={n_bg}")
    else:
        labels_all = None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = ckpt_path.parent
    sample_tag = tiff_path.stem[:30]   # keep filename prefix in output name

    if labels_all is None:
        print("[INFO] No ROI — skipping per-channel scoring and top-3 figures.")
        return

    # ── Figure 3: per-channel silhouette scores ───────────────────────────────
    from sklearn.metrics import silhouette_score
    print("[INFO] Computing per-channel silhouette scores …")
    ch_scores = []
    for ch_idx in range(C):
        toks = all_tokens[ch_idx]
        labs = all_labels[ch_idx]
        n1 = (labs == 1).sum(); n0 = (labs == 0).sum()
        if n1 < 4 or n0 < 4:
            ch_scores.append(-999.0); continue
        ch_scores.append(float(silhouette_score(toks, labs, metric="cosine")))
    ch_scores = np.array(ch_scores)

    valid     = ch_scores > -999
    top3_idx  = np.argsort(ch_scores)[::-1][:3]
    print(f"  Top-3: ch={top3_idx.tolist()}  scores={ch_scores[top3_idx].round(4).tolist()}")
    print(f"  Mean (valid channels): {ch_scores[valid].mean():.4f}")

    # ── Save arrays for figure assembly ──────────────────────────────────────
    import umap as _umap_lib
    from sklearn.decomposition import PCA as _PCA

    def _spatial_maps(ch_idx):
        """Return (ion, pc1_norm, umap1_norm, pca_var) for one channel, all in
        padded-square (S, S) space to match the encoder's actual input geometry."""
        toks = all_tokens[ch_idx]
        pca_ = _PCA(n_components=2, random_state=42)
        pc_  = pca_.fit_transform(toks)
        pc1_ = (lambda g: (g - g.min()) / (g.max() - g.min() + 1e-8))(
                    upscale(pc_[:, 0].reshape(PATCH_GRID, PATCH_GRID), S, S))
        um_  = _umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                               metric="cosine", random_state=42).fit_transform(toks)
        um1_ = (lambda g: (g - g.min()) / (g.max() - g.min() + 1e-8))(
                    upscale(um_[:, 0].reshape(PATCH_GRID, PATCH_GRID), S, S))
        ion_S = pad_to_square(stack[ch_idx].astype(np.float32), S, pad_top, pad_left)
        return (ion_S,
                pc1_.astype(np.float32),
                um1_.astype(np.float32),
                pca_.explained_variance_ratio_.astype(np.float32),
                pc_.astype(np.float32),
                um_.astype(np.float32))

    print("[INFO] Computing PCA + UMAP spatial maps for top-3 channels …")
    top3_ion, top3_pc1, top3_umap1 = [], [], []
    top3_pca_var, top3_patch_pca, top3_patch_umap = [], [], []
    for ci in top3_idx:
        ion_, pc1_, um1_, var_, ppca_, pumap_ = _spatial_maps(int(ci))
        top3_ion.append(ion_);      top3_pc1.append(pc1_)
        top3_umap1.append(um1_);   top3_pca_var.append(var_)
        top3_patch_pca.append(ppca_); top3_patch_umap.append(pumap_)
        print(f"  ch {ci} done")

    best_ch = int(top3_idx[0])
    roi_mask_S = pad_to_square(ann_mask.astype(np.float32), S, pad_top, pad_left) > 0.5

    npz_path = out_dir / f"arrays_{sample_tag}_{epoch_tag}.npz"
    np.savez(str(npz_path),
             # best channel (backward compat) — all arrays in padded-square
             # (S, S) space to match the encoder's actual input geometry
             ion_image    = top3_ion[0],
             pc1_map      = top3_pc1[0],
             umap1_map    = top3_umap1[0],
             roi_mask     = roi_mask_S,
             ch_scores    = ch_scores.astype(np.float32),
             best_ch      = np.array(best_ch),
             best_score   = np.array(ch_scores[best_ch]),
             pca_var      = top3_pca_var[0],
             patch_labels = all_labels[best_ch].astype(np.int8),
             patch_pca    = top3_patch_pca[0],
             patch_umap   = top3_patch_umap[0],
             # top-3 arrays (shape: 3, H, W)
             top3_chs     = top3_idx.astype(np.int32),
             top3_scores  = ch_scores[top3_idx].astype(np.float32),
             top3_ion     = np.stack(top3_ion),
             top3_pc1     = np.stack(top3_pc1),
             top3_umap1   = np.stack(top3_umap1))
    print(f"[OK] Arrays: {npz_path.name}")

    fig, ax = plt.subplots(figsize=(max(8, C//2), 3))
    bar_col = ["#d73027" if i in top3_idx else "#4575b4" for i in range(C)]
    ax.bar(range(C), np.where(ch_scores > -999, ch_scores, 0), color=bar_col)
    ax.set_xlabel("Channel index"); ax.set_ylabel("Silhouette score (cosine)")
    ax.set_title(f"Per-channel annotated-vs-background separation — {label} — {epoch_tag}")
    for i in top3_idx:
        ax.text(i, ch_scores[i] + 0.005, str(i), ha="center", fontsize=7, color="darkred")
    plt.tight_layout()
    scores_path = out_dir / f"ch_scores_{sample_tag}_{epoch_tag}.png"
    plt.savefig(scores_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Scores: {scores_path.name}")

    # ── Figure 4: top-3 channel PCA scatter plots ─────────────────────────────
    from sklearn.decomposition import PCA
    print("[INFO] Running per-channel PCA for top-3 …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for plot_i, ch_idx in enumerate(top3_idx):
        toks = all_tokens[ch_idx]; labs = all_labels[ch_idx]
        pca = PCA(n_components=2, random_state=42)
        emb_ch = pca.fit_transform(toks)
        ax = axes[plot_i]
        for lbl, name, c in zip([0, 1], ["Background", "Annotated"], ["#4575b4", "#d73027"]):
            sel = labs == lbl
            ax.scatter(emb_ch[sel, 0], emb_ch[sel, 1], c=c, s=4, alpha=0.5, label=name)
        var = pca.explained_variance_ratio_
        ax.set_title(f"Ch {ch_idx}  (score={ch_scores[ch_idx]:.4f})", fontsize=10)
        ax.legend(markerscale=3, fontsize=7)
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    fig.suptitle(f"Top-3 channels PCA — {label} — {epoch_tag}", fontsize=12)
    plt.tight_layout()
    top3_path = out_dir / f"pca_top3ch_{sample_tag}_{epoch_tag}.png"
    plt.savefig(top3_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Top-3 PCA: {top3_path.name}")

    # ── Figure 4b: top-3 channel UMAP scatter plots ───────────────────────────
    try:
        import umap as umap_lib
    except ImportError:
        print("[ERROR] pip install umap-learn"); sys.exit(1)
    print("[INFO] Running per-channel UMAP for top-3 …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for plot_i, ch_idx in enumerate(top3_idx):
        toks = all_tokens[ch_idx]; labs = all_labels[ch_idx]
        r_sc = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                              metric="cosine", random_state=42)
        emb_ch = r_sc.fit_transform(toks)
        ax = axes[plot_i]
        for lbl, name, c in zip([0, 1], ["Background", "Annotated"], ["#4575b4", "#d73027"]):
            sel = labs == lbl
            ax.scatter(emb_ch[sel, 0], emb_ch[sel, 1], c=c, s=4, alpha=0.5, label=name)
        ax.set_title(f"Ch {ch_idx}  (score={ch_scores[ch_idx]:.4f})", fontsize=10)
        ax.legend(markerscale=3, fontsize=7)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.suptitle(f"Top-3 channels UMAP — {label} — {epoch_tag}", fontsize=12)
    plt.tight_layout()
    umap_top3_path = out_dir / f"umap_top3ch_{sample_tag}_{epoch_tag}.png"
    plt.savefig(umap_top3_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Top-3 UMAP scatter: {umap_top3_path.name}")

    # ── Figure 5: top-3 spatial maps ─────────────────────────────────────────
    print("[INFO] Building top-3 channel spatial maps …")
    ann_up_mask = upscale(
        patch_label_from_mask(ann_mask, H, W).reshape(PATCH_GRID, PATCH_GRID).astype(np.float32),
        H, W) > 0.5

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for row, ch_idx in enumerate(top3_idx):
        toks = all_tokens[ch_idx]
        r_ch = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                             metric="cosine", random_state=42)
        emb_ch  = r_ch.fit_transform(toks)
        umap1_g = emb_ch[:, 0].reshape(PATCH_GRID, PATCH_GRID)
        umap1_u = upscale(umap1_g, H, W)
        u_n     = (umap1_u - umap1_u.min()) / (umap1_u.max() - umap1_u.min() + 1e-8)
        bg_ch   = stack[ch_idx]

        axes[row, 0].imshow(bg_ch, cmap="viridis")
        axes[row, 0].set_title(f"Ch {ch_idx} ion image  (score={ch_scores[ch_idx]:.4f})", fontsize=9)
        axes[row, 0].axis("off")
        _sp = _SAMPLE_PATH_BY_DATASET_PREFIX.get(tiff_path.stem.split("__")[0])
        if _sp is not None:
            add_scale_bar_stretched(axes[row, 0], _sp, bg_ch.shape[1], bg_ch.shape[0], fontsize=6)

        im = axes[row, 1].imshow(u_n, cmap="RdBu_r", vmin=0, vmax=1)
        axes[row, 1].set_title("UMAP-1 spatial map", fontsize=9)
        axes[row, 1].axis("off")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(bg_ch, cmap="gray")
        ov = np.zeros((*bg_ch.shape, 4), dtype=np.float32)
        ov[ann_up_mask] = [1, 0, 0, 0.45]
        axes[row, 2].imshow(ov)
        axes[row, 2].set_title("Annotated ROI", fontsize=9)
        axes[row, 2].axis("off")

    fig.suptitle(f"Top-3 channel spatial maps — {label} — {epoch_tag}", fontsize=12)
    plt.tight_layout()
    top3_sp = out_dir / f"spatial_top3ch_{sample_tag}_{epoch_tag}.png"
    plt.savefig(top3_sp, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Top-3 spatial: {top3_sp.name}")

    # ── Figure 6: top-3 spatial maps (PCA) ───────────────────────────────────
    print("[INFO] Building top-3 channel PCA spatial maps …")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for row, ch_idx in enumerate(top3_idx):
        toks = all_tokens[ch_idx]
        pca_sp = PCA(n_components=2, random_state=42)
        emb_ch  = pca_sp.fit_transform(toks)
        pc1_g   = emb_ch[:, 0].reshape(PATCH_GRID, PATCH_GRID)
        pc1_u   = upscale(pc1_g, H, W)
        u_n     = (pc1_u - pc1_u.min()) / (pc1_u.max() - pc1_u.min() + 1e-8)
        bg_ch   = stack[ch_idx]
        var_r   = pca_sp.explained_variance_ratio_

        axes[row, 0].imshow(bg_ch, cmap="viridis")
        axes[row, 0].set_title(f"Ch {ch_idx} ion image  (score={ch_scores[ch_idx]:.4f})", fontsize=9)
        axes[row, 0].axis("off")
        _sp = _SAMPLE_PATH_BY_DATASET_PREFIX.get(tiff_path.stem.split("__")[0])
        if _sp is not None:
            add_scale_bar_stretched(axes[row, 0], _sp, bg_ch.shape[1], bg_ch.shape[0], fontsize=6)

        im = axes[row, 1].imshow(u_n, cmap="RdBu_r", vmin=0, vmax=1)
        axes[row, 1].set_title(f"PC1 spatial map ({var_r[0]*100:.1f}%)", fontsize=9)
        axes[row, 1].axis("off")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(bg_ch, cmap="gray")
        ov = np.zeros((*bg_ch.shape, 4), dtype=np.float32)
        ov[ann_up_mask] = [1, 0, 0, 0.45]
        axes[row, 2].imshow(ov)
        axes[row, 2].set_title("Annotated ROI", fontsize=9)
        axes[row, 2].axis("off")

    fig.suptitle(f"Top-3 channel spatial maps (PCA) — {label} — {epoch_tag}", fontsize=12)
    plt.tight_layout()
    top3_sp_pca = out_dir / f"spatial_top3ch_pca_{sample_tag}_{epoch_tag}.png"
    plt.savefig(top3_sp_pca, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Top-3 PCA spatial: {top3_sp_pca.name}")


if __name__ == "__main__":
    main()
