"""
plot_figS13.py
--------------
Supplementary Figure S13: does concatenating all channels' Stage 1 tokens
before PCA blend away real per-channel structure, in the untargeted MSI
modality used throughout this study? (methodology/robustness diagnostic for
the Fig. S12 result, part of the manuscript's H&E-comparison analysis). The MALDI-IHC
counterpart of this diagnostic (curated to just the amyloid marker) is
folded into Supplementary Fig. S15 directly, as the justification for using
PC5 there, rather than repeated as its own separate figure -- see
plot_figS15.py.

Lung (METASPACE MSI, 32 MSM-ranked metabolite channels), the representative
sample used throughout this MSI block (Figs. S12, S14).

Panels:
  A  PC1-PC5 spatial maps (interior tissue only) -- if a channel's pattern
     isn't in PC1, a later component can still surface it.
  B  Raw per-channel metabolite maps (interior tissue only) -- the actual
     unmixed ground truth each channel shows.
  C  Channel x PC Spearman correlation heatmap -- the quantitative version
     of "does any PC actually track this channel," with channels that are
     weakly tracked by all of PC1-5 (|rho|<0.3) flagged.

Usage:
  python plot_figS13.py   (base conda env -- matplotlib/sklearn/scipy only)
"""

from __future__ import annotations

from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from plot_utils import set_nature_style, add_scale_bar_known_pixel_size
import embed_histology_comparison as ehc
import embed_ihc_histology_comparison as eihc

set_nature_style()

HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
REG_DIR = METABOFM_ROOT / "outputs/optical_images/registration"

OUT_DIR = METABOFM_ROOT / "outputs/figures"
PANEL_DIR = OUT_DIR / "figS13_concatenation_diagnostic"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

LUNG_ORGAN, LUNG_DATASET = "Lung", "2023-06-27_22h58m39s"
IHC_CONDITION = "alz"
WEAK_THRESHOLD = 0.3

# Native ion-grid physical pixel sizes (um/px): Lung MSI Pixel_Size (live
# METASPACE fetch); MALDI-IHC raster step, given directly (no METASPACE
# record exists for this sample).
LUNG_ION_UM_PER_PX = 35.0
IHC_ION_UM_PER_PX = 20.0


def write_caption(n_weak: int, n_total: int):
    caption = f"""\
Supplementary Figure 13 | Concatenation-across-channels diagnostic, untargeted MSI.

Lung (METASPACE MSI, {n_total} MSM-ranked metabolite channels), the representative sample used throughout the MSI block of this comparison (Supplementary Figs. S12, S14).

(a) MetaboFM Stage 1 PC1-PC5 (interior tissue tokens only, hand-annotated tissue boundary). PC1 alone need not capture every channel's pattern -- concatenating all channels before PCA finds the dominant *joint* axis of covariation, so a channel with a real but independent spatial pattern can appear in a later component instead.

(b) Raw per-channel metabolite maps (same interior tissue mask), each channel independently normalized to its own 99th percentile (colorbar) -- each channel's actual, unmixed spatial distribution, for direct comparison against (a). Curated to the best-tracked channels (proving PC1-PC5 reflects real channel structure; at least {N_TOP_TRACKED}, topped up further to a fixed row length) plus every weakly-tracked channel (the ones the diagnostic in (c) exists to surface) -- showing all {n_total} channels was impractical in one figure; (c) already covers every channel quantitatively, and every channel's individual raw map is exported alongside this figure's source data regardless of whether it is shown here. Each panel is labeled with its nonzero-pixel coverage: the {n_weak} weakly-tracked channels (m/z 322.27, 619.16, 818.60, marked SPARSE) have under 0.5% pixel coverage -- there is essentially no spatial pattern for any component to have captured in the first place.

(c) Spearman correlation (rho) between each channel's raw patch-mean intensity and each of PC1-PC5's interior-token scores, for all {n_total} channels. Channels weakly tracked by every one of PC1-PC5 (|rho|<{WEAK_THRESHOLD:g}, marked with a black outline) are channels whose distinct pattern the top-5-component summary in (a) does not capture -- visible instead only in their own raw map in (b).
"""
    (PANEL_DIR / "captions.txt").write_text(caption, encoding="utf-8")
    print("  saved captions.txt")


def save_panel(fig, stem):
    fig.suptitle("")
    path = PANEL_DIR / stem
    fig.savefig(str(path) + ".svg", bbox_inches="tight", pad_inches=0.05, dpi=DPI)
    print(f"  saved panel {stem}.svg")


def load_lung():
    d = np.load(HIST_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_tokens_data.npz", allow_pickle=False)
    tokens = d["concat_tokens"]
    H, W = int(d["H"]), int(d["W"])
    rd = np.load(REG_DIR / f"{LUNG_ORGAN}_{LUNG_DATASET}_registration_data.npz", allow_pickle=False)
    affine = rd["affine_ion_to_optical"]
    from optical_alignment import native_optical_crop
    optical_crop = native_optical_crop(rd["optical"], affine, (H, W))
    interior_mask = ehc.load_annotated_tissue_patch_mask(LUNG_ORGAN, LUNG_DATASET, affine, optical_crop, (H, W))
    interior_flat = interior_mask.flatten()
    pca = PCA(n_components=5, random_state=42)
    pc_interior = pca.fit_transform(tokens[interior_flat])
    channel_labels = [f"mz={mz:.4f}" for mz in d["matched_mz"]]
    return dict(label="Lung (MSI)", pca=pca, pc_interior=pc_interior, interior_flat=interior_flat,
                channel_images=d["channel_images"], channel_labels=channel_labels,
                H=H, W=W, ion_um_per_px=LUNG_ION_UM_PER_PX)


def load_ihc():
    d = np.load(HIST_DIR / f"BrainIHC_{IHC_CONDITION}_tokens_data.npz", allow_pickle=False)
    tokens = d["concat_tokens"]
    H, W = int(d["H"]), int(d["W"])
    he_shape = (int(d["he_height_px"]), int(d["he_width_px"]))
    interior_mask = eihc.build_tissue_patch_mask(d["tissue_border_he_yx"], he_shape, (H, W))
    interior_flat = interior_mask.flatten()
    pca = PCA(n_components=5, random_state=42)
    pc_interior = pca.fit_transform(tokens[interior_flat])
    channel_labels = [str(n) for n in d["channel_names"]]
    return dict(label="Brain MALDI-IHC (Alzheimer's model)", pca=pca, pc_interior=pc_interior,
                interior_flat=interior_flat, channel_images=d["channel_images"], channel_labels=channel_labels,
                H=H, W=W, ion_um_per_px=IHC_ION_UM_PER_PX)


def _sanitize(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in label)


def save_single(draw_fn, figsize, stem):
    """Saves one standalone image (no title, no padding) -- per-panel export
    convention, so every sub-image of a combined grid can be repositioned
    independently in PowerPoint."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.savefig(str(PANEL_DIR / stem) + ".svg", bbox_inches="tight", pad_inches=0, dpi=DPI)
    plt.close(fig)
    print(f"  saved individual panel {stem}.svg")


def panel_a(samples):
    n_pcs = 5
    fig, axes = plt.subplots(len(samples), n_pcs, figsize=(3.2 * n_pcs, 3.4 * len(samples)))
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    for row, s in enumerate(samples):
        for k in range(n_pcs):
            grid_full = np.full(ehc.PATCH_GRID * ehc.PATCH_GRID, np.nan, dtype=np.float32)
            grid_full[s["interior_flat"]] = s["pc_interior"][:, k]
            grid = grid_full.reshape(ehc.PATCH_GRID, ehc.PATCH_GRID)
            valid = ~np.isnan(grid)
            vmin, vmax = grid[valid].min(), grid[valid].max()
            norm = (grid - vmin) / (vmax - vmin + 1e-8)
            display = np.ma.masked_where(~valid, norm)
            ax = axes[row, k]
            ax.imshow(display, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"PC{k+1} (var={s['pca'].explained_variance_ratio_[k]*100:.1f}%)", fontsize=7)
            ax.axis("off")
            # patch-grid pixel size: each PATCH_GRID cell covers S/PATCH_GRID
            # native ion pixels, where S = the padded-to-square side (same
            # geometry the encoder itself sees; see embed_histology_comparison.py).
            S = max(s["H"], s["W"])
            patch_um_per_px = s["ion_um_per_px"] * S / ehc.PATCH_GRID
            add_scale_bar_known_pixel_size(
                ax, um_per_native_px=patch_um_per_px,
                native_width_px=ehc.PATCH_GRID, display_width_px=ehc.PATCH_GRID, color="black",
            )

            def _draw(ax_, display=display, patch_um_per_px=patch_um_per_px, cmap=cmap):
                ax_.imshow(display, cmap=cmap, vmin=0, vmax=1)
                ax_.axis("off")
                add_scale_bar_known_pixel_size(
                    ax_, um_per_native_px=patch_um_per_px,
                    native_width_px=ehc.PATCH_GRID, display_width_px=ehc.PATCH_GRID, color="black",
                )
            sample_stem = _sanitize(s["label"])
            save_single(_draw, (3.2, 3.2), f"figS13_panelA_{sample_stem}_PC{k+1}")
        axes[row, 0].text(-0.15, 0.5, s["label"], transform=axes[row, 0].transAxes,
                           fontsize=8, fontweight="bold", rotation=90, va="center", ha="center")
    fig.tight_layout()
    save_panel(fig, "figS13_panelA_pc1_to_pc5")
    plt.close(fig)

    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 2.2))
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb)
    cb.set_label("PC score\n(min-max normalized per component)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout()
    save_panel(fig_cb, "figS13_panelA_colorbar")
    plt.close(fig_cb)


N_TOP_TRACKED = 4  # best-tracked channels to show, proving PC1-5 reflects real channel structure


def select_curated_channels(sample, n_top: int = N_TOP_TRACKED):
    """Panel B previously showed all 32+19=51 raw channel maps -- too many
    for one printed figure. Curate down to the channels that actually carry
    the diagnostic's argument: the ``n_top`` best-tracked (proving PC1-5
    isn't disconnected from real channel structure) plus every weakly-tracked
    channel (WEAK_THRESHOLD -- exactly the ones the diagnostic exists to
    surface). Panel C's heatmap still covers every channel quantitatively;
    this only curates which get a qualitative raw-map panel. Full data for
    every channel remains available as individual SVG exports regardless.
    ``n_top`` is per-sample (not a fixed constant) so panel_b can equalize
    how many channels each sample's row shows -- see its docstring."""
    rho_matrix, labels = _channel_correlation_matrix(sample)
    best_abs = np.abs(rho_matrix).max(axis=1)
    order = np.argsort(-best_abs)
    top_idx = order[:n_top]
    weak_idx = np.where(best_abs < WEAK_THRESHOLD)[0]
    selected = sorted(set(top_idx.tolist()) | set(weak_idx.tolist()), key=lambda i: -best_abs[i])
    return selected, best_abs


def _weak_count(sample) -> int:
    rho_matrix, _ = _channel_correlation_matrix(sample)
    return int((np.abs(rho_matrix).max(axis=1) < WEAK_THRESHOLD).sum())


def panel_b(samples, stems):
    """One row per sample (not one grid per sample) -- keeps the whole
    curated set on a single, more compact figure. Each sample's number of
    weakly-tracked channels differs (Lung=3, alz=5), so a fixed
    N_TOP_TRACKED per sample would leave the shorter row with empty,
    dangling grid cells. Instead the target row length is set by
    whichever sample has the most weakly-tracked channels, and every
    other sample's top-tracked count is topped up to match -- every row
    fills completely, with no sample showing fewer real channels than it has."""
    weak_counts = [_weak_count(sample) for sample in samples]
    target_total = max(weak_counts) + N_TOP_TRACKED

    per_sample = []
    for sample, stem_suffix, weak_n in zip(samples, stems, weak_counts):
        n_top = target_total - weak_n
        all_labels = sample["channel_labels"]
        selected, best_abs = select_curated_channels(sample, n_top=n_top)
        # Flag near-empty channels explicitly -- a weakly-tracked channel with
        # almost no detected signal (low pixel coverage) has essentially
        # nothing for any PC to have missed, unlike a weakly-tracked channel
        # with dense, real, just-uncorrelated signal (see caption).
        coverage = [(sample["channel_images"][i] > 0).mean() for i in selected]
        labels = [
            f"{all_labels[i]}\n(best |rho|={best_abs[i]:.2f}, coverage={cov*100:.1f}%)"
            + ("\nSPARSE" if cov < 0.05 else "")
            for i, cov in zip(selected, coverage)
        ]
        print(f"  panel B [{sample['label']}]: showing {len(selected)}/{len(all_labels)} curated channels "
              f"(top {n_top}-tracked + {weak_n} weakly-tracked)")
        per_sample.append((sample, stem_suffix, all_labels, selected, labels))

    n_cols = max(len(labels) for _, _, _, _, labels in per_sample)
    n_rows = len(per_sample)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 3.1 * n_rows))
    axes = np.atleast_2d(axes)
    viridis = plt.get_cmap("viridis").copy()
    viridis.set_bad(color="white")

    for row, (sample, stem_suffix, all_labels, selected, labels) in enumerate(per_sample):
        channel_images = sample["channel_images"]
        S = max(sample["H"], sample["W"])  # pad_to_square side -- matches model-view convention
        for col in range(n_cols):
            if col >= len(selected):
                axes[row, col].axis("off")
                continue
            channel_idx, label = selected[col], labels[col]
            img = ehc.pad_to_square(channel_images[channel_idx].astype(np.float32))
            vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1
            axes[row, col].imshow(img, cmap=viridis, vmin=0, vmax=vmax)
            axes[row, col].set_title(label, fontsize=7)
            axes[row, col].axis("off")
            add_scale_bar_known_pixel_size(
                axes[row, col], um_per_native_px=sample["ion_um_per_px"],
                native_width_px=S, display_width_px=S,
            )

            def _draw(ax_, img=img, vmax=vmax, viridis=viridis, S=S, um=sample["ion_um_per_px"]):
                ax_.imshow(img, cmap=viridis, vmin=0, vmax=vmax)
                ax_.axis("off")
                add_scale_bar_known_pixel_size(ax_, um_per_native_px=um, native_width_px=S, display_width_px=S)
            save_single(_draw, (3.0, 3.0), f"figS13_panelB_{stem_suffix}_{_sanitize(all_labels[channel_idx])}")
        axes[row, 0].text(-0.15, 0.5, sample["label"], transform=axes[row, 0].transAxes,
                           fontsize=8, fontweight="bold", rotation=90, va="center", ha="center")

    fig.tight_layout(h_pad=3.0)
    save_panel(fig, "figS13_panelB_raw_channels")
    plt.close(fig)

    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 2.2))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap=viridis), cax=ax_cb)
    cb.set_label("Relative intensity\n(a.u., 0 to each channel's own 99th percentile)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout()
    save_panel(fig_cb, "figS13_panelB_colorbar")
    plt.close(fig_cb)


def _channel_correlation_matrix(sample):
    n_pcs = sample["pc_interior"].shape[1]
    labels = sample["channel_labels"]
    rho_matrix = np.zeros((len(labels), n_pcs))
    for i, img in enumerate(sample["channel_images"]):
        patch_means = ehc._channel_patch_means(img).flatten()[sample["interior_flat"]]
        for k in range(n_pcs):
            from scipy import stats
            rho, _ = stats.spearmanr(sample["pc_interior"][:, k], patch_means)
            rho_matrix[i, k] = rho
    return rho_matrix, labels


def panel_c(samples, stems):
    """Horizontal layout: channels along x, PC1-5 along y -- with 32
    channels for the Lung sample, a vertical (channels-as-rows) heatmap
    would be needlessly tall; transposed, it reads left-to-right like the
    rest of this figure's panels."""
    fig, axes = plt.subplots(len(samples), 1, figsize=(0.19 * max(
        len(s["channel_labels"]) for s in samples) + 1.4, 1.6 * len(samples)))
    axes = np.atleast_1d(axes)
    for ax, sample, stem in zip(axes, samples, stems):
        rho_matrix, labels = _channel_correlation_matrix(sample)
        order = np.argsort(-np.abs(rho_matrix).max(axis=1))
        rho_sorted = rho_matrix[order].T
        labels_sorted = [labels[i] for i in order]
        weak = np.abs(rho_sorted).max(axis=0) < WEAK_THRESHOLD

        im = ax.imshow(rho_sorted, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
        ax.set_yticks(range(rho_sorted.shape[0]))
        ax.set_yticklabels([f"PC{k+1}" for k in range(rho_sorted.shape[0])], fontsize=6)
        ax.set_xticks(range(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, fontsize=5, rotation=90, ha="center")
        ax.set_title(sample["label"], fontsize=7)
        for i, is_weak in enumerate(weak):
            if is_weak:
                ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, rho_sorted.shape[0],
                                            fill=False, edgecolor="black", linewidth=1.2))

        def _draw(ax_, rho_sorted=rho_sorted, labels_sorted=labels_sorted, weak=weak):
            ax_.imshow(rho_sorted, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
            ax_.set_yticks(range(rho_sorted.shape[0]))
            ax_.set_yticklabels([f"PC{k+1}" for k in range(rho_sorted.shape[0])], fontsize=6)
            ax_.set_xticks(range(len(labels_sorted)))
            ax_.set_xticklabels(labels_sorted, fontsize=5, rotation=90, ha="center")
            for i, is_weak in enumerate(weak):
                if is_weak:
                    ax_.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, rho_sorted.shape[0],
                                                 fill=False, edgecolor="black", linewidth=1.2))
        save_single(_draw, (0.19 * len(labels_sorted) + 1.0, 2.0), f"figS13_panelC_{_sanitize(sample['label'])}")
    fig.tight_layout()
    save_panel(fig, "figS13_panelC_correlation_heatmaps")
    plt.close(fig)

    fig_cb, ax_cb = plt.subplots(figsize=(1.3, 2.2))
    norm = plt.Normalize(vmin=-0.7, vmax=0.7)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=ax_cb)
    cb.set_label("Spearman rho", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig_cb.tight_layout()
    save_panel(fig_cb, "figS13_panelC_colorbar")
    plt.close(fig_cb)


def main():
    lung = load_lung()
    samples = [lung]

    panel_a(samples)
    panel_b(samples, ["lung"])
    panel_c(samples, ["lung"])

    n_weak = _weak_count(lung)
    n_total = len(lung["channel_labels"])
    write_caption(n_weak, n_total)
    print("FigS13 done.")


if __name__ == "__main__":
    main()
