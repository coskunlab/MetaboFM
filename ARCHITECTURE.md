# MetaboFM Architecture

## Overview

MetaboFM is a two-stage hierarchical foundation model for mass spectrometry imaging (MSI) data. It produces three levels of embeddings from a single forward pass:

- **Patch-level**: 28×28 spatially-resolved embeddings per ion channel (256-dim per patch location)
- **Channel-level**: per-ion embeddings capturing molecular identity and spatial distribution (256-dim Stage 1 / 512-dim Stage 2 `channel_refined`)
- **Sample-level**: whole-sample embeddings capturing global metabolic phenotype (512-dim Stage 2 `sample_cls`)

The architecture addresses the core constraints of public MSI data: variable number of detected ions per sample, no uniform m/z axis, channels ordered by MSM score rather than m/z, and highly variable tissue types and organisms.

The manuscript refers to Stage 2 as "the cross-channel aggregation Transformer" in all reader-facing text (captions, figures, Methods prose) — never by the internal class name `ChannelAggregator` below. Keep that distinction when writing anything that ends up in the paper.

---

## Architecture

### Stage 1 — ResNet-18 Ion Image Encoder (`models/resnet_encoder.py`)

A ResNet-18 encoder (trained from scratch, 1-channel input) operating on **single-channel ion images**. Each ion image (one m/z) is treated as an independent 2D grayscale signal — no channel-shuffle augmentation is needed because channels are never batched together at this stage.

```
Input:  (1, H, W)  — one ion image, single channel
Resize: (1, 224, 224)

Encoder (ResNet-18 backbone):
  stem   → 112×112 × 64
  layer1 → 56×56 × 64
  layer2 → 28×28 × 128   ← patch-token source
  layer3 → 14×14 × 256
  layer4 → 7×7  × 512    ← CLS-token source (after global average pool)

Both branches are linearly projected to EMBED_DIM = 256:
  CLS token    : layer4 → global-avg-pool (512,) → Linear(512, 256)   → (256,)
  Patch tokens : layer2 (28×28, 128) → Linear(128, 256), per position → (784, 256)
```

Pretrained with the **Barlow Twins** self-supervised objective (not masked reconstruction). Two augmented
views of each ion image (random resized crop, flips, intensity jitter, optional Gaussian blur) are passed
through the shared encoder and a separate 3-layer MLP projector (256 → 2048 → 2048 → 2048, discarded after
pretraining); the loss drives the cross-correlation matrix between the two projected views toward the
identity matrix. Two auxiliary losses supervise the patch tokens directly:
- **Spatial coherence loss** — penalizes low cosine similarity between horizontally/vertically adjacent
  patch tokens, encouraging smooth spatial variation.
- **Patch-level Barlow Twins loss** — a separate 3-layer MLP projector (256 → 512) applied to all 784 patch
  tokens, redundancy-reducing across spatial locations.

Total loss: `L = L_BT + α_sp · L_spatial + α_patch · L_patch-BT`, with `α_sp = α_patch = 0.1`.

Single-channel input means:
- No channel-ordering problem — each ion image processed independently
- Patch tokens unambiguously encode spatial position (channel permutation invariance is handled entirely by
  Stage 2, not by augmenting Stage 1 to be order-invariant)
- Weights shared across all ion images (same encoder for every m/z)

### Stage 2 — `ChannelAggregator` Transformer (`models/channel_aggregator.py`)

A Transformer that operates on the **set of Stage 1 CLS tokens** for all channels in an MSI sample.
Permutation-invariant by design (no fixed positional encoding over the channel sequence) — channel order is
instead informed by a **learned m/z embedding**, not a sinusoidal one.

```
Input:  C Stage 1 CLS tokens          → (B, C, 256)
        + per-channel m/z scalar      → MLP(1 → 128 → 64) → (B, C, 64)
        concatenated                  → (B, C, 320) → Linear → (B, C, 512)
        + prepended learnable [AGG_CLS] token → (B, C+1, 512)

Transformer: 4 layers, d_model=512, 8 heads, FFN dim 2048, GELU, pre-norm,
             dropout 0.1, key_padding_mask over unfilled channel slots

Output:
  - sample_cls       : (512,)      ← [AGG_CLS] token, projected  = z_sample
  - channel_refined   : (C, 512)   ← per-channel tokens, projected = z_chan^refined
```

Pretrained with a **masked channel prediction** objective: mask ~40% of channel tokens
(`MASK_RATIO = 0.40`), and predict each masked position's original Stage 1 CLS embedding from the
unmasked context via a small head (Linear → GELU → LayerNorm → Linear). Loss is **cosine embedding loss**
on the masked positions only, analogous to a BERT masked-language-model objective but over ion channels
instead of tokens. Stage 1 weights are frozen during Stage 2 training.

`sample_cls` (`z_sample`) and `channel_refined` (`z_chan^refined`) are used for different benchmark families —
see the manuscript Methods, "MSI representation variants," before assuming one can substitute for the other.

---

## Why This Design

### Problem with the v1 single-model approach

v1 fed the full `(C, H, W)` ion image stack into a single ViT-MAE. This required:
1. **Channel shuffle augmentation** — because channels are ordered by MSM score (not m/z), which varies across MSI samples. Without shuffling the model memorises ordering shortcuts.
2. **Keep-one-channel augmentation** — to make single-channel inference in-distribution.

Both augmentations destroy the spatial structure of patch tokens. The v1 model learned permutation invariance at the cost of spatial coherence — patch tokens carried no reliable spatial signal.

### Why separating the axes works

| Property | Spatial (H, W) — Stage 1 | Spectral (channels) — Stage 2 |
|---|---|---|
| Ordering | Fixed, meaningful | Variable, MSM-ranked |
| Dimensionality | Fixed (224×224 after resize) | Variable (1–32+) |
| Permutation invariance needed | No | Yes |
| Barlow Twins objective meaningful | Yes (spatial augmentations only) | N/A — Stage 2 uses masked prediction instead |

By separating spatial and spectral processing, each stage operates exactly where its assumptions hold. The
result: Stage 1 patch tokens are genuinely spatial, enabling the 28×28 metabolic microregion maps
(spatial contiguity 0.539 vs. 0.167 random baseline at k=6 clusters — see Key Results).

---

## Training

### Stage 1 pretraining

```
Objective:    Barlow Twins (image-level) + spatial coherence + patch-level Barlow Twins (auxiliary)
Input:        single ion images (1, 224, 224), one per forward pass, 2 augmented views per image
Corpus:       158,405 ion images
Epochs:       200 (cosine decay, 1 warmup epoch)
Batch size:   128
LR:           3e-4, AdamW, weight decay 1e-4, gradient clip norm 3.0
Augmentation: random horizontal/vertical flip (p=0.5 each), random resized crop (80-100% area),
              intensity rescale (uniform [0.6, 1.4]), optional Gaussian blur (p=0.5, σ∈[0.1, 2.0])
Script:       pretrain_stage1.py
Checkpoint:   checkpoints/stage1/
```

### Stage 2 pretraining

```
Objective:    masked channel prediction (mask ratio 0.40)
              loss: cosine embedding loss on masked positions only
Input:        set of Stage 1 CLS tokens (frozen) + per-channel m/z, for all channels in an MSI sample
Corpus:       5,600 MSI samples
Epochs:       50
Script:       pretrain_stage2.py
Checkpoint:   checkpoints/stage2/
```

---

## Outputs

| Embedding | Shape | File | Script | Description |
|---|---|---|---|---|
| Stage 1 CLS | `(N_ch, 256)` | `resnet_cls_embeddings.npy` | `extract_stage1_embeddings.py` | Per-ion image embedding (channel-level), `z_chan` |
| Stage 1 patches | `(N_samp, 28, 28, 256)` | `resnet_patch_embeddings.npy` | `extract_stage1_patch_embeddings.py` | Spatially-resolved 28×28 feature maps |
| Stage 2 channel_refined | `(N_ch, 512)` | `stage2_channel_refined.npy` | `extract_stage2_embeddings.py` | Per-ion with cross-channel context, `z_chan^refined` |
| Stage 2 sample_cls | `(N_samp, 512)` | `stage2_sample_cls.npy` | `extract_stage2_embeddings.py` | Whole-sample metabolic phenotype, `z_sample` |
| SMILES-fused | `(N_ch_unambig, 768)` | `resnet+smiles.npy` | `fuse_embeddings.py` | ResNet CLS + MolFormer (n_cand==1 only, post-hoc oracle, not part of MetaboFM) |

All embedding files land in `outputs/embeddings/` alongside matching `*_meta.csv` files that map row indices to sample/channel metadata.

---

## Key Results (current manuscript, Submission 2)

HMDB chemical-taxonomy classification/retrieval is the **primary** evaluation family in the current
manuscript (Results section 1), not a supporting result — that framing changed from an earlier draft.

### HMDB super-class classification (macro-F1) and retrieval (MAP@10)

| Variant | Macro-F1 | Notes |
|---|---|---|
| Stage 2 channel-refined, unambiguous subset (n_cand=1) | **0.354** | primary comparison |
| Stage 2 channel-refined, all channels | 0.272 | full-corpus completeness check |
| Stage 1 channel-level, all channels | 0.194 | |
| Stage 1 channel-level | 0.147 | vs. m/z-only 0.143, ImageNet-ResNet 0.073 |
| SMILES-only (structure baseline, not part of MetaboFM) | 0.053 | |
| ResNet + SMILES post-hoc fusion (oracle) | 0.188 | does **not** beat Stage 1 alone — fusion adds no signal beyond images |
| m/z-only baseline | 0.143 | |
| ImageNet-ResNet (zero-shot) | 0.073 | |
| Metadata-only baseline | 0.097 | acquisition covariates only (ionisation source, analyser, polarity, organism) |

Molecule-level retrieval (per-class MAP@10, Stage 2): lipids 0.840, organoheterocyclics 0.507, organic acids
0.396, benzenoids 0.306; overall MAP@10 0.804.

### Molecular identity preservation (within- vs. between-molecule cosine similarity)

| Variant | Within | Between | Gap |
|---|---|---|---|
| Stage 2 channel-refined | 0.881 | 0.785 | **0.096** (largest among image-only methods) |
| Stage 1 channel-level | 0.782 | 0.723 | 0.059 |
| ResNet + SMILES (oracle) | — | — | 0.030 |
| SMILES-only | — | — | −0.002 (no separation — structure alone doesn't discriminate acquisition context) |

### Spatial structure (Stage 1 patch tokens)

| Metric | Value |
|---|---|
| Patch-PC1 vs. spatial position, mean Pearson r (row / col / combined) | 0.237 / 0.171 / 0.204 |
| Spatial contiguity score, k=6 clusters, 5,600 samples | **0.539** vs. 0.167 random baseline |
| Channel colocalization Spearman ρ (Stage 1 vs. m/z-only) | 0.055 vs. 0.013 |

### Cross-study / cross-platform generalisation

| Metric | Value |
|---|---|
| Leave-one-study-out weighted Recall@1 (Stage 2 / Stage 1 mean-pool / random) | 0.807 / 0.786 / 0.302 |
| Same-tissue same-platform cosine similarity | 0.891 |
| Same-tissue, different-platform (Δ vs. above) | 0.842 (Δ = 0.049) |
| Different-tissue, different-platform | 0.829 |

---

## File Structure

```
├── ARCHITECTURE.md                        ← this file
├── metabofm_paths.py                      ← central path configuration (set METABOFM_ROOT / METABOFM_RAW_DIR)
│
├── models/
│   ├── __init__.py
│   ├── resnet_encoder.py                  ← Stage 1: ResNet-18 spatial encoder (Barlow Twins)
│   └── channel_aggregator.py              ← Stage 2: permutation-invariant channel Transformer
│
├── dataset.py                             ← per-channel loader, returns (1, H, W) tensors
├── utils.py                               ← shared utilities (normalization, metrics, etc.)
│
├── pretrain_stage1.py                     ← Stage 1 Barlow Twins pretraining (ResNet-18)
├── pretrain_stage2.py                     ← Stage 2 masked channel pretraining
├── plot_training.py                       ← training loss/metric plots from checkpoint logs
│
├── filter_samples.py                      ← quality-filter raw MSI samples (aspect ratio, sparsity)
├── build_channel_csv.py                   ← build per-channel metadata CSV from MSI manifests
├── build_molecule_centroids.py            ← per-m/z centroid embeddings for molecule-level analyses
├── build_figure7_data.py                  ← drug-likeness scoring data (feeds main Fig. 6 panels d-f)
├── generate_dataset_diversity_table.py    ← Supplementary Table 1: sample counts by platform/organ
│
├── extract_stage1_embeddings.py           ← extract Stage 1 CLS tokens (resnet_cls_embeddings.npy)
├── extract_stage1_patch_embeddings.py     ← extract 28×28 patch maps (resnet_patch_embeddings.npy)
├── extract_stage2_embeddings.py           ← extract Stage 2 embeddings (sample_cls + channel_refined)
├── extract_stage2_unambiguous.py          ← Stage 2 embeddings restricted to n_cand==1 channels
├── extract_unambiguous_variants.py        ← all representation variants, unambiguous-candidate subset
├── extract_imagenet_baseline.py           ← ImageNet-pretrained ResNet zero-shot baseline
├── fuse_embeddings.py                     ← post-hoc ResNet CLS + MolFormer fusion (n_cand==1 only)
├── align_embeddings.py                    ← align embedding spaces across candidate/filtered manifests
│
├── benchmarks.py                          ← HMDB super_class/class F1, leave-platform-out probes
├── ablation_datasize.py                   ← Stage 2 training-data-scale ablation
├── ablation_multiseed.py / ablation_rerun_50_75.py  ← ablation seed/checkpoint reruns
├── compute_stage1_hmdb_map.py             ← Stage 1-only HMDB MAP@k benchmark
├── compute_lisi_scores.py / compute_lisi_scores_raw.py  ← LISI technical-covariate mixing scores
├── compute_contiguity_baseline.py         ← random-assignment contiguity baseline (k=6)
├── probe_spatial_patches.py               ← spatial patch analysis: coherence, microregions, leave-out probes
├── probe_patch_coherence.py               ← training-time sanity check: spatial coherence vs checkpoint
├── probe_resnet_umap.py                   ← per-TIFF UMAP with ROI annotations (requires TIFF stacks)
├── probe_sample_umap.py                   ← sample-level UMAP of stage2_sample_cls embeddings
├── probe_molecule_spatial_consistency.py  ← within- vs between-organ spatial map similarity
├── probe_molecule_variance.py             ← full within/between-molecule cosine-similarity analysis (all m/z)
├── probe_channel_colocalization.py        ← channel-pair spatial colocalization vs Stage 1 embedding similarity
├── probe_crossdataset_retrieval.py        ← leave-one-acquisition-out cross-acquisition retrieval (primary generalisation result)
├── probe_leave_study_out.py               ← strict leave-one-study-out retrieval (real METASPACE submitter identity)
├── probe_crossplatform_consistency.py     ← cross-platform sample-embedding consistency
├── probe_crossplatform_retrieval.py       ← cross-platform channel-level retrieval breakdown
│
├── smiles_retrieval.py                    ← cross-modal image→SMILES retrieval via InfoNCE projector
├── make_liver_nodule_roi.py               ← ROI annotation tool for liver nodule TIFF
├── show_fixed_samples.py                  ← visualise fixed/representative samples from dataset
├── export_explorer_data.py                ← generates the data bundled into the interactive embedding explorer
│
├── plot_figure1_bc.py … plot_figure7.py   ← main-figure panel generation
├── plot_figS1.py … plot_figS14.py         ← supplementary-figure panel generation
├── plot_utils.py                          ← shared plotting helpers (quality filters, pipeline diagrams,
│                                             representative-image selection: median-variance, not max)
└── save_legends.py                        ← standalone colorbar/legend SVGs for the figure scripts above
```

The `plot_figure*.py` / `plot_figS*.py` / `save_legends.py` scripts are the figure-generation layer consumed
by the manuscript at `Y:\coskun-lab\Efe\MetaboFM\manuscript\Submission 2\` — see that directory's `CLAUDE.md`
for the full figure pipeline and the numbering-consistency requirements between this directory and the `.tex`
sources.
