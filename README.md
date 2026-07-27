# MetaboFM

Self-supervised representation learning for mass spectrometry imaging (MSI). MetaboFM is a
two-stage framework: Stage 1 learns spatially aware embeddings of individual ion images with a
Barlow Twins objective; Stage 2 aggregates all detected channels in a sample with a Transformer,
using each channel's *m/z* as positional context, to produce a refined per-channel embedding and
a pooled per-sample embedding. Trained on 158,405 ion images from 5,600 public MSI samples
spanning 8 ionization sources, 17 mass analyzer types, and both polarities.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full model and pipeline description.

## Layout

All source lives in [`code/`](code/); run scripts as `python code/<script>.py` from the
repository root (or `cd code` first — either works). `data/` and `outputs/` are expected as
siblings of `code/`, not inside it — see Configuration below.

## Installation

```bash
conda env create -f environment.yml
conda activate metabofm
```

Requires a CUDA-capable GPU for training and embedding extraction; benchmark and plotting scripts
can run on CPU.

## Configuration

All scripts import their data/output directories from [`code/metabofm_paths.py`](code/metabofm_paths.py)
instead of hardcoding paths. Set these environment variables before running anything:

```bash
export METABOFM_ROOT=/path/to/metabofm      # contains data/ and outputs/
export METABOFM_RAW_DIR=/path/to/raw/msi    # raw per-sample .npz ion-image stacks
```

## Data

MetaboFM is trained on public MSI datasets aggregated from [METASPACE](https://metaspace2020.eu/)
via its GraphQL API. `filter_samples.py` applies the quality filters described in Methods
(minimum pixel dimensions, sparsity, channel count) and `build_channel_csv.py` expands the
resulting sample-level manifest into the channel-level manifest used everywhere downstream.

## Pipeline

1. **Data curation** — `filter_samples.py`, `build_channel_csv.py`.
2. **Stage 1 training** — `pretrain_stage1.py`: ResNet-18 encoder trained with Barlow Twins +
   spatial coherence + patch-level auxiliary losses on individual ion images.
3. **Stage 2 training** — `pretrain_stage2.py`: cross-channel aggregation Transformer trained
   with masked-channel prediction, producing refined channel embeddings and a pooled sample
   embedding.
4. **Embedding extraction** — `extract_stage1_embeddings.py`, `extract_stage1_patch_embeddings.py`,
   `extract_stage2_embeddings.py`, `extract_imagenet_baseline.py`, `fuse_embeddings.py`.
5. **Benchmarks** — `benchmarks.py`: linear probing and retrieval against HMDB chemical taxonomy,
   plus every baseline variant (SMILES-only, *m/z*-only, ImageNet, metadata-only, ResNet+SMILES
   fusion). `probe_crossdataset_retrieval.py` and `probe_leave_study_out.py` cover organ/organism
   retrieval generalization, including the strict real-study-identity validation.
6. **Figures** — one script per main/supplementary figure; see the table below.

## Reproducing the figures

Run any script with `python code/<script>.py`; each writes per-panel SVGs and a caption file to
`outputs/figures/<name>/`.

| Figure | Script |
|---|---|
| Fig. 1b–c | `plot_figure1_bc.py` |
| Fig. 2 | `plot_figure2.py` |
| Fig. 3 | `plot_figure3.py` |
| Fig. 4 | `plot_figure4.py` |
| Fig. 5 | `plot_figure5.py` |
| Fig. 6 | `plot_figure6.py` |
| Fig. 7 | `plot_figure7.py` |
| Supp. Fig. S1 | `plot_figS1.py` |
| Supp. Fig. S2 | `plot_figS2.py` |
| Supp. Fig. S3 | `plot_figS3.py` |
| Supp. Fig. S4 | `plot_figS4.py` |
| Supp. Fig. S5 | `plot_figS5.py` |
| Supp. Fig. S6 | `plot_figS6.py` |
| Supp. Fig. S7 | `plot_figS7.py` |
| Supp. Fig. S8 | `plot_figS8.py` |
| Supp. Fig. S9 | `plot_figS9.py` |
| Supp. Fig. S10 | `plot_figS10.py` |
| Supp. Fig. S11 | `plot_figS11.py` |
| Supp. Fig. S12 | `plot_figS12.py` |
| Supp. Fig. S13 | `plot_figS13.py` |
| Supp. Fig. S14 | `plot_figS14.py` |
| Supp. Software 1 (interactive embedding explorer) | `export_explorer_data.py` generates the data bundled into the self-contained HTML explorer |

The compiled explorer (a single self-contained HTML file, no install/server/account needed) is available
as a download from the [Supplementary Software release](https://github.com/coskunlab/MetaboFM/releases/tag/explorer-v1)
rather than committed to the repository, since it embeds ~130 MB of data and exceeds GitHub's per-file size limit.

## Pretrained weights

Final Stage 1 and Stage 2 checkpoints (the exact weights used to produce every result in the
manuscript) are attached to the [`weights-v1` release](https://github.com/coskunlab/MetaboFM/releases/tag/weights-v1):

- `stage1_encoder_final.pt` — Stage 1 spatial encoder (ResNet-18, Barlow Twins).
- `stage2_aggregator_final.pt` — Stage 2 channel-aggregation Transformer.

Download and point the extraction scripts' `--checkpoint` argument at the downloaded files to skip
training and go straight to embedding extraction.

## Citation

A citation entry will be added here once the manuscript is published.

## License

Released under the [MIT License](LICENSE).
