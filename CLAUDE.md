# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This directory is the public release of MetaboFM (mirrored to `github.com/coskunlab/MetaboFM`) and
is the **only directory in the wider `MetaboFM` working tree with real git history** — commits and
pushes happen here, not at the working-tree root. It's a cleaned copy of the internal `code_v2/`
codebase (one level up): hardcoded local paths replaced with `METABOFM_ROOT`/`METABOFM_RAW_DIR` via
`code/metabofm_paths.py`, source nested under `code/`.

See [`README.md`](README.md) for installation, the figure-reproduction table, and pretrained-weight
download instructions, and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full model design, training
config, and key manuscript results — both are kept current and should be read before making
non-trivial changes rather than re-deriving that information from source.

## Commands

```bash
conda env create -f environment.yml
conda activate metabofm

export METABOFM_ROOT=/path/to/metabofm      # contains data/ and outputs/, parent of code/
export METABOFM_RAW_DIR=/path/to/raw/msi    # raw per-sample .npz ion-image stacks

python code/<script>.py                     # run any pipeline script from the repo root (or `cd code` first)
```

Requires a CUDA GPU for training (`pretrain_stage1.py`, `pretrain_stage2.py`) and embedding
extraction (`extract_stage1*.py`, `extract_stage2*.py`); benchmark (`benchmarks.py`, `probe_*.py`)
and plotting (`plot_figure*.py`, `plot_figS*.py`) scripts run on CPU. No test suite or linter is
configured — validation is empirical: rerun the relevant extraction/benchmark script and compare
its printed metrics against the "Key Results" table in `ARCHITECTURE.md`.

## Architecture in brief

Two-stage self-supervised model, run in this order (full detail in `ARCHITECTURE.md`):

1. **Stage 1** (`code/models/resnet_encoder.py`) — ResNet-18, Barlow Twins objective, one
   single-channel `(1, 224, 224)` ion image per forward pass. Produces a 256-dim CLS embedding
   (`z_chan`) and 784×256 patch tokens (28×28 spatial map) per channel.
2. **Stage 2** (`code/models/channel_aggregator.py`) — a permutation-invariant Transformer
   (`ChannelAggregator`) over the *set* of Stage 1 CLS tokens for all channels in a sample, using a
   learned *m/z* embedding instead of positional encoding. Masked-channel-prediction pretraining.
   Produces a pooled `sample_cls` (512-dim, `z_sample`) and per-channel `channel_refined` (512-dim,
   `z_chan^refined`).

This split exists because ion channels are ordered by MSM score (not *m/z*) and vary in count
across samples — see `ARCHITECTURE.md`, "Why This Design" for the full argument against a
single-model approach over the full `(C, H, W)` stack.

**Terminology**: call Stage 2 "the cross-channel aggregation Transformer" in any reader-facing text
(docs, docstrings, commit messages) — never the internal class name `ChannelAggregator`. Distinguish
"acquisition"/`dataset_id` (a single MSI upload) from genuine "study" (real METASPACE-submitter
identity) — only `probe_leave_study_out.py` uses true study identity; everywhere else use
"cross-dataset"/"cross-acquisition", not "cross-study".

## H&E / MALDI-IHC histology-comparison pipeline

`code/probe_optical_availability.py` through `code/plot_figS16.py` (see README's "H&E and
MALDI-IHC comparison" section for the exact run order) test whether Stage 1's spatial embeddings
track real anatomy and resolve structure invisible in registered histology. This pipeline needs
**two conda environments run in sequence**, not one: a GPU-enabled env for METASPACE queries and
Stage 1 inference (`probe_*.py`), then the base `metabofm` env for PCA/UMAP/plotting (`embed_*.py`,
`plot_fig*.py`) — some GPU-enabled setups have a BLAS conflict that crashes silently on
`matplotlib.savefig`, `sklearn`, and `numpy.linalg` calls. Keep this split when extending the
pipeline rather than merging fetch and plot steps into one script. `metabofm_paths.py`'s
`IHC_RAW_DIR` points at the external MALDI-IHC dataset (not part of the METASPACE training corpus).

## Working conventions

- Every script imports its directories from `code/metabofm_paths.py` (`METABOFM_ROOT`,
  `METABOFM_RAW_DIR`/`MSI_RAW_DIR`) rather than hardcoding paths — keep new scripts consistent with
  this so the repo stays runnable on any machine.
- `code/plot_utils.py` is shared across all `plot_figure*.py`/`plot_figS*.py` scripts:
  representative-sample/channel selection, image-quality filters, the `EXCLUDED_DATASET_IDS`
  hard-exclusion set, and scale-bar helpers. Changes here ripple into every figure script — rerun
  the affected ones and check whether the manuscript's figure captions cite numbers that just changed.
- This repo is the resync **target**, not the source of truth for active development — the working
  codebase is `code_v2/` one level up in the wider working tree (not itself a git repo). When
  porting a change from there: strip hardcoded paths to `METABOFM_ROOT`/`METABOFM_RAW_DIR`, strip
  "v2" references from filenames/docstrings/comments, and remove `cd code_v2` usage lines, before
  committing here.
- The compiled interactive embedding explorer (Supplementary Software) is not committed to this
  repo — it's ~130 MB, over GitHub's per-file limit — and instead ships as a release asset
  (`explorer-v1` tag); `code/export_explorer_data.py` generates the data bundled into it. Pretrained
  checkpoints ship the same way, under the `weights-v1` tag.
