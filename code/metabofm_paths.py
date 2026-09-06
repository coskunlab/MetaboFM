"""
metabofm_paths.py
------------------
Central path configuration for the MetaboFM pipeline. Every other script
imports its directory constants from here instead of hardcoding absolute
paths, so the pipeline can run on any machine.

Override via environment variables:
  METABOFM_ROOT     Repository root (the parent of this file's own `code/`
                     directory). Expected to contain a `data/` directory
                     (curated MSI corpus, metadata, candidate annotations)
                     and an `outputs/` directory (all pipeline artifacts:
                     embeddings, benchmark results, figures) as siblings of
                     `code/`. Defaults to that parent directory.
  METABOFM_RAW_DIR   Directory containing the raw per-sample MSI files
                     (.npz ion-image stacks) downloaded from METASPACE.
                     Defaults to METABOFM_ROOT / "data" / "raw".
  METABOFM_IHC_RAW_DIR   Directory containing the raw MALDI-IHC dataset
                     (mass-tag antibody panel; not sourced from METASPACE),
                     used only by the H&E/optical-image comparison pipeline.
                     Defaults to METABOFM_ROOT / "data_external" / "maldi_ihc".
"""
import os
from pathlib import Path

METABOFM_ROOT = Path(os.environ.get("METABOFM_ROOT", Path(__file__).resolve().parent.parent))
MSI_RAW_DIR = Path(os.environ.get("METABOFM_RAW_DIR", METABOFM_ROOT / "data" / "raw"))
IHC_RAW_DIR = Path(os.environ.get("METABOFM_IHC_RAW_DIR", METABOFM_ROOT / "data_external" / "maldi_ihc"))
