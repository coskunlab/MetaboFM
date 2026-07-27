"""
utils.py
--------
Shared utility functions for MetaboFM.
Self-contained — no dependency on the v1 code/ directory.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# CID HELPERS
# ============================================================

_CID_RE = re.compile(r"\bCID\s*0*([0-9]+)\b", flags=re.IGNORECASE)


def normalize_cid(x) -> str | None:
    """Normalize any CID representation to 'CID<int>' or None."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan"}:
        return None
    m = _CID_RE.search(s)
    if m:
        return f"CID{int(m.group(1))}"
    if s.isdigit():
        return f"CID{int(s)}"
    try:
        f = float(s)
        if f == int(f):
            return f"CID{int(f)}"
    except ValueError:
        pass
    return None


def parse_cids_field(raw) -> list[str]:
    """Parse a cand_pubchem_cids cell into a deduplicated list of CID strings."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        toks = list(raw)
    else:
        s = str(raw).strip()
        if not s or s.lower() in {"none", "nan"}:
            return []
        toks = re.split(r"[;,|]\s*|\s*;\s*", s)
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        cid = normalize_cid(t)
        if cid and cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out


def build_cid_index(pq_df: pd.DataFrame, cid_col: str = "cid") -> dict[str, int]:
    """Build {normalized_cid -> row_index} map from a molecule index DataFrame."""
    if cid_col not in pq_df.columns:
        raise KeyError(f"DataFrame missing column {cid_col!r}. Available: {list(pq_df.columns)}")
    idx: dict[str, int] = {}
    for i, raw in enumerate(pq_df[cid_col].values):
        cid = normalize_cid(raw)
        if cid and cid not in idx:
            idx[cid] = i
    return idx


# ============================================================
# VECTOR HELPERS
# ============================================================

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1-D vector."""
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / max(norm, eps)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of a 2-D matrix."""
    X = np.asarray(X, dtype=np.float32, order="C")
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def mean_pool(X: np.ndarray) -> np.ndarray:
    """Mean-pool a 2-D array along axis 0. Raises if empty."""
    X = np.asarray(X)
    if X.size == 0:
        raise ValueError("Cannot mean-pool an empty array.")
    return X.mean(axis=0)


# ============================================================
# PARQUET CHUNKED WRITER
# ============================================================

class ChunkedParquetWriter:
    """Context manager for streaming chunked Parquet writes."""

    def __init__(self, path: str | Path, chunk_rows: int = 500_000,
                 compression: str = "zstd"):
        self.path        = Path(path)
        self.chunk_rows  = int(chunk_rows)
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._buf: list[dict] = []

    def append(self, row: dict):
        self._buf.append(row)
        if len(self._buf) >= self.chunk_rows:
            self._flush()

    def _flush(self):
        if not self._buf:
            return
        df    = pd.DataFrame(self._buf)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, table.schema,
                                            compression=self.compression)
        self._writer.write_table(table)
        self._buf = []

    def close(self):
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ============================================================
# V2 EMBEDDING HELPERS
# ============================================================

def group_rows_by_sample(meta_df: pd.DataFrame) -> dict[str, list[int]]:
    """Return {sample_path -> [row_indices]} mapping for a channel-level DataFrame."""
    groups: dict[str, list[int]] = {}
    for i, sp in enumerate(meta_df["sample_path"].tolist()):
        groups.setdefault(str(sp), []).append(i)
    return groups


group_rows_by_tile = group_rows_by_sample   # backward-compat alias


def pack_channel_embeddings(
    cls_array: np.ndarray,   # (N_channels, D)
    meta_df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "",
) -> None:
    """
    Save per-sample stacked channel embeddings (N_samples, C_max, D) and metadata.
    MSI samples with fewer than C_max channels are zero-padded.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups  = group_rows_by_sample(meta_df)
    C_max   = max(len(v) for v in groups.values())
    D       = cls_array.shape[1]
    samples = list(groups.keys())
    N       = len(samples)

    stacked = np.zeros((N, C_max, D), dtype=np.float32)
    sample_meta_rows = []

    for sample_i, sp in enumerate(samples):
        idxs = groups[sp]
        for ci, row_i in enumerate(idxs):
            stacked[sample_i, ci] = cls_array[row_i]
        sample_meta_rows.append({
            "sample_i":    sample_i,
            "sample_path": sp,
            "n_channels":  len(idxs),
        })

    stem = f"{prefix}." if prefix else ""
    np.save(out_dir / f"{stem}channel_embeddings_grouped.npy", stacked)
    pd.DataFrame(sample_meta_rows).to_csv(out_dir / f"{stem}sample_meta.csv", index=False)
    print(f"[OK] Stacked channel embeddings: {stacked.shape} → {out_dir}")
