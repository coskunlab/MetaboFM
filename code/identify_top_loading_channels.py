"""
identify_top_loading_channels.py
------------------------------
For each organ's PC1 channel-loading ranking (from embed_histology_comparison.py),
looks up molecule identities for the top-loading m/z channels via METASPACE's
annotation database, and downloads their ion images for visual comparison
against the H&E — part of the manuscript's H&E-comparison analysis evidence
pipeline (does MetaboFM find real, named metabolite structure invisible in H&E?).

Must run under the torch_gpu conda env (needs `metaspace`).

Usage
-----
  python identify_top_loading_channels.py
"""

from __future__ import annotations

from pathlib import Path
from metabofm_paths import METABOFM_ROOT

import numpy as np
import pandas as pd
from metaspace import SMInstance

HIST_DIR = METABOFM_ROOT / "outputs/optical_images/histology_comparison"
TOP_N = 4
MZ_TOL = 0.001

CANDIDATES = {
    "Lung": "2023-06-27_22h58m39s",
    "Placenta": "2024-02-26_22h36m04s",
    "Pancreas": "2023-07-05_22h26m29s",
    "Brain": "2019-11-25_17h14m31s",
}


def process_one(sm: SMInstance, organ: str, dataset_id: str):
    csv_path = HIST_DIR / f"{organ}_{dataset_id}_pc1_channel_loadings.csv"
    if not csv_path.exists():
        print(f"[SKIP] {organ}: no loadings CSV at {csv_path}")
        return
    loadings = pd.read_csv(csv_path).sort_values("loading_norm", ascending=False).head(TOP_N)

    print(f"\n=== {organ} ({dataset_id}) — top {TOP_N} PC1-loading channels ===")
    ds = sm.dataset(id=dataset_id)
    res = ds.results(database=("HMDB", "v4")).reset_index()

    images = {}
    id_rows = []
    for _, row in loadings.iterrows():
        target_mz = float(row["mz"])
        diffs = (res["mz"] - target_mz).abs()
        if diffs.empty or diffs.min() > MZ_TOL:
            print(f"  [WARN] mz={target_mz:.4f}: no live annotation match")
            continue
        best = res.loc[diffs.idxmin()]
        names = best["moleculeNames"]
        name_str = ", ".join(names[:3]) if isinstance(names, list) else str(names)
        print(f"  channel_idx={int(row['channel_idx'])} mz={target_mz:.4f} "
              f"formula={best['formula']} adduct={best['adduct']} msm={best['msm']:.3f}")
        print(f"      -> {name_str}")

        try:
            imgs = ds.isotope_images(sf=best["formula"], adduct=best["adduct"])
            img = np.nan_to_num(np.asarray(imgs._images[0]))
        except Exception as e:
            print(f"      [WARN] image fetch failed: {e}")
            continue

        key = f"mz{target_mz:.1f}_{best['formula']}"
        images[key] = img
        id_rows.append({
            "channel_idx": int(row["channel_idx"]), "mz": target_mz,
            "formula": best["formula"], "adduct": best["adduct"],
            "msm": float(best["msm"]), "top_names": name_str,
            "image_key": key,
        })

    if not images:
        print(f"[SKIP] {organ}: no channels resolved to live images")
        return

    npz_path = HIST_DIR / f"{organ}_{dataset_id}_top_loading_images.npz"
    np.savez(npz_path, **images)
    csv_out = HIST_DIR / f"{organ}_{dataset_id}_top_loading_identities.csv"
    pd.DataFrame(id_rows).to_csv(csv_out, index=False)
    print(f"[DONE] images -> {npz_path}")
    print(f"[DONE] identities -> {csv_out}")


def main():
    sm = SMInstance()
    for organ, dataset_id in CANDIDATES.items():
        try:
            process_one(sm, organ, dataset_id)
        except Exception as e:
            print(f"[ERROR] {organ} ({dataset_id}): {type(e).__name__} {str(e)[:300]}")


if __name__ == "__main__":
    main()
