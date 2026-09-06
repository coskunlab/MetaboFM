"""
probe_optical_availability.py
------------------------------
Checks how many samples in the trained MetaboFM corpus have a real optical
(microscopy) image attached on METASPACE, and applies a coarse colour heuristic
to flag which of those look like genuine H&E histology stains.

Motivation: the manuscript's H&E-comparison analysis needs a concrete task where
MetaboFM answers a biology question histology cannot (e.g. an H&E-invisible
tumour margin) rather than a "complementarity" argument. Before attempting
that, we need to know whether real, usable H&E data is actually reachable for
any of our trained samples — METASPACE datasets can carry an "optical image"
field (used for MSI/microscopy co-registration), but this field is a mixed bag
in practice: bright-field photos of unstained sections, fluorescence images,
phase-contrast, and occasionally true H&E. Manual spot-checks (six samples,
one per major organ) found only 1/6 was genuine H&E; the rest were
fluorescence, phase-contrast, or unstained bright-field.

Two-stage pipeline:
  1. scan_availability() — for every dataset_id in the trained corpus
     (channels_v2.csv), query METASPACE's getRawOpticalImage GraphQL endpoint
     for optical-image presence (metadata only, no image download — fast).
  2. classify_he_like() — for datasets that do have an optical image, download
     a thumbnail and apply a colour heuristic: restrict to tissue-like pixels
     (exclude near-white background and near-black borders/frames), then check
     whether their hue falls in the eosin-pink/hematoxylin-purple band
     characteristic of H&E.

IMPORTANT — this heuristic is a triage shortlist, not ground truth. Validated
against 6 manually-inspected samples it got 5/6 right; the one failure was a
solid red border artefact in an unstained image that dominated the pixel
sample and read as "eosin pink" by hue alone (colour heuristics cannot
distinguish a real stain from a solid-colour frame/artifact without spatial
reasoning). Any dataset intended for use in the manuscript or response letter
MUST be visually confirmed (open the image, or use plot_utils-style QC) before
being cited as H&E.

Usage
-----
  python probe_optical_availability.py             # runs both stages
  python probe_optical_availability.py --skip-scan # reuse cached availability CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path
from metabofm_paths import METABOFM_ROOT
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
from metaspace import SMInstance

CHANNEL_CSV = METABOFM_ROOT / "metaspace_images_dump/channels_v2.csv"
OUT_DIR = METABOFM_ROOT / "outputs/optical_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AVAILABILITY_CSV = OUT_DIR / "optical_availability.csv"
CLASSIFICATION_CSV = OUT_DIR / "he_classification.csv"

# Colour-heuristic thresholds (see classify_he_like docstring)
BRIGHTNESS_MIN = 0.08   # excludes near-black borders / frames / burned-in text
BRIGHTNESS_MAX = 0.92   # excludes near-white slide background
SAT_MIN = 0.10          # per-pixel saturation floor to count as "coloured"
COLORED_FRAC_MIN = 0.20 # fraction of tissue pixels that must be coloured
HE_HUE_FRAC_MIN = 0.55  # fraction of coloured tissue pixels in the H&E hue band
THUMBNAIL_SIZE = (400, 400)


def scan_availability(sm: SMInstance) -> pd.DataFrame:
    """Query optical-image presence for every dataset_id in the trained corpus."""
    ch = pd.read_csv(CHANNEL_CSV, usecols=["dataset_id"])
    ids = ch["dataset_id"].dropna().unique().tolist()
    print(f"[INFO] scanning {len(ids)} trained dataset_ids for optical image availability")

    rows = []
    for i, did in enumerate(ids):
        try:
            raw = sm._gqclient.getRawOpticalImage(did)
            raw_im = raw.get("rawOpticalImage") if raw else None
            if raw_im and raw_im.get("url"):
                status, url = "HAS_OPTICAL", raw_im["url"]
            else:
                status, url = "NO_OPTICAL", None
        except Exception as e:
            status, url = f"ERROR:{type(e).__name__}", None
        rows.append({"dataset_id": did, "status": status, "url": url})
        if (i + 1) % 500 == 0:
            n_ok = sum(1 for r in rows if r["status"] == "HAS_OPTICAL")
            print(f"[{i + 1}/{len(ids)}] has_optical_so_far={n_ok}")

    df = pd.DataFrame(rows)
    df.to_csv(AVAILABILITY_CSV, index=False)
    n_ok = (df.status == "HAS_OPTICAL").sum()
    print(f"[DONE] total={len(df)} has_optical={n_ok} -> {AVAILABILITY_CSV}")
    return df


def _hue_degrees(r, g, b, maxc, minc):
    d = maxc - minc
    hh = np.zeros_like(r)
    is_r = maxc == r
    is_g = (maxc == g) & ~is_r
    is_b = (maxc == b) & ~is_r & ~is_g
    hh[is_r] = ((g[is_r] - b[is_r]) / d[is_r]) % 6
    hh[is_g] = (b[is_g] - r[is_g]) / d[is_g] + 2
    hh[is_b] = (r[is_b] - g[is_b]) / d[is_b] + 4
    return hh * 60


def _he_like_score(img_arr: np.ndarray) -> tuple[bool, float, float, int]:
    """Returns (is_he_like, colored_frac, he_hue_frac, n_tissue_px)."""
    if img_arr.ndim == 2:
        return False, 0.0, 0.0, 0
    if img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]

    h, w = img_arr.shape[:2]
    ys = np.linspace(0, h - 1, 120).astype(int)
    xs = np.linspace(0, w - 1, 120).astype(int)
    patch = img_arr[np.ix_(ys, xs)].reshape(-1, 3).astype(np.float32) / 255.0
    r, g, b = patch[:, 0], patch[:, 1], patch[:, 2]
    maxc, minc = patch.max(axis=1), patch.min(axis=1)
    brightness = (maxc + minc) / 2
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0)

    tissue = (brightness > BRIGHTNESS_MIN) & (brightness < BRIGHTNESS_MAX)
    n_tissue = int(tissue.sum())
    if n_tissue < 20:
        return False, 0.0, 0.0, n_tissue

    r_t, g_t, b_t = r[tissue], g[tissue], b[tissue]
    maxc_t, minc_t, sat_t = maxc[tissue], minc[tissue], sat[tissue]
    colored_frac = float((sat_t > SAT_MIN).mean())

    idx = (maxc_t != minc_t) & (sat_t > SAT_MIN)
    if idx.sum() < 5:
        return False, colored_frac, 0.0, n_tissue

    hue_deg = _hue_degrees(r_t[idx], g_t[idx], b_t[idx], maxc_t[idx], minc_t[idx])
    # H&E band: eosin pink/red (wraps through 0) + hematoxylin purple/blue
    he_band = ((hue_deg >= 250) & (hue_deg <= 360)) | (hue_deg <= 20)
    he_frac = float(he_band.mean())

    is_he = colored_frac > COLORED_FRAC_MIN and he_frac > HE_HUE_FRAC_MIN
    return is_he, colored_frac, he_frac, n_tissue


def classify_he_like(sm: SMInstance, availability: pd.DataFrame) -> pd.DataFrame:
    """Download a thumbnail for every HAS_OPTICAL dataset and score it against
    the H&E colour heuristic. See module docstring for the accuracy caveat."""
    has = availability[availability.status == "HAS_OPTICAL"].reset_index(drop=True)
    print(f"[INFO] classifying {len(has)} optical images")

    rows = []
    for i, row in has.iterrows():
        did = row["dataset_id"]
        try:
            r = requests.get(row["url"], timeout=20)
            im = Image.open(BytesIO(r.content))
            im.thumbnail(THUMBNAIL_SIZE)
            arr = np.asarray(im.convert("RGB"))
            is_he, colored_frac, he_frac, n_tissue = _he_like_score(arr)
            rows.append({
                "dataset_id": did,
                "classified": "HE_LIKE" if is_he else "OTHER",
                "colored_frac": colored_frac,
                "he_hue_frac": he_frac,
                "n_tissue_px": n_tissue,
            })
        except Exception as e:
            rows.append({"dataset_id": did, "classified": f"ERROR:{type(e).__name__}"})
        if (i + 1) % 50 == 0:
            n_he = sum(1 for x in rows if x["classified"] == "HE_LIKE")
            print(f"[{i + 1}/{len(has)}] he_like_so_far={n_he}")

    df = pd.DataFrame(rows)
    df.to_csv(CLASSIFICATION_CSV, index=False)
    print(df["classified"].value_counts())
    print(f"[DONE] -> {CLASSIFICATION_CSV}")
    print("[REMINDER] this is a coarse triage shortlist (colour heuristic only, "
          "~5/6 accuracy against manual spot-checks) — visually confirm any "
          "candidate before citing it as H&E in the manuscript or response letter.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-scan", action="store_true",
                     help="reuse the cached availability CSV instead of re-querying METASPACE")
    args = ap.parse_args()

    sm = SMInstance()

    if args.skip_scan and AVAILABILITY_CSV.exists():
        availability = pd.read_csv(AVAILABILITY_CSV)
    else:
        availability = scan_availability(sm)

    classify_he_like(sm, availability)


if __name__ == "__main__":
    main()
