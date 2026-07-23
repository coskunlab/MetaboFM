"""
plot_training.py
----------------
Plot training loss curves from any run's training_log.jsonl.
Auto-discovers the latest run, or you can specify a run directory.

Usage
-----
  # Plot latest stage 1 run
  python plot_training.py --stage 1

  # Plot latest stage 2 run
  python plot_training.py --stage 2

  # Plot a specific run by folder name
  python plot_training.py --stage 1 --run run_20260629_010425

  # List all available runs
  python plot_training.py --list
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

STAGE1_BASE = METABOFM_ROOT / "metabofm_v2/stage1_resnet"
STAGE2_BASE = METABOFM_ROOT / "metabofm_v2/stage2"
OUT_DIR     = METABOFM_ROOT / "outputs/training_plots"


# ============================================================
# HELPERS
# ============================================================

def find_runs(base: Path) -> list[Path]:
    """Return all run directories sorted newest first."""
    if not base.exists():
        return []
    runs = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "training_log.jsonl").exists()],
        key=os.path.getmtime, reverse=True
    )
    return runs


def load_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows)


def smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def plot_run(df: pd.DataFrame, label: str, color: str,
             ax_loss, ax_lr) -> dict:
    """Plot one run onto axes. Returns summary stats."""
    step_df  = df[df["loss"].notna()].copy()  if "loss"      in df.columns else pd.DataFrame()
    epoch_df = df[df["epoch_end"].notna()].copy() if "epoch_end" in df.columns else pd.DataFrame()
    stats    = {}

    if not step_df.empty:
        # Use step if available, else sequential index
        x = step_df["step"].values.astype(float) if "step" in step_df else np.arange(len(step_df), dtype=float)
        y = step_df["loss"].values.astype(float)
        y_sm = smooth(y)

        ax_loss.plot(x, y, alpha=0.15, color=color, linewidth=0.7)
        ax_loss.plot(x[-len(y_sm):], y_sm, color=color, linewidth=2.0, label=label)

        stats = {
            "n_steps":    len(y),
            "start_loss": float(y[0]),
            "min_loss":   float(np.min(y)),
            "cur_loss":   float(y[-1]),
            "cur_epoch":  float(step_df["epoch"].values[-1]) if "epoch" in step_df else 0,
        }

        lr_col = next((c for c in ("lr", "learning_rate") if c in step_df.columns), None)
        if lr_col:
            lr_vals = step_df[lr_col].values.astype(float)
            ax_lr.plot(x, lr_vals, color=color, linewidth=1.2, label=label)

    if not epoch_df.empty and "avg_loss" in epoch_df.columns:
        ex = epoch_df["global_step"].values.astype(float) if "global_step" in epoch_df \
             else epoch_df["epoch_end"].values.astype(float)
        ey = epoch_df["avg_loss"].values.astype(float)
        ax_loss.scatter(ex, ey, marker="D", s=50, color=color, zorder=5,
                        label=f"{label} epoch avg")
        stats["epoch_avgs"] = list(zip(epoch_df["epoch_end"].values.tolist(), ey.tolist()))

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="1", choices=["1", "2"],
                        help="Which stage to plot (default: 1)")
    parser.add_argument("--run",   default=None,
                        help="Specific run folder name (e.g. run_20260629_010425). "
                             "Omit to use the latest run.")
    parser.add_argument("--all",   action="store_true",
                        help="Plot all runs for this stage overlaid")
    parser.add_argument("--list",  action="store_true",
                        help="List all available runs and exit")
    args = parser.parse_args()

    base = STAGE1_BASE if args.stage == "1" else STAGE2_BASE
    runs = find_runs(base)

    # --list
    if args.list:
        print(f"Stage {args.stage} runs in {base}:")
        if not runs:
            print("  (none found)")
        for r in runs:
            log = r / "training_log.jsonl"
            df  = load_log(log)
            n   = len(df[df["loss"].notna()]) if "loss" in df.columns else 0
            ep  = round(df["epoch"].max(), 2) if "epoch" in df.columns and not df.empty else "?"
            print(f"  {r.name}  steps={n}  epoch={ep}")
        return

    if not runs:
        print(f"No runs found under {base}")
        return

    # Select runs to plot
    if args.all:
        selected = runs
    elif args.run:
        match = [r for r in runs if r.name == args.run]
        if not match:
            print(f"Run '{args.run}' not found. Available:")
            for r in runs: print(f"  {r.name}")
            return
        selected = match
    else:
        selected = [runs[0]]   # latest

    # ---- Build figure ----
    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(11, 7),
                                          gridspec_kw={"height_ratios": [3, 1]})
    label_name = f"Stage {args.stage} (ion encoder)" if args.stage == "1" \
                 else f"Stage {args.stage} (channel aggregator)"
    fig.suptitle(f"MetaboFM — {label_name}", fontsize=12)

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(selected))]

    for i, run_dir in enumerate(selected):
        df    = load_log(run_dir / "training_log.jsonl")
        label = run_dir.name
        stats = plot_run(df, label, colors[i], ax_loss, ax_lr)

        print(f"\n  Run: {run_dir.name}")
        if stats:
            print(f"    steps       : {stats.get('n_steps', 0)}")
            print(f"    epoch       : {stats.get('cur_epoch', 0):.2f}")
            print(f"    start loss  : {stats.get('start_loss', 0):.4f}")
            print(f"    min loss    : {stats.get('min_loss', 0):.4f}")
            print(f"    current loss: {stats.get('cur_loss', 0):.4f}")
            if "epoch_avgs" in stats:
                for ep, avg in stats["epoch_avgs"]:
                    print(f"    epoch {int(ep)} avg : {avg:.4f}")
        else:
            print("    No data yet")

    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Barlow Twins Loss")
    ax_loss.legend(fontsize=8, frameon=False)
    ax_loss.spines[["top", "right"]].set_visible(False)

    ax_lr.set_xlabel("Training step")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.legend(fontsize=8, frameon=False)
    ax_lr.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag      = args.run or ("all" if args.all else selected[0].name)
    out_path = OUT_DIR / f"stage{args.stage}_{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Plot saved: {out_path}")


if __name__ == "__main__":
    main()
