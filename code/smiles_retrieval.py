"""
smiles_retrieval.py
--------------------
Cross-modal retrieval: ion image embeddings → SMILES library.

Trains a lightweight MLP projector (image embedding → SMILES embedding space)
using InfoNCE loss on paired (image, SMILES) data. Evaluates whether a new
ion image can retrieve structurally related metabolites from a library.

Variants compared:
  stage2   : Stage 2 channel_refined (512-dim) → projector → SMILES space
  stage1   : Stage 1 CLS (256-dim)             → projector → SMILES space
  random   : random 512-dim vectors             → projector → SMILES space
  smiles   : direct SMILES→SMILES (upper bound, same-modality)

Metrics: R@1, R@5, R@10 (correct super_class), MRR

Outputs (in OUT_DIR/):
  smiles_retrieval_results.csv   per-variant retrieval metrics
  smiles_retrieval_results.png   bar chart
  projector_stage2.pt            trained projector weights
  projector_stage1.pt            trained projector weights

Usage
-----
  python smiles_retrieval.py
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
from pathlib import Path

from metabofm_paths import METABOFM_ROOT
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMB_DIR    = METABOFM_ROOT / "outputs/embeddings_v2"
CAND_PQ    = METABOFM_ROOT / "metaspace_images_dump/channels_with_candidates.parquet"
OUT_DIR    = METABOFM_ROOT / "outputs/smiles_retrieval"

SMILES_DIM  = 768   # MolFormer output dim
PROJ_HIDDEN = 512
PROJ_OUT    = 768   # project into SMILES space directly

TRAIN_FRAC  = 0.70
TEST_FRAC   = 0.20
SEED        = 42

EPOCHS      = 30
BATCH_SIZE  = 512
LR          = 3e-4
TEMPERATURE = 0.07

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
RETR_KS = [1, 5, 10]
FP_RADIUS = 2
FP_BITS   = 2048

# ── HELPERS ──────────────────────────────────────────────────────────────────

def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    return x / n


def dataset_grouped_split(dataset_ids: np.ndarray, train_frac: float,
                           test_frac: float, seed: int):
    """Split by dataset_id so no dataset spans train and test."""
    rng = np.random.default_rng(seed)
    unique_ds = np.unique(dataset_ids)
    rng.shuffle(unique_ds)
    n = len(unique_ds)
    n_train = max(1, int(n * train_frac))
    n_test  = max(1, int(n * test_frac))
    train_ds = set(unique_ds[:n_train])
    test_ds  = set(unique_ds[n_train: n_train + n_test])
    m_train = np.array([d in train_ds for d in dataset_ids])
    m_test  = np.array([d in test_ds  for d in dataset_ids])
    return m_train, m_test


# ── MODEL ─────────────────────────────────────────────────────────────────────

class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden: int = PROJ_HIDDEN, out_dim: int = PROJ_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── INFONCE LOSS ──────────────────────────────────────────────────────────────

def infonce_loss(z_img: torch.Tensor, z_smi: torch.Tensor, temp: float) -> torch.Tensor:
    """Symmetric InfoNCE. z_img, z_smi: (B, D) L2-normalised."""
    logits = z_img @ z_smi.T / temp          # (B, B)
    labels = torch.arange(len(z_img), device=z_img.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_s = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_s) / 2


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train_projector(X_img: np.ndarray, X_smi: np.ndarray,
                    in_dim: int, out_dir: Path, tag: str) -> Projector:
    """Train image projector to align with fixed L2-normalised SMILES embeddings."""
    proj = Projector(in_dim, PROJ_HIDDEN, PROJ_OUT).to(DEVICE)
    opt  = torch.optim.Adam(proj.parameters(), lr=LR)

    # SMILES targets are fixed and L2-normalised — no smi_proj needed
    X_smi_norm = l2(X_smi.astype(np.float32))
    X_img_t    = torch.from_numpy(X_img.astype(np.float32))
    X_smi_t    = torch.from_numpy(X_smi_norm)

    ds = TensorDataset(X_img_t, X_smi_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    proj.train()
    for ep in range(1, EPOCHS + 1):
        ep_loss = 0.0
        for ximg, xsmi in loader:
            ximg, xsmi = ximg.to(DEVICE), xsmi.to(DEVICE)
            z_img = proj(ximg)          # already L2-normed by Projector.forward
            loss  = infonce_loss(z_img, xsmi, TEMPERATURE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep % 10 == 0 or ep == 1:
            print(f"    [{tag}] epoch {ep:3d}/{EPOCHS}  loss={ep_loss/len(loader):.4f}")

    torch.save(proj.state_dict(), out_dir / f"projector_{tag}.pt")
    proj.eval()
    return proj


# ── EVALUATE ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def project_batch(model: nn.Module, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i+batch].astype(np.float32)).to(DEVICE)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out, axis=0)


# ── TANIMOTO ─────────────────────────────────────────────────────────────────

def build_morgan_fps(smiles_list: list[str]) -> list:
    """Compute Morgan (ECFP4) fingerprint for each SMILES. Returns list of fps (None if invalid)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def tanimoto_topk(q_fp, g_fps: list, k: int) -> float:
    """Max Tanimoto between query fp and top-k gallery fps (by Tanimoto). Returns mean Tanimoto@k."""
    from rdkit.Chem import DataStructs
    if q_fp is None:
        return float("nan")
    sims = []
    for gfp in g_fps:
        if gfp is not None:
            sims.append(DataStructs.TanimotoSimilarity(q_fp, gfp))
    if not sims:
        return float("nan")
    sims.sort(reverse=True)
    return float(np.mean(sims[:k]))


def tanimoto_retrieval_metrics(q_emb: np.ndarray, g_emb: np.ndarray,
                                q_fps: list, g_fps: list,
                                ks: list[int]) -> dict:
    """
    Rank gallery by cosine similarity on embeddings.
    Evaluate quality of top-k by Tanimoto fingerprint similarity.
    Returns mean Tanimoto@k over all queries (higher = retrieved more similar structures).
    """
    sim = q_emb @ g_emb.T      # (Nq, Ng)
    results = {}
    for k in ks:
        tan_vals = []
        for qi in range(len(q_emb)):
            top_idx = np.argpartition(-sim[qi], min(k, len(g_emb) - 1))[:k]
            top_fps = [g_fps[gi] for gi in top_idx]
            t = tanimoto_topk(q_fps[qi], top_fps, k)
            if not np.isnan(t):
                tan_vals.append(t)
        results[f"Tan@{k}"] = round(float(np.mean(tan_vals)), 4) if tan_vals else 0.0
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {DEVICE}")

    # ── Load embeddings ───────────────────────────────────────────────────────
    print("[LOAD] embeddings ...")
    S1_all  = np.load(str(EMB_DIR / "resnet_cls_embeddings.npy"),  mmap_mode="r")
    S2_all  = np.load(str(EMB_DIR / "stage2_channel_refined.npy"), mmap_mode="r")
    smi_ids = np.load(str(EMB_DIR / "row_ids__smiles_only.npy"))   # (152042,)
    SMI_all = np.load(str(EMB_DIR / "smiles_only.npy"))            # (152042, 768)
    meta    = pd.read_csv(EMB_DIR / "resnet_cls_meta.csv")

    # ── Load candidate SMILES and join by (sample_path, channel_idx) ─────────
    print("[LOAD] candidate SMILES ...")
    cand = pd.read_parquet(CAND_PQ,
                           columns=["sample_path", "channel_idx", "cand_smiles", "msm"])
    # Take first candidate SMILES per channel (all candidates share same MSM for a sum formula)
    cand["first_smiles"] = cand["cand_smiles"].str.split(";").str[0].str.strip()
    cand_idx = cand.set_index(["sample_path", "channel_idx"])["first_smiles"]

    # Build SMILES list aligned to smi_ids
    meta_sub = meta.iloc[smi_ids].reset_index(drop=True)
    smiles_list = []
    for _, row in meta_sub.iterrows():
        key = (row["sample_path"], row["channel_idx"])
        smiles_list.append(cand_idx.get(key, None))
    print(f"  SMILES found: {sum(s is not None for s in smiles_list):,} / {len(smiles_list):,}")

    # ── Compute Morgan fingerprints ───────────────────────────────────────────
    print("[FP] Computing Morgan fingerprints (ECFP4) ...")
    valid_smiles = [s if s is not None else "" for s in smiles_list]
    fps_all = build_morgan_fps(valid_smiles)
    n_valid_fp = sum(f is not None for f in fps_all)
    print(f"  Valid fingerprints: {n_valid_fp:,} / {len(fps_all):,}")

    # ── Filter to channels that have a valid fingerprint ─────────────────────
    m_fp = np.array([f is not None for f in fps_all])
    smi_ids_k  = smi_ids[m_fp]
    SMI_k      = SMI_all[m_fp]
    S1_k       = np.asarray(S1_all[smi_ids_k], dtype=np.float32)
    S2_k       = np.asarray(S2_all[smi_ids_k], dtype=np.float32)
    fps_k      = [f for f, v in zip(fps_all, m_fp) if v]
    ds_ids_k   = meta.iloc[smi_ids_k]["dataset_id"].to_numpy()
    print(f"  Channels with valid fp: {len(smi_ids_k):,}")

    # ── Dataset-grouped split ─────────────────────────────────────────────────
    m_train, m_test = dataset_grouped_split(ds_ids_k, TRAIN_FRAC, TEST_FRAC, SEED)
    print(f"  train: {m_train.sum():,}  test: {m_test.sum():,}")

    X_s1_tr, X_s1_te   = l2(S1_k[m_train]),   l2(S1_k[m_test])
    X_s2_tr, X_s2_te   = l2(S2_k[m_train]),   l2(S2_k[m_test])
    X_smi_tr, X_smi_te = l2(SMI_k[m_train]),  l2(SMI_k[m_test])
    fps_tr = [f for f, v in zip(fps_k, m_train) if v]
    fps_te = [f for f, v in zip(fps_k, m_test)  if v]

    rng = np.random.default_rng(SEED)
    X_rand_k = l2(rng.standard_normal((len(smi_ids_k), 512)).astype(np.float32))
    X_rand_tr, X_rand_te = X_rand_k[m_train], X_rand_k[m_test]

    # Fixed L2-normed SMILES gallery
    g_smi_tr = l2(X_smi_tr)
    g_smi_te = l2(X_smi_te)

    # ── Train projectors ──────────────────────────────────────────────────────
    print("\n[TRAIN] Stage 2 projector ...")
    proj_s2   = train_projector(X_s2_tr,   X_smi_tr, 512, OUT_DIR, "stage2")

    print("\n[TRAIN] Stage 1 projector ...")
    proj_s1   = train_projector(X_s1_tr,   X_smi_tr, 256, OUT_DIR, "stage1")

    print("\n[TRAIN] Random baseline projector ...")
    proj_rand = train_projector(X_rand_tr, X_smi_tr, 512, OUT_DIR, "random")

    # ── Evaluate: Tanimoto@k ──────────────────────────────────────────────────
    print("\n[EVAL] Tanimoto retrieval metrics ...")

    def eval_variant(q_emb: np.ndarray, tag: str) -> dict:
        m = tanimoto_retrieval_metrics(q_emb, g_smi_tr, fps_te, fps_tr, RETR_KS)
        m["variant"] = tag
        print(f"  {tag:22s}  " +
              "  ".join(f"Tan@{k}={m[f'Tan@{k}']:.4f}" for k in RETR_KS))
        return m

    rows = []
    with torch.no_grad():
        rows.append(eval_variant(project_batch(proj_s2,   X_s2_te),   "Stage 2 (ours)"))
        rows.append(eval_variant(project_batch(proj_s1,   X_s1_te),   "Stage 1"))
        rows.append(eval_variant(project_batch(proj_rand, X_rand_te), "Random"))

    # Upper bound: SMILES embedding → SMILES gallery (no modality gap)
    rows.append(eval_variant(g_smi_te, "SMILES (upper bound)"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "smiles_retrieval_results.csv", index=False)
    print(f"\n[OK] {OUT_DIR / 'smiles_retrieval_results.csv'}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    metric_cols = [f"Tan@{k}" for k in RETR_KS]
    colors = ["steelblue", "seagreen", "lightgray", "salmon"]
    variants = df["variant"].tolist()

    x = np.arange(len(metric_cols))
    bar_w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(9, 5))
    for vi, (vname, color) in enumerate(zip(variants, colors)):
        row = df[df["variant"] == vname].iloc[0]
        vals = [row[m] for m in metric_cols]
        offset = (vi - len(variants) / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=vname, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Tanimoto@{k}" for k in RETR_KS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Tanimoto similarity of top-k retrieved")
    ax.set_title("Cross-modal SMILES retrieval — Tanimoto@k\n"
                 "image embedding → nearest SMILES in library (Morgan ECFP4, 2048 bits)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = OUT_DIR / "smiles_retrieval_results.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_png}")
    print(f"\n[DONE] {OUT_DIR}")


if __name__ == "__main__":
    main()
