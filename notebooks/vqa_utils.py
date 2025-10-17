import math, random, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from fm_utils import *

SEED = 6740
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# --- Tiny question encoder: HuggingFace or fallback GRU ---
try:
    from transformers import AutoModel, AutoTokenizer
    _HAS_HF = True
except Exception:
    _HAS_HF = False

class SimpleTextEnc(nn.Module):
    """Fallback: word-embed + GRU → pooled vector."""
    def __init__(self, vocab_size=30000, dim=256, out_dim=384):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2*dim, out_dim)
    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)                       # (B,L,D)
        x, _ = self.gru(x)                      # (B,L,2D)
        x = (x * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-6)
        return self.proj(x)                     # (B,out_dim)

class HFTextEnc(nn.Module):
    def __init__(self, name: str, out_dim: int):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        hid = self.model.config.hidden_size
        self.proj = nn.Linear(hid, out_dim)
        self.name = name
    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, return_dict=True)
        # mean pool
        x = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-6)
        return self.proj(x)

# --- Cross-attention block: question queries visual tokens ---
class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 6, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, int(d_model*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model*mlp_ratio), d_model),
        )
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, 1, D)   kv: (B, N, D)
        qn = self.q_norm(q); kvn = self.kv_norm(kv)
        h, _ = self.attn(qn, kvn, kvn)         # (B, 1, D)
        h = h + q
        h = h + self.mlp(h)
        return h                                # (B, 1, D)

@dataclass
class VQASchema:
    # Define which heads we have
    num_yesno: int = 2
    cls_spaces: Dict[str, List[str]] = None  # e.g. {"organism": ["mouse","human","-"], "polarity":["positive","negative","-"]}
    num_bins_numeric: int = 32               # for binned numbers; set 0 to skip

class MSIVQA(nn.Module):
    """
    Wraps your MSI encoder + text encoder + cross-attn,
    exposes multi-head answers.
    """
    def __init__(self, msi_encoder, dim: int, schema: VQASchema,
                 text_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = msi_encoder         # your fm_MSI_Encoder_Timm (teacher or student in eval)
        self.dim = dim
        # text encoder
        if _HAS_HF:
            self.text_enc = HFTextEnc(text_model, out_dim=dim)
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(text_model)
        else:
            self.text_enc = SimpleTextEnc(out_dim=dim)
            self.tok = None  # you can plug your own tokenizer

        # cross-attn (1 query = question; keys/values = visual tokens)
        self.xattn = CrossAttnBlock(d_model=dim, n_heads=min(8, dim//64))

        # heads
        self.head_yesno = nn.Linear(dim, schema.num_yesno)
        self.cls_heads = nn.ModuleDict()
        self.spaces = {}
        schema.cls_spaces = schema.cls_spaces or {}
        for name, vocab in schema.cls_spaces.items():
            self.cls_heads[name] = nn.Linear(dim, len(vocab))
            self.spaces[name] = vocab

        self.num_bins = schema.num_bins_numeric
        self.head_bins = nn.Linear(dim, self.num_bins) if self.num_bins > 0 else None
        self.head_reg  = nn.Linear(dim, 1)            # optional direct regression

    # ---- tokenization helper ----
    def encode_text(self, questions: List[str], device):
        if self.tok is None:
            # naive whitespace tokenizer to ids in [0, 29999]
            max_len = 48
            ids = torch.zeros((len(questions), max_len), dtype=torch.long, device=device)
            mask= torch.zeros_like(ids)
            for i,q in enumerate(questions):
                toks = q.lower().split()[:max_len]
                for j,t in enumerate(toks):
                    ids[i,j] = (abs(hash(t)) % 29999) + 1
                    mask[i,j]= 1
            return ids, mask
        enc = self.tok(questions, padding=True, truncation=True, max_length=64, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def forward(self, patch: torch.Tensor, mz: torch.Tensor, questions: List[str]):
        """
        patch: (B,C,H,W) in [0,1]
        mz:    (B,C)
        """
        device = patch.device
        ids, mask = self.encode_text(questions, device)
        q_vec = self.text_enc(ids, mask)                  # (B,D)
        q_vec = q_vec.unsqueeze(1)                        # (B,1,D)

        # ---- MSI features ----
        # Ask encoder for tokens; ensure we can reconstruct (B, K, T, D)
        with torch.no_grad():
            self.encoder.eval()
            ps = getattr(self.encoder, "patch_size", 16)
            Ht = patch.shape[-2] // ps
            Wt = patch.shape[-1] // ps
            assert Ht * ps == patch.shape[-2] and Wt * ps == patch.shape[-1], \
                f"Input size {tuple(patch.shape[-2:])} must be divisible by patch_size={ps}"

            pos_2d = sine_pos2d(Ht, Wt, getattr(self.encoder, "embed_dim")).to(
                patch.device, dtype=patch.dtype
            )

            img_emb, marker_emb, token_emb, hw = self.encoder(
                patch, mz, pos_enc_2d=pos_2d, channel_dropout_p=0.0, out="features"
            )
        B, T, KD = token_emb.shape
        D = KD // mz.shape[1]
        K = mz.shape[1]
        tokens = token_emb.view(B, T, K, D).permute(0,2,1,3).contiguous().view(B, K*T, D)  # (B, K*T, D)

        # ---- cross-attend question → visual tokens ----
        h = self.xattn(q_vec, tokens).squeeze(1)          # (B,D)

        # ---- heads ----
        logits_yesno = self.head_yesno(h)                 # (B,2)
        logits_cls = {name: head(h) for name, head in self.cls_heads.items()}
        logits_bins = self.head_bins(h) if self.head_bins is not None else None
        y_reg = self.head_reg(h).squeeze(-1)              # (B,)

        return {
            "yesno": logits_yesno,
            "cls": logits_cls,
            "bins": logits_bins,
            "reg":  y_reg,
            "q":    q_vec.squeeze(1),
            "img":  img_emb,          # may be useful for auxiliary losses
        }

import os, math, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from functools import lru_cache

rng = np.random.default_rng(6740)

# ---------------- I/O & preprocessing helpers ----------------
def _load_npz(path):
    with np.load(path, mmap_mode="r") as z:
        patch = z["patch"].astype(np.float32)  # (C,H,W)
        mz    = z["mz"].astype(np.float32)     # (C,)
    if patch.max() > 1.0:  # uint16-ish → [0,1]
        patch /= 65535.0
    return patch, mz

def _center_crop_pad_np(patch, out_h, out_w, pad_value=0.0):
    # patch: (C,H,W) → fixed (C,out_h,out_w)
    C, H, W = patch.shape
    # crop
    if H > out_h:
        i = (H - out_h) // 2
        patch = patch[:, i:i+out_h, :]
        H = out_h
    if W > out_w:
        j = (W - out_w) // 2
        patch = patch[:, :, j:j+out_w]
        W = out_w
    # pad
    pad_h = max(0, out_h - H)
    pad_w = max(0, out_w - W)
    if pad_h or pad_w:
        top  = pad_h // 2;  bot  = pad_h - top
        left = pad_w // 2;  right = pad_w - left
        patch = np.pad(patch, ((0,0), (top,bot), (left,right)),
                       mode="constant", constant_values=pad_value)
    return patch

def _select_or_pad_channels_np(patch, mz, K):
    # patch: (C,H,W), mz: (C,) → fixed K
    C, H, W = patch.shape
    if C == 0:
        patch = np.zeros((1, H, W), np.float32)
        mz    = np.zeros((1,), np.float32)
        C = 1
    var = patch.reshape(C, -1).var(axis=1)
    anchor = int(var.argmax())
    order = np.argsort(-var)
    picks = [anchor] + [int(i) for i in order if int(i) != anchor]
    if len(picks) >= K:
        picks = picks[:K]
        return patch[picks], mz[picks]
    else:
        need = K - len(picks)
        picks = picks + [anchor]*need
        return patch[picks], mz[picks]

# ---------------- text normalization ----------------
def _canon(x: str) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    if s in ("", "none", "nan", "na", "null"):
        return "unknown"
    # normalize some common polarity variants
    if s in ("positive", "+", "pos", "positive ion mode", "positive mode"):
        return "positive"
    if s in ("negative", "-", "neg", "negative ion mode", "negative mode"):
        return "negative"
    return s

# Map logical task keys → DataFrame column names
TASK_TO_COLUMN = {
    "organism":         "organism",
    "polarity":         "polarity",
    "organ":            "Organism_Part",   # <- your manifest example
    "condition":        "Condition",       # <- your manifest example
    "analyzerType":     "analyzerType",
    "ionisationSource": "ionisationSource"
}

# Human-readable question prompts
TASK_QUESTION = {
    "organism":         "What organism is this sample?",
    "polarity":         "What is the ionization polarity?",
    "organ":            "Which organ (organism part) is this sample from?",
    "condition":        "What is the sample condition?",
    "analyzerType":     "What analyzer type was used?",
    "ionisationSource": "What ionisation source was used?"
}

def _build_or_validate_cls_spaces(df: pd.DataFrame, cls_spaces: dict | None):
    """
    Ensure each task has a vocabulary list (lowercased) and includes 'unknown' as a catch-all.
    If missing, infer from df unique values.
    """
    out = {}
    for task, col in TASK_TO_COLUMN.items():
        # infer if needed
        if cls_spaces is None or task not in cls_spaces:
            if col in df.columns:
                vals = df[col].astype(str).map(_canon).unique().tolist()
            else:
                vals = []
            vocab = sorted(set(vals)) if vals else []
            if "unknown" not in vocab:
                vocab.append("unknown")
            out[task] = list(vocab)
        else:
            vocab = [_canon(v) for v in cls_spaces[task]]
            if "unknown" not in vocab:
                vocab.append("unknown")
            out[task] = list(dict.fromkeys(vocab))  # de-dupe preserve order
    return out

# ---------------- Dataset ----------------
class VQADatasetLazy(Dataset):
    """
    Classification-only VQA dataset.
    Returns ONE QA per item with fixed (K,H,W), for tasks:
      organism, polarity, organ, condition, analyzerType, ionisationSource
    """
    def __init__(self, index_df: pd.DataFrame, cls_spaces: dict | None = None,
                 samples_per_epoch: int = 4, target_h: int = 256, target_w: int = 256,
                 k_target: int = 4):
        self.df = index_df.reset_index(drop=True).copy()

        # Normalize all potential columns up-front if they exist
        for task, col in TASK_TO_COLUMN.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(_canon)

        self.cls_spaces = _build_or_validate_cls_spaces(self.df, cls_spaces)

        # Which tasks are actually available (column present)?
        self.available_tasks = [
            t for t, col in TASK_TO_COLUMN.items()
            if col in self.df.columns
        ]
        if not self.available_tasks:
            raise ValueError("None of the expected columns are present: "
                             f"{list(TASK_TO_COLUMN.values())}")

        self.samples_per_epoch = int(max(1, samples_per_epoch))
        self.target_h, self.target_w = int(target_h), int(target_w)
        self.k_target = int(k_target)

        if "sample_path" not in self.df.columns:
            raise ValueError("index_df must contain a 'sample_path' column pointing to .npz samples.")

    def __len__(self):
        return len(self.df) * self.samples_per_epoch

    def _pick_task(self):
        # Uniformly sample among available tasks
        return rng.choice(self.available_tasks)

    def _prepare_patch_mz(self, path):
        patch, mz = _load_npz(path)                                      # (C,H,W), (C,)
        patch = _center_crop_pad_np(patch, self.target_h, self.target_w) # fixed H,W
        patch, mz = _select_or_pad_channels_np(patch, mz, self.k_target) # fixed K
        return torch.from_numpy(patch), torch.from_numpy(mz)

    def __getitem__(self, i):
        j = i % len(self.df)
        row = self.df.iloc[j]
        path = row["sample_path"]

        task = self._pick_task()
        col  = TASK_TO_COLUMN[task]
        vocab = self.cls_spaces[task]

        # Canonicalize target (already canon'd in __init__, but be safe)
        tgt = _canon(row.get(col, "unknown"))
        idx = vocab.index(tgt) if tgt in vocab else vocab.index("unknown")

        q = TASK_QUESTION[task]
        patch_t, mz_t = self._prepare_patch_mz(path)

        # e.g., returns {"y_cls_organism": LongTensor([...])}
        y_key = f"y_cls_{task}"
        return {
            "patch": patch_t,
            "mz": mz_t,
            "question": q,
            y_key: torch.tensor(idx, dtype=torch.long),
        }

def load_vqa(run_dir: str, teacher, dim: int):
    from pathlib import Path
    import torch

    run = Path(run_dir)

    # choose a checkpoint: best → last → latest step_*.pt
    ckpt_path = run / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run / "last.pt"
    if not ckpt_path.exists():
        step_ckpts = sorted(run.glob("step_*.pt"))
        if step_ckpts:
            ckpt_path = step_ckpts[-1]
        else:
            raise FileNotFoundError(f"No best.pt/last.pt/step_*.pt in {run}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # strip potential 'module.' prefixes from DDP saves
    def _strip_module(sd: dict):
        return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

    sd = _strip_module(ckpt["model"])

    # ---- Decide bins head presence/size ----
    # Prefer explicit value if present; otherwise infer from weights.
    num_bins = int(ckpt.get("num_bins_numeric", -1))
    if num_bins < 0:
        if "head_bins.weight" in sd:
            num_bins = int(sd["head_bins.weight"].shape[0])
        else:
            num_bins = 0  # CLS-only

    schema = VQASchema(cls_spaces=ckpt.get("cls_spaces", {}), num_bins_numeric=num_bins)

    # device fallback if DEVICE not defined
    dev = globals().get("DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    vqa = MSIVQA(teacher, dim=int(dim), schema=schema).to(dev)

    # Lenient load, but we’ll report anything unexpected
    missing, unexpected = vqa.load_state_dict(sd, strict=False)

    # Ignore numeric-head keys if we intentionally built CLS-only
    ignorable = set()
    if num_bins == 0:
        ignorable.update({"head_bins.weight", "head_bins.bias", "head_reg.weight", "head_reg.bias"})

    real_missing = [k for k in missing if k not in ignorable]

    if real_missing:
        print(f"[VQA] Missing keys: {len(real_missing)} (showing 5): {real_missing[:5]}")
    if unexpected:
        print(f"[VQA] Unexpected keys: {len(unexpected)} (showing 5): {unexpected[:5]}")

    vqa.eval()
    for p in vqa.parameters():
        p.requires_grad_(False)

    return vqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 6740
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------- INFERENCE (multi-question) -----------------
@torch.inference_mode()
def vqa_on_npz_multi(vqa, npz_path: str, questions, target_hw=256, k_target=4):
    """Runs VQA on one .npz for a list of questions; returns results + the fixed patch/mz for self-checks."""
    if isinstance(questions, str): questions = [questions]
    with np.load(npz_path, mmap_mode="r") as z:
        patch = z["patch"].astype(np.float32)  # (C,H,W)
        mz    = z["mz"].astype(np.float32)     # (C,)
    if patch.max() > 1.0: patch /= 65535.0

    # Match training preprocessing
    patch_fix = _center_crop_pad_np(patch, target_hw, target_hw)
    patch_fix, mz_fix = _select_or_pad_channels_np(patch_fix, mz, k_target)

    pt = torch.from_numpy(patch_fix)[None].to(DEVICE)  # (1,K,H,W)
    mzt = torch.from_numpy(mz_fix)[None].to(DEVICE)    # (1,K)

    results = []
    for q in questions:
        out = vqa(pt, mzt, [q])
        item = {"question": q, "result": {}}

        # yes/no
        yesno = F.softmax(out["yesno"], dim=-1)[0].tolist()
        item["result"]["yesno"] = {
            "pred": "yes" if yesno[1] >= yesno[0] else "no",
            "confidence": float(max(yesno)),
            "probs": {"no": float(yesno[0]), "yes": float(yesno[1])},
        }

        # class heads
        cls_block = {}
        for name, logits in out["cls"].items():
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            idx = int(probs.argmax())
            label = vqa.spaces[name][idx]
            cls_block[name] = {
                "pred": label, "index": idx, "confidence": float(probs[idx]),
                "top3": [(vqa.spaces[name][int(i)], float(probs[int(i)])) for i in np.argsort(-probs)[:3]]
            }
        item["result"]["cls"] = cls_block

        # numeric
        if out["bins"] is not None:
            bprobs = F.softmax(out["bins"], dim=-1)[0].cpu().numpy()
            item["result"]["bins"] = {"pred_bin": int(bprobs.argmax()), "confidence": float(bprobs.max())}
        else:
            item["result"]["bins"] = None

        # regression aux
        item["result"]["reg"] = float(out["reg"][0].item())
        results.append(item)

    return results, patch_fix, mz_fix

import json, re
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Normalization (match training/app) ----------
def _canon_base(x):
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    return "unknown" if s in ("", "none", "nan", "na", "null") else s

def normalize_organism(x):
    s = _canon_base(x)
    if s == "unknown": return s
    if "mus musculus" in s or s in {"mouse", "mouse brain", "mus", "m. musculus"}:
        return "mus musculus"
    if "homo sapiens" in s or s in {"human", "h. sapiens"}:
        return "homo sapiens"
    return "unknown"

def normalize_polarity(x):
    s = _canon_base(x)
    if s in {"pos", "+", "positive", "positive ion mode", "positive mode"}:  return "positive"
    if s in {"neg", "-", "negative", "negative ion mode", "negative mode"}:  return "negative"
    return "unknown"

def normalize_text(x):
    return _canon_base(x)

# ---------- Intent mapping (natural question → CLS head) ----------
INTENT = {
    "organism":         re.compile(r"\b(organism|species)\b", re.I),
    "polarity":         re.compile(r"\bpolari(?:ty)?\b", re.I),
    "organ":            re.compile(r"\b(organ(?:ism)?\s*part|organ\b|tissue)\b", re.I),
    "condition":        re.compile(r"\b(condition|status)\b", re.I),
    "analyzerType":     re.compile(r"\banaly[sz]er(?:\s*type)?\b", re.I),
    "ionisationSource": re.compile(r"\bion(i[sz]ation)?\s*source\b|\bion[i|z]iser\b", re.I)
}

def detect_head(question: str) -> str | None:
    q = (question or "").strip()
    if not q: return None
    for head, rx in INTENT.items():
        if rx.search(q): return head
    return None

# ---------- JSON extraction ----------
def _load_metadata_json(p: str | Path) -> dict:
    p = Path(p)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_fields_from_json(meta: dict) -> dict:
    """
    Pulls both top-level fields and nested 'metadata_uploaded' blocks.
    Returns a dict with original (raw) values plus normalized counterparts.
    """
    top = meta or {}
    up  = (top.get("metadata_uploaded") or {})  # nested dicts

    sample_info = up.get("Sample_Information", {}) or {}
    ms_analysis = up.get("MS_Analysis", {}) or {}

    out_raw = {
        # dataset-id
        "dataset_id": top.get("dataset_id") or "",

        # organism: prefer nested Sample_Information.Organism, else top-level 'organism'
        "organism": sample_info.get("Organism", top.get("organism")),

        # polarity: prefer nested MS_Analysis.Polarity, else top-level 'polarity'
        "polarity": ms_analysis.get("Polarity", top.get("polarity")),

        # organ / condition from Sample_Information
        "Organism_Part": sample_info.get("Organism_Part"),
        "Condition":     sample_info.get("Condition"),

        # analyzer / ion source: nested MS_Analysis preferred, then top-level aliases
        "analyzerType":     ms_analysis.get("Analyzer", top.get("analyzerType")),
        "ionisationSource": ms_analysis.get("Ionisation_Source", top.get("ionisationSource")),
    }

    out_norm = {
        "organism_norm":         normalize_organism(out_raw["organism"]),
        "polarity_norm":         normalize_polarity(out_raw["polarity"]),
        "Organism_Part_norm":    normalize_text(out_raw["Organism_Part"]),
        "Condition_norm":        normalize_text(out_raw["Condition"]),
        "analyzerType_norm":     normalize_text(out_raw["analyzerType"]),
        "ionisationSource_norm": normalize_text(out_raw["ionisationSource"]),
        "sum_formula_norm":      "unknown",  # not available here
    }

    return {**out_raw, **out_norm}

# Map CLS head → keys in the extracted dict
TASK_TO_KEYS = {
    "organism":         ("organism_norm", "organism"),
    "polarity":         ("polarity_norm", "polarity"),
    "organ":            ("Organism_Part_norm", "Organism_Part"),
    "condition":        ("Condition_norm", "Condition"),
    "analyzerType":     ("analyzerType_norm", "analyzerType"),
    "ionisationSource": ("ionisationSource_norm", "ionisationSource")
}

def gt_answers_for_questions_json(questions: list[str], sample_path: str | Path):
    """
    Given a list of natural-language questions and a path to metadata_full.json,
    returns a tidy DataFrame with normalized & original ground-truth answers.
    """
    meta = _load_metadata_json(sample_path)
    ex   = _extract_fields_from_json(meta)

    records = []
    for q in questions:
        head = detect_head(q)
        if head is None:
            records.append({
                "dataset_id": ex.get("dataset_id", ""),
                "question": q,
                "head": "(unknown)",
                "gt_normalized": "",
                "gt_original": "",
                "source": Path(sample_path).as_posix(),
            })
            continue

        nkey, okey = TASK_TO_KEYS[head]
        nval = ex.get(nkey, "unknown")
        oval = ex.get(okey, "unknown")
        # Ensure string-ish
        nval = "unknown" if (nval is None or (isinstance(nval, float) and np.isnan(nval))) else str(nval)
        oval = "unknown" if (oval is None or (isinstance(oval, float) and np.isnan(oval))) else str(oval)

        records.append({
            "dataset_id": ex.get("dataset_id", ""),
            "question": q,
            "head": head,
            "gt_normalized": nval,
            "gt_original": oval,
            "source": Path(sample_path).as_posix(),
        })

    return pd.DataFrame.from_records(records)