import os, math, copy, random, argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

# -------------------------
# Positional / m/z encodings
# -------------------------

def sine_pos2d(h_tokens: int, w_tokens: int, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    y, x = torch.meshgrid(torch.arange(h_tokens), torch.arange(w_tokens), indexing='ij')
    y = y.flatten().float()
    x = x.flatten().float()
    omega = torch.arange(dim // 4).float() / (dim // 4)
    omega = 1.0 / (10000 ** omega)
    y_enc = torch.einsum('t,d->td', y, omega)
    x_enc = torch.einsum('t,d->td', x, omega)
    pe = torch.zeros(h_tokens * w_tokens, dim)
    pe[:, 0:dim//4]         = torch.sin(y_enc)
    pe[:, dim//4:dim//2]    = torch.cos(y_enc)
    pe[:, dim//2:3*dim//4]  = torch.sin(x_enc)
    pe[:, 3*dim//4:dim]     = torch.cos(x_enc)
    return pe

def sine_mz_encoding(mz: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim % 2 == 0
    B, K = mz.shape
    mz_norm = (mz - mz.amin(dim=1, keepdim=True)) / (mz.amax(dim=1, keepdim=True) - mz.amin(dim=1, keepdim=True) + 1e-6)
    idx = torch.arange(dim // 2, device=mz.device).float()
    denom = 10000 ** (idx / (dim // 2))
    arg = mz_norm[..., None] / denom
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)

# -------------------------
# Collate helpers (works with dict, (patch,mz), or .npz file path)
# -------------------------

def _to_patch_mz(item):
    if isinstance(item, dict):
        patch = item['patch']; mz = item['mz']
        if isinstance(patch, np.ndarray): patch = torch.from_numpy(patch).float()
        if isinstance(mz,    np.ndarray): mz    = torch.from_numpy(mz).float()
        return patch.float(), mz.float()
    if isinstance(item, (list, tuple)) and len(item) == 2:
        patch, mz = item
        if isinstance(patch, np.ndarray): patch = torch.from_numpy(patch).float()
        if isinstance(mz,    np.ndarray): mz    = torch.from_numpy(mz).float()
        return patch.float(), mz.float()
    if isinstance(item, str):
        data = np.load(item)
        patch = torch.from_numpy(data['patch']).float()
        mz    = torch.from_numpy(data['mz']).float()
        return patch, mz
    raise TypeError(f"Unsupported dataset item type: {type(item)}")

def collate_as_list_of_dicts(batch):
    out = []
    for item in batch:
        patch, mz = _to_patch_mz(item)
        if isinstance(patch, torch.Tensor): patch = patch.contiguous()
        if isinstance(mz, torch.Tensor):    mz    = mz.contiguous()
        if patch.ndim != 3: raise ValueError(f"patch must be (C,H,W); got {tuple(patch.shape)}")
        if mz.ndim != 1:    raise ValueError(f"mz must be (C,); got {tuple(mz.shape)}")
        C, H, W = patch.shape
        if mz.shape[0] != C: raise ValueError(f"channel mismatch: patch C={C} vs mz len={mz.shape[0]}")
        out.append({'patch': patch, 'mz': mz})
    return out

# -------------------------
# Shared per-channel patch embedding (1 input channel)
# -------------------------

class SharedPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 384, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, K, H, W = x.shape
        x = x.view(B*K, 1, H, W)
        t = self.proj(x)                          # (B*K, D, H/ps, W/ps)
        t = t.permute(0, 2, 3, 1).contiguous()    # (B*K, Ht, Wt, D)
        Ht, Wt, D = t.shape[1:]
        t = t.view(B, K, Ht*Wt, D)                # (B, K, T, D)
        return t, Ht, Wt

# -------------------------
# DINO-like losses & EMA
# -------------------------

@dataclass
class DinoParams:
    t_student: float = 0.1
    t_teacher: float = 0.04

def dino_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, params: DinoParams) -> torch.Tensor:
    t = F.softmax(t_logits / params.t_teacher, dim=-1).detach()
    s = F.log_softmax(s_logits / params.t_student, dim=-1)
    return -(t * s).sum(dim=-1).mean()

def masked_mse(s_tokens: torch.Tensor, t_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = s_tokens[mask] - t_tokens[mask].detach()
    return (diff.pow(2).sum(dim=-1)).mean()

@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, m: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

# -------------------------
# Masking: same spatial positions across channels
# -------------------------

def make_mask_same_spatial(B: int, K: int, Ht: int, Wt: int, ratio: float, device) -> torch.Tensor:
    T = Ht * Wt
    num_mask_spatial = max(1, int(T * ratio))
    mask = torch.zeros(B, K*T, dtype=torch.bool, device=device)
    for b in range(B):
        idx_spatial = torch.randperm(T, device=device)[:num_mask_spatial]
        idx_all = torch.cat([idx_spatial + k*T for k in range(K)], dim=0)
        mask[b, idx_all] = True
    return mask

# -------------------------
# Augmentations & two-view maker
# -------------------------

def random_crop(x: torch.Tensor, out_size: int) -> torch.Tensor:
    K, H, W = x.shape
    i = random.randint(0, H - out_size)
    j = random.randint(0, W - out_size)
    return x[:, i:i+out_size, j:j+out_size]

def spatial_augs(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5: x = torch.flip(x, dims=[2])
    if random.random() < 0.5: x = torch.flip(x, dims=[1])
    k = random.randint(0, 3)
    if k > 0: x = torch.rot90(x, k, dims=[1,2])
    return x

def intensity_jitter(x: torch.Tensor, scale_range=(0.8,1.2), bias_sigma=0.05, noise_sigma=0.01) -> torch.Tensor:
    K = x.shape[0]
    device = x.device
    scales = torch.empty(K, device=device).uniform_(*scale_range)
    biases = torch.randn(K, device=device).mul_(bias_sigma)
    x = x * scales[:, None, None] + biases[:, None, None]
    if noise_sigma > 0:
        x = x + torch.randn_like(x) * noise_sigma
    return x

def pick_anchor_channel(intens: torch.Tensor) -> int:
    C = intens.shape[0]
    var = intens.contiguous().reshape(C, -1).var(dim=1)  # <- reshape instead of view
    return int(var.argmax().item())

def make_two_views(
    patch: torch.Tensor,
    mz: torch.Tensor,
    K: int,
    input_size: int,
    crop_size: int
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    # Ensure (C, input_size, input_size) via center pad/crop (no assert)
    patch = _center_crop_or_pad_square(patch, input_size, pad_value=0.0)

    # Select K channels (repeat anchor if needed)
    xK, mz_sel = _select_K_channels_with_repeat(patch, mz, K)

    # Define view maker
    def one_view(x_in: torch.Tensor) -> torch.Tensor:
        # Safe random crop when crop_size <= input_size
        if crop_size < input_size:
            i = random.randint(0, input_size - crop_size)
            j = random.randint(0, input_size - crop_size)
            x_aug = x_in[:, i:i+crop_size, j:j+crop_size]
        else:
            # if equal, no-op crop
            x_aug = x_in

        x_aug = spatial_augs(x_aug)
        x_aug = intensity_jitter(x_aug)
        return x_aug

    x1 = one_view(xK)
    x2 = one_view(xK)
    return (x1, mz_sel), (x2, mz_sel)

# -------------------------
# MSI dataset placeholder (replace with your loader)
# -------------------------

class MSIPatchDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

# -------------------------
# Timm integration: load ViT and adapt
# -------------------------

def _interpolate_conv_weight_2d(w: torch.Tensor, k_out: int, k_in: int, target_ks: int) -> torch.Tensor:
    # w: (out_ch, in_ch, ks, ks). Resize spatially to target_ks via bicubic.
    if w.shape[-1] == target_ks:
        return w
    w = F.interpolate(w, size=(target_ks, target_ks), mode='bicubic', align_corners=True)
    return w

def _adapt_patch_embed_from_rgb_to_gray(w_rgb: torch.Tensor) -> torch.Tensor:
    # w_rgb: (D, 3, ks, ks) -> (D, 1, ks, ks) by averaging channels
    return w_rgb.mean(dim=1, keepdim=True)

class fm_MSI_Encoder_Timm(nn.Module):
    def __init__(self, timm_blocks: nn.ModuleList, timm_norm: nn.Module,
                 embed_dim: int, patch_size: int, proto_out: int = 8192, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.blocks = timm_blocks
        self.norm = timm_norm
        self.patch_embed = SharedPatchEmbed(embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.prototypes = nn.Linear(embed_dim, proto_out, bias=False)
        self.pos_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,                # (B, K, H, W)
        mz: torch.Tensor,               # (B, K)
        pos_enc_2d: torch.Tensor,       # (T, D) with T = Ht*Wt
        channel_dropout_p: float = 0.0,
        out: str = "train"              # "train" (default) or "features"
    ):
        B, K, H, W = x.shape

        # channel dropout (student-only during training)
        if self.training and channel_dropout_p > 0.0 and K > 1:
            keep_mask = (torch.rand(B, K, device=x.device) > channel_dropout_p).float()
            force_one = (keep_mask.sum(dim=1, keepdim=True) == 0)
            if force_one.any():
                idx = torch.randint(0, K, (force_one.sum().item(),), device=x.device)
                keep_mask[force_one.squeeze(), idx] = 1.0
            x = x * keep_mask[:, :, None, None]

        # patch -> tokens
        patch_tokens, Ht, Wt = self.patch_embed(x)                   # (B, K, T, D), T=Ht*Wt
        T = patch_tokens.shape[2]
        pos_enc = pos_enc_2d.to(x.device).unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
        mz_enc  = sine_mz_encoding(mz, self.embed_dim).unsqueeze(2)  # (B,K,1,D)

        tokens = patch_tokens + pos_enc + mz_enc                     # (B, K, T, D)
        tokens_flat = tokens.view(B, K*T, self.embed_dim)            # (B, K*T, D)

        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)                       # (B, 1, D)
        seq = torch.cat([cls, tokens_flat], dim=1)                   # (B, 1+K*T, D)
        seq = self.pos_drop(seq)

        # transformer
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)

        cls_out = seq[:, 0, :]                # (B, D)
        tok_out = seq[:, 1:, :]               # (B, K*T, D)
        tok_out_kt = tok_out.view(B, K, T, self.embed_dim)  # (B, K, T, D)

        if out == "features":
            # 1) image-level (CLS)
            image_emb = cls_out                              # (B, D)

            # 2) marker-specific (mean over spatial tokens per marker)
            marker_emb = tok_out_kt.mean(dim=2)              # (B, K, D)

            # 3) token-specific (concat across markers at each spatial pos)
            token_emb = tok_out_kt.permute(0, 2, 1, 3)       # (B, T, K, D)
            token_emb = token_emb.contiguous().view(B, T, K * self.embed_dim)  # (B, T, K*D)

            return image_emb, marker_emb, token_emb, (Ht, Wt)

        # default "train" path: keep your original outputs for DINO + iBOT
        logits = self.prototypes(cls_out)     # (B, proto_out)
        return logits, tok_out, (Ht, Wt)

def load_timm_vit(vit_name: str, pretrained: bool = True):
    try:
        import timm
    except ImportError as e:
        raise RuntimeError("Please install timm: pip install timm") from e
    model = timm.create_model(vit_name, pretrained=pretrained)
    # Basic checks
    if not hasattr(model, 'blocks') or not hasattr(model, 'norm'):
        raise ValueError(f"Model {vit_name} is not a ViT-like backbone with blocks/norm.")
    # infer embed dim from first block
    embed_dim = model.blocks[0].norm1.normalized_shape[0]
    # patch size (guess if available)
    patch_size = getattr(getattr(model, 'patch_embed', None), 'patch_size', None)
    if isinstance(patch_size, tuple): patch_size = patch_size[0]
    return model, embed_dim, patch_size

def build_msi_encoder_from_timm(vit_name: str, target_patch: int, proto_out: int, dropout: float, use_pretrained=True):
    timm_model, embed_dim, src_patch = load_timm_vit(vit_name, pretrained=use_pretrained)

    # Create MSI encoder using timm blocks + norm
    encoder = fm_MSI_Encoder_Timm(
        timm_blocks=timm_model.blocks,
        timm_norm=timm_model.norm,
        embed_dim=embed_dim,
        patch_size=target_patch,
        proto_out=proto_out,
        dropout=dropout,
    )

    # Adapt patch embed weights (if timm has them)
    with torch.no_grad():
        if hasattr(timm_model, 'patch_embed') and hasattr(timm_model.patch_embed, 'proj'):
            w = timm_model.patch_embed.proj.weight.data.clone()  # (D, 3, ks, ks) typically
            if w.shape[1] == 3:
                w = _adapt_patch_embed_from_rgb_to_gray(w)       # (D,1,ks,ks)
            # spatially interpolate kernel if patch size differs
            if w.shape[-1] != target_patch:
                w = _interpolate_conv_weight_2d(w, w.shape[0], w.shape[1], target_patch)
            # assign
            if encoder.patch_embed.proj.weight.shape == w.shape:
                encoder.patch_embed.proj.weight.copy_(w)
            else:
                # Fallback: fan-in/out match required
                print(f"[warn] Patch kernel shape mismatch; re-init used. got {w.shape} -> {tuple(encoder.patch_embed.proj.weight.shape)}")
        # Bias init from timm if shapes match
        if hasattr(timm_model, 'patch_embed') and hasattr(timm_model.patch_embed, 'proj'):
            b_src = timm_model.patch_embed.proj.bias
            if b_src is not None and encoder.patch_embed.proj.bias is not None and \
               encoder.patch_embed.proj.bias.shape == b_src.shape:
                encoder.patch_embed.proj.bias.copy_(b_src.data)

    return encoder, embed_dim

# -------------------------
# Optim utils
# -------------------------

def param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndimension() == 1 or n.endswith('bias') or 'norm' in n or 'bn' in n or 'ln' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]

def set_first_n_blocks_trainable(encoder: fm_MSI_Encoder_Timm, n_unfrozen: int):
    # Freeze all, then unfreeze first n_unfrozen? We want often to FREEZE the first N and train the rest.
    # We'll provide both helpers:
    for i, blk in enumerate(encoder.blocks):
        for p in blk.parameters():
            p.requires_grad = True if i >= n_unfrozen else False

def freeze_first_n_blocks(encoder: fm_MSI_Encoder_Timm, n_freeze: int):
    for i, blk in enumerate(encoder.blocks):
        req = False if i < n_freeze else True
        for p in blk.parameters():
            p.requires_grad = req

def unfreeze_all(encoder: fm_MSI_Encoder_Timm):
    for p in encoder.blocks.parameters():
        p.requires_grad = True

def _center_crop_or_pad_square(patch: torch.Tensor, target: int, pad_value: float = 0.0) -> torch.Tensor:
    assert patch.ndim == 3
    C, H, W = patch.shape
    pad_h = max(0, target - H); pad_w = max(0, target - W)
    if pad_h > 0 or pad_w > 0:
        top  = pad_h // 2; bot  = pad_h - top
        left = pad_w // 2; right = pad_w - left
        patch = F.pad(patch, (left, right, top, bot), value=pad_value)
        H, W = patch.shape[1], patch.shape[2]
    if H > target or W > target:
        i = (H - target) // 2; j = (W - target) // 2
        patch = patch[:, i:i+target, j:j+target]
    return patch.contiguous()   # <- add this

def _select_K_channels_with_repeat(patch: torch.Tensor, mz: torch.Tensor, K: int):
    # Returns K channels; if C<K, repeats the anchor channel to fill.
    C = patch.shape[0]
    var = patch.contiguous().reshape(C, -1).var(dim=1)   # <- reshape instead of view
    anchor = int(var.argmax().item())
    order = torch.argsort(var, descending=True).tolist()
    picks = [anchor] + [i for i in order if i != anchor]
    if len(picks) >= K:
        picks = picks[:K]
    else:
        picks = picks + [anchor] * (K - len(picks))
    return patch[picks], mz[picks]

class NPZPathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
    def __len__(self): 
        return len(self.paths)
    def __getitem__(self, idx):
        # Return the string path; collate will open the .npz and extract 'patch' & 'mz'
        return self.paths[idx]

# -------------------------
# Config
# -------------------------

@dataclass
class CFG:
    # model
    vit_name: str = "deit_small_distilled_patch16_224"
    patch_size: int = 16
    dim: Optional[int] = None  # inferred from timm model
    proto_out: int = 8192
    dropout: float = 0.0

    # training
    steps: int = 200_000
    warmup_steps: int = 8_000
    freeze_blocks_warmup: int = 4
    batch_size: int = 32
    grad_accum: int = 2
    lr_base: float = 1e-3
    wd: float = 0.05
    betas: Tuple[float,float] = (0.9, 0.95)
    amp_dtype: torch.dtype = torch.bfloat16
    ema_start: float = 0.996
    ema_target: float = 0.9995

    # views & masking
    crop_size: int = 224
    input_size: int = 256
    channels_per_view: int = 3
    mask_ratio_min: float = 0.30
    mask_ratio_max: float = 0.60
    channel_dropout_p: float = 0.33

    # dino temps (teacher temp warmed from hi->lo)
    t_student: float = 0.10
    t_teacher_start: float = 0.07
    t_teacher_end: float = 0.04

    # data loader
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 6740

    # misc
    save_path: str = "msi_fm_all_data.pt"

def collate_identity(batch):
    # return the list as-is (no stacking)
    return batch

import numpy as np
from pathlib import Path

def _load_item(it):
    # dict path
    if isinstance(it, dict):
        patch = it["patch"]
        mz    = it["mz"]
        path  = it.get("path") or it.get("sample_path", "")
        # -> torch.float32
        if isinstance(patch, np.ndarray): patch = torch.from_numpy(patch)
        if isinstance(mz, np.ndarray):    mz    = torch.from_numpy(mz)
        patch = patch.to(torch.float32, copy=False)  # <-- ensure float
        mz    = mz.to(torch.float32, copy=False)     # <-- ensure float
        return {"patch": patch, "mz": mz, "path": str(path)}

    # string/path -> load .npz
    p = str(it)
    d = np.load(p, allow_pickle=False, mmap_mode="r")
    patch = d["patch"]   # (K,H,W)
    mz    = d["mz"]      # (K,)
    # -> torch.float32
    patch = torch.from_numpy(patch).to(torch.float32, copy=False)
    mz    = torch.from_numpy(mz).to(torch.float32, copy=False)
    return {"patch": patch, "mz": mz, "path": p}

# -------------------------
# Training
# -------------------------
def train(
    cfg: CFG,
    dataset: Dataset,
    *,
    log_fn=None,                 # optional CSV logger callback
    checkpoint_dir=None,         # directory to save step_*.pt
    ckpt_every=None,             # save every N steps
):
    import math, time, os, random, copy
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)

    # -------------------------------
    # DataLoader: SAFE defaults
    # -------------------------------
    num_workers = int(getattr(cfg, "num_workers", 0))
    pin_memory  = bool(getattr(cfg, "pin_memory", True)) and torch.cuda.is_available()
    prefetch_factor = int(getattr(cfg, "prefetch_factor", 2)) if num_workers > 0 else None
    persistent_workers = bool(getattr(cfg, "persistent_workers", False)) if num_workers > 0 else False

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_identity,  # already provided
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    # -------------------------------
    # Build models
    # -------------------------------
    student, embed_dim = build_msi_encoder_from_timm(
        vit_name=cfg.vit_name,
        target_patch=cfg.patch_size,
        proto_out=cfg.proto_out,
        dropout=getattr(cfg, "dropout", 0.0),
        use_pretrained=True
    )
    if cfg.dim is None:
        cfg.dim = embed_dim

    student = student.to(device).train()

    # Re-init heads (projector/predictor/cls head) to encourage adaptation
    def _reinit_heads(m):
        for name, mod in m.named_modules():
            if any(k in name for k in ["projector", "predictor", "head"]):
                for p in mod.parameters():
                    if p.dim() > 1:
                        torch.nn.init.kaiming_normal_(p)
                    else:
                        torch.nn.init.zeros_(p)
    _reinit_heads(student)

    teacher = copy.deepcopy(student).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Optionally freeze a few earliest blocks briefly
    freeze_first_n_blocks(student, getattr(cfg, "freeze_blocks_warmup", 0))

    # LR scaling by effective batch
    eff_batch = cfg.batch_size * cfg.grad_accum
    base_lr = cfg.lr_base * (eff_batch / 256.0)

    optim = torch.optim.AdamW(
        param_groups(student, cfg.wd),   # already separates wd / no-wd
        lr=base_lr,
        betas=getattr(cfg, "betas", (0.9, 0.999))
    )

    # Cosine LR schedule with warmup
    total_steps  = int(cfg.steps)
    warmup_steps = int(min(getattr(cfg, "warmup_steps", 0), max(0, total_steps - 1)))

    def cosine_lr(step):
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * float(step + 1) / float(warmup_steps)
        # cosine decay after warmup
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * t))

    # AMP autocast
    if torch.cuda.is_available():
        amp_dtype = getattr(cfg, "amp_dtype", torch.bfloat16)
        scaler_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype)
    else:
        from contextlib import nullcontext
        scaler_ctx = nullcontext()

    # DINO params/schedules
    dino_params = DinoParams(t_student=cfg.t_student, t_teacher=cfg.t_teacher_start)

    # EMA momentum schedule (cosine ramp from ema_start -> ema_target)
    ema_start  = float(getattr(cfg, "ema_start", 0.996))
    ema_target = float(getattr(cfg, "ema_target", 0.9995))
    def ema_m(step):
        # ramp over entire training
        t = min(1.0, step / max(1, total_steps))
        return ema_target - (ema_target - ema_start) * (0.5 * (1.0 + math.cos(math.pi * t)))

    # Teacher temperature schedule (linear from start -> end)
    def teacher_temp(step):
        if step >= total_steps: step = total_steps - 1
        return cfg.t_teacher_end + (cfg.t_teacher_start - cfg.t_teacher_end) * (1.0 - step / max(1, total_steps))

    # Mask ratio schedule (linear min -> max)
    def mask_ratio(step):
        return cfg.mask_ratio_min + (cfg.mask_ratio_max - cfg.mask_ratio_min) * (step / max(1, total_steps))

    # Checkpoint dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Make sure we actually unfreeze if requested
    warmup_steps = int(min(getattr(cfg, "warmup_steps", 0), max(0, total_steps - 1)))

    t0 = time.time()
    step = 0
    epoch = 0
    student.train(); teacher.eval()

    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break

            # Build two views
            xs1, mzs1, xs2, mzs2 = [], [], [], []
            for it in batch:
                sample = _load_item(it)  # already provided
                full_patch = sample["patch"]        # (K,H,W)
                full_mz    = sample["mz"]           # (K,)
                (x1, mz1), (x2, mz2) = make_two_views(  # already provided
                    full_patch, full_mz,
                    cfg.channels_per_view, cfg.input_size, cfg.crop_size
                )
                xs1.append(x1); mzs1.append(mz1); xs2.append(x2); mzs2.append(mz2)

            x1  = torch.stack(xs1, dim=0).to(device, non_blocking=True)
            x2  = torch.stack(xs2, dim=0).to(device, non_blocking=True)
            mz1 = torch.stack(mzs1, dim=0).to(device, non_blocking=True)
            mz2 = torch.stack(mzs2, dim=0).to(device, non_blocking=True)

            Ht = cfg.crop_size // cfg.patch_size
            Wt = cfg.crop_size // cfg.patch_size
            pos_2d = sine_pos2d(Ht, Wt, cfg.dim)  # already provided

            # Unfreeze full backbone after warmup (if any blocks were frozen)
            if step == warmup_steps:
                unfreeze_all(student)  # already provided

            # Update schedules
            # LR
            cur_lr = cosine_lr(step)
            for pg in optim.param_groups:
                pg["lr"] = cur_lr

            # Temps & mask
            dino_params.t_teacher = teacher_temp(step)
            mratio = mask_ratio(step)

            with scaler_ctx:
                s_logits1, s_tokens1, _ = student(x1, mz1, pos_2d, channel_dropout_p=cfg.channel_dropout_p)
                s_logits2, s_tokens2, _ = student(x2, mz2, pos_2d, channel_dropout_p=cfg.channel_dropout_p)

                with torch.no_grad():
                    t_logits1, t_tokens1, _ = teacher(x1, mz1, pos_2d, channel_dropout_p=0.0)
                    t_logits2, t_tokens2, _ = teacher(x2, mz2, pos_2d, channel_dropout_p=0.0)

                # SSL losses (already provided helpers)
                L_dino = dino_loss(s_logits1, t_logits2, dino_params) + dino_loss(s_logits2, t_logits1, dino_params)
                B, S, D = s_tokens1.shape
                mask = make_mask_same_spatial(B, cfg.channels_per_view, Ht, Wt, mratio, device)
                L_mim = masked_mse(s_tokens1, t_tokens1, mask) + masked_mse(s_tokens2, t_tokens2, mask)
                loss = L_dino + L_mim

            # Grad accumulation
            (loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                # EMA teacher update
                update_ema(student, teacher, ema_m(step))  # already provided

            # ---- LOG EVERY STEP ----
            if log_fn is not None:
                try:
                    log_lr = optim.param_groups[0]["lr"]
                except Exception:
                    log_lr = None
                log_fn(
                    step=step,
                    epoch=epoch,
                    loss=float(loss.detach().item()) if torch.is_tensor(loss) else float(loss),
                    lr=float(log_lr) if log_lr is not None else None,
                    t_student=float(getattr(cfg, "t_student", None)) if hasattr(cfg, "t_student") else None,
                    t_teacher=float(dino_params.t_teacher),
                    proto_entropy=None,
                )

            # ---- PERIODIC CHECKPOINT ----
            if checkpoint_dir and ckpt_every and ((step + 1) % int(ckpt_every) == 0):
                ckpt_path = os.path.join(checkpoint_dir, f"step_{step + 1}.pt")
                torch.save(
                    {
                        "step": step + 1,
                        "epoch": epoch,
                        "wall_clock_sec": time.time() - t0,
                        "model_student": student.state_dict(),
                        "model_teacher": teacher.state_dict(),
                        "optimizer": optim.state_dict(),
                        "cfg": dict(cfg.__dict__),
                    },
                    ckpt_path,
                )

            if step % 100 == 0:
                print(
                    f"[{step:06d}] loss={loss.item():.4f}  "
                    f"Ldino={L_dino.item():.4f}  Lmim={L_mim.item():.4f}  "
                    f"tT={dino_params.t_teacher:.3f}  mask={mratio:.2f}  lr={cur_lr:.2e}"
                )

            step += 1
            if step >= total_steps:
                break
        epoch += 1

    # ---- ALWAYS save LAST checkpoint ----
    last_obj = {
        "step": step,
        "epoch": epoch - 1,
        "wall_clock_sec": time.time() - t0,
        "model_student": student.state_dict(),
        "model_teacher": teacher.state_dict(),
        "optimizer": optim.state_dict(),
        "cfg": dict(cfg.__dict__),
    }
    torch.save(last_obj, cfg.save_path)
    print(f"Saved checkpoint to {cfg.save_path}")

import os, numpy as np, torch, pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
from contextlib import nullcontext
from collections import defaultdict
from tqdm import tqdm

def _center_crop(patch: torch.Tensor, out_h: int, out_w: int):
    H, W = patch.shape[-2:]
    i = max(0, (H - out_h) // 2); j = max(0, (W - out_w) // 2)
    return patch[:, i:i+out_h, j:j+out_w]

def _pad_to(patch: torch.Tensor, out_h: int, out_w: int):
    H, W = patch.shape[-2:]
    pad_h, pad_w = max(0, out_h - H), max(0, out_w - W)
    # (left, right, top, bottom)
    return F.pad(patch, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))

def _fix_size(patch: torch.Tensor, out_h: int, out_w: int):
    H, W = patch.shape[-2:]
    if H > out_h or W > out_w:
        patch = _center_crop(patch, min(H, out_h), min(W, out_w))
    if patch.shape[-2] < out_h or patch.shape[-1] < out_w:
        patch = _pad_to(patch, out_h, out_w)
    return patch.contiguous()

def _pick_anchor_then_top(patch: torch.Tensor, K: int):
    C = patch.shape[0]
    var = patch.reshape(C, -1).var(dim=1)   # <-- reshape
    anchor = int(var.argmax())
    order = torch.argsort(var, descending=True).tolist()
    idx = [anchor] + [i for i in order if i != anchor][:max(0, K-1)]
    return idx

def _select_or_pad_channels(patch: torch.Tensor, mz: torch.Tensor, K: int, pad_mode="repeat"):
    C, H, W = patch.shape
    if C >= K:
        idx = _pick_anchor_then_top(patch, K)
        return patch[idx], mz[idx]

    pad_n = K - C
    if pad_mode == "repeat":
        var = patch.reshape(C, -1).var(dim=1)   # <-- reshape
        anchor = int(var.argmax())
        pad_patch = patch[anchor:anchor+1].expand(pad_n, H, W).clone()
        pad_mz    = mz[anchor:anchor+1].expand(pad_n).clone()
    else:
        pad_patch = patch.new_zeros((pad_n, H, W))
        pad_mz    = mz.new_zeros((pad_n,))
    return torch.cat([patch, pad_patch], 0), torch.cat([mz, pad_mz], 0)

def get_all_paths(ds):
    if hasattr(ds, "paths"):
        return list(ds.paths)
    return [ds[i] if isinstance(ds[i], str) else ds[i]["path"] for i in range(len(ds))]

class NPZDataset(Dataset):
    """
    Loads .npz with mmap, scales to [0,1] if needed, fixes HxW, and unifies K via pick/pad.
    Returns float32 tensors ready for GPU.
    """
    def __init__(self, paths, target_h, target_w, k_target, scale_u16=True, pad_mode="repeat", sort_by_mz=False):
        self.paths      = list(paths)
        self.target_h   = target_h
        self.target_w   = target_w
        self.k_target   = k_target
        self.scale_u16  = scale_u16
        self.pad_mode   = pad_mode
        self.sort_by_mz = sort_by_mz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        with np.load(path, mmap_mode="r") as data:
            patch = torch.from_numpy(data["patch"]).float()  # (C,H,W), often uint16
            mz    = torch.from_numpy(data["mz"]).float()     # (C,)

        # scale to [0,1] if values look like uint16
        if self.scale_u16 and patch.max().item() > 1.0:
            patch = patch / 65535.0

        # (optional) sort channels by m/z – skip if you don’t need deterministic order
        if self.sort_by_mz:
            idx = torch.argsort(mz)
            patch, mz = patch[idx], mz[idx]

        # fix spatial size once here
        patch = _fix_size(patch, self.target_h, self.target_w)

        # unify K
        patch, mz = _select_or_pad_channels(patch, mz, self.k_target, pad_mode=self.pad_mode)

        return {"patch": patch, "mz": mz, "path": path}

def collate_simple(batch):
    return {
        "patch": torch.stack([b["patch"] for b in batch], 0),  # (B,K,H,W)
        "mz":    torch.stack([b["mz"]    for b in batch], 0),  # (B,K)
        "path":  [b["path"] for b in batch]
    }