"""Phyg training v2 — extends 6_train_inception_tree_protoL1_mech.py with:

  Exp 2: --mech_warmup_epoch K  (0 = off; >0 ramps mech_w linearly after K epochs)
  Exp 3: --mech_high_conf_pct P (P<1.0 → use only top |sp_z| or |sw_z| samples)
  Exp 4: --slowwave_w, --spindle_w  (asymmetric; if 0/0 fallback to mech_w for both)
  Exp 6: --pg_topk K            (default 12; pass-through to ProtoHeadL1)

Use: same CLI as 6_train_inception_tree_protoL1_mech.py plus these flags.
Output ckpt path: <out_dir>/best_model.pt (cleaner than original's runs/ relative).
"""
from __future__ import annotations
import os, sys, json, argparse, random, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1, ProtoHeadL1, ProtoGraphRefiner  # type: ignore
from src.data_sleepedf_packed import open_packed, SleepEDFPackedDataset, subject_split_indices  # type: ignore


# ===================== model patches =====================

def patch_pg_topk(model: InceptionTreeProtoL1, topk: int):
    """Rebuild ProtoGraphRefiner with new topk, copy MLP weights."""
    if topk == 12:
        return
    old = model.l1.pg
    new = ProtoGraphRefiner(
        D=old.mlp[0].in_features // 1 if hasattr(old, "mlp") else model.l1.proto.shape[1],
        C=3, Fdim=0, hidden=64, topk=topk, tau_edge=0.2,
    )
    # Try copy weights; if dims mismatch, leave fresh
    try:
        new.load_state_dict(old.state_dict())
    except Exception:
        pass
    model.l1.pg = new
    print(f"[patch] PG-R topk={topk} (rebuilt)")


# ===================== mech reg v2 (returns L_sp, L_sw separately, with optional mask) =====================

def mech_regularizer_v2(
    model: InceptionTreeProtoL1,
    extras: dict,
    sp_z: torch.Tensor,
    sw_z: torch.Tensor,
    margin: float,
    high_conf_pct: float = 1.0,
    high_conf_pct_sp: float = -1.0,
    high_conf_pct_sw: float = -1.0,
):
    """Returns (L_sp, L_sw) separately.
    high_conf_pct: applied to both sp/sw symmetrically (legacy).
    high_conf_pct_sp / high_conf_pct_sw (if >0): per-channel override.
    """
    eps = 1e-8
    assign1 = extras.get("assign1", None)
    if assign1 is None:
        zero = torch.zeros((), device=sp_z.device)
        return zero, zero

    A = assign1                                  # (B,K)
    B = A.shape[0]

    hcp_sp = high_conf_pct_sp if high_conf_pct_sp > 0 else high_conf_pct
    hcp_sw = high_conf_pct_sw if high_conf_pct_sw > 0 else high_conf_pct
    if hcp_sp < 1.0 and B > 1:
        k = max(1, int(round(B * hcp_sp)))
        sp_thresh = torch.topk(sp_z.abs(), k).values.min()
        sp_mask = (sp_z.abs() >= sp_thresh).float().unsqueeze(1)
        A_sp = A * sp_mask
    else:
        A_sp = A
    if hcp_sw < 1.0 and B > 1:
        k = max(1, int(round(B * hcp_sw)))
        sw_thresh = torch.topk(sw_z.abs(), k).values.min()
        sw_mask = (sw_z.abs() >= sw_thresh).float().unsqueeze(1)
        A_sw = A * sw_mask
    else:
        A_sw = A

    denom_sp = A_sp.sum(dim=0).clamp_min(eps)
    denom_sw = A_sw.sum(dim=0).clamp_min(eps)
    proto_sp = (A_sp.t() @ sp_z) / denom_sp
    proto_sw = (A_sw.t() @ sw_z) / denom_sw

    def to_weight(v):
        v = (v - v.mean()) / v.std().clamp_min(1e-6)
        return torch.sigmoid(v)

    w_sp = to_weight(proto_sp)
    w_sw = to_weight(proto_sw)

    logits = extras.get("proto_logits_l1", None)
    if logits is None:
        logits = model.l1.proto_logits

    # spindle -> N2 (index 1)
    other_n2 = torch.stack([logits[:, 0], logits[:, 2]], dim=1).max(dim=1).values
    delta_n2 = logits[:, 1] - other_n2
    L_sp = (w_sp * F.softplus(margin - delta_n2)).mean()

    # slowwave -> N3 (index 0)
    other_n3 = torch.stack([logits[:, 1], logits[:, 2]], dim=1).max(dim=1).values
    delta_n3 = logits[:, 0] - other_n3
    L_sw = (w_sw * F.softplus(margin - delta_n3)).mean()

    return L_sp, L_sw


# ===================== misc utils  (copied from 6_train) =====================

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def level_targets(y):
    y1 = torch.where(y == 3, torch.zeros_like(y),
        torch.where(y == 2, torch.ones_like(y), 2*torch.ones_like(y)))
    y2 = torch.where(y == 1, torch.ones_like(y), torch.zeros_like(y))  # arbitrary, masked
    y3 = torch.where(y == 0, torch.zeros_like(y), torch.ones_like(y))
    return y1, y2, y3


def proto_regularizers(model, extras, rep_margin, device):
    A = extras.get("assign1")
    if A is None:
        return torch.zeros((), device=device), torch.zeros((), device=device)
    usage = A.sum(dim=0)
    usage = usage / usage.sum().clamp_min(1e-8)
    L_use = -(usage * (usage.clamp_min(1e-8)).log()).sum()
    L_use = -L_use  # encourage entropy → negative
    # rep: prototype separation margin
    P = model.l1.proto
    p2 = (P ** 2).sum(dim=1, keepdim=True)
    d2 = p2 + p2.t() - 2.0 * P @ P.t()
    d2.fill_diagonal_(1e9)
    L_rep = F.relu(rep_margin - d2.clamp_min(0).sqrt()).mean()
    return L_use, L_rep


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
    model.eval()
    ys, ps = [], []
    for x30, y, sp, sw in loader:
        x30 = x30.to(device, non_blocking=True)
        _, _, _, p5, _ = model(x30)
        ys.append(y.cpu().numpy()); ps.append(p5.argmax(-1).cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "bal_acc":  float(balanced_accuracy_score(y_true, y_pred)),
        "kappa":    float(cohen_kappa_score(y_true, y_pred)),
        "per_f1":   f1_score(y_true, y_pred, average=None, labels=list(range(5))).tolist(),
    }


# ===================== main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--out_ch", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--K1", type=int, default=96)
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=3, help="train warmup (uses aux losses only for first N epochs)")
    ap.add_argument("--use_w", type=float, default=0.02)
    ap.add_argument("--rep_w", type=float, default=0.01)
    ap.add_argument("--rep_margin", type=float, default=0.2)
    ap.add_argument("--mech_w", type=float, default=0.2, help="(legacy) used if --slowwave_w/--spindle_w both 0")
    ap.add_argument("--mech_margin", type=float, default=1.0)
    ap.add_argument("--use_pg", action="store_true", default=True)
    # === new ===
    ap.add_argument("--mech_warmup_epoch", type=int, default=0, help="Exp 2: keep mech_w=0 for first N epochs, then ramp")
    ap.add_argument("--mech_high_conf_pct", type=float, default=1.0, help="Exp 3: top-P% samples by |sp_z|/|sw_z|")
    ap.add_argument("--mech_high_conf_pct_sp", type=float, default=-1.0, help="per-channel override for spindle high-conf (Exp A fusion)")
    ap.add_argument("--mech_high_conf_pct_sw", type=float, default=-1.0, help="per-channel override for slowwave high-conf")
    ap.add_argument("--slowwave_w", type=float, default=0.0, help="Exp 4: separate slowwave/N3 weight (overrides mech_w if >0)")
    ap.add_argument("--spindle_w", type=float, default=0.0, help="Exp 4: separate spindle/N2 weight (overrides mech_w if >0)")
    ap.add_argument("--pg_topk", type=int, default=12, help="Exp 6: PG-R adjacency topk")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    # decide asymmetric weights
    if args.slowwave_w == 0.0 and args.spindle_w == 0.0:
        sw_w0, sp_w0 = args.mech_w, args.mech_w
    else:
        sw_w0, sp_w0 = args.slowwave_w, args.spindle_w

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    packed = open_packed(args.packed_dir, readonly=True)
    tr_idx, va_idx, te_idx = subject_split_indices(packed, seed=args.seed)
    print(f"[split seed={args.seed}] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    tr_ds = SleepEDFPackedDataset(packed, tr_idx, want_mech=True)
    va_ds = SleepEDFPackedDataset(packed, va_idx, want_mech=True)
    te_ds = SleepEDFPackedDataset(packed, te_idx, want_mech=True)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = InceptionTreeProtoL1(
        z_dim=args.z_dim, blocks=args.blocks, out_ch=args.out_ch,
        head_hidden=args.head_hidden, K1=args.K1, tau=args.tau,
    )
    model.l1.use_pg = bool(args.use_pg)
    patch_pg_topk(model, args.pg_topk)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1.0; best_state = None; history = []
    for ep in range(1, args.epochs + 1):
        # mech curriculum
        if args.mech_warmup_epoch > 0:
            denom = max(args.epochs - args.mech_warmup_epoch, 1)
            ramp = max(0.0, (ep - args.mech_warmup_epoch) / denom)
        else:
            ramp = 1.0
        cur_sw_w = sw_w0 * ramp
        cur_sp_w = sp_w0 * ramp

        model.train()
        losses = []
        for x30, y, sp, sw in tqdm(tr_ld, desc=f"Ep{ep}/{args.epochs}", leave=False):
            x30 = x30.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)
            z, (l1, l2, l3), (p1, p2, p3), p5, extras = model(x30)
            logp5 = (p5.clamp_min(1e-8)).log()
            loss_main = F.nll_loss(logp5, y)
            y1, y2, y3 = level_targets(y)
            L1 = F.cross_entropy(l1, y1)
            m2 = (y != 2) & (y != 3)
            L2 = F.cross_entropy(l2[m2], y2[m2]) if m2.any() else torch.zeros((), device=device)
            m3 = (y == 0) | (y == 1)
            L3 = F.cross_entropy(l3[m3], y3[m3]) if m3.any() else torch.zeros((), device=device)
            loss_aux = (L1 + L2 + L3) / 3.0
            L_use, L_rep = proto_regularizers(model, extras, rep_margin=args.rep_margin, device=device)
            loss_reg = args.use_w * L_use + args.rep_w * L_rep
            L_sp, L_sw = mech_regularizer_v2(model, extras, sp, sw, margin=args.mech_margin,
                                             high_conf_pct=args.mech_high_conf_pct,
                                             high_conf_pct_sp=args.mech_high_conf_pct_sp,
                                             high_conf_pct_sw=args.mech_high_conf_pct_sw)
            loss_mech = cur_sw_w * L_sw + cur_sp_w * L_sp
            if ep <= args.warmup:
                loss = (L1 + L2 + L3) + loss_reg + loss_mech
            else:
                loss = loss_main + args.aux * loss_aux + loss_reg + loss_mech
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        val_m = evaluate(model, va_ld, device)
        print(f"[ep {ep}] cur_sw={cur_sw_w:.3f} cur_sp={cur_sp_w:.3f} loss={np.mean(losses):.4f} val_F1={val_m['macro_f1']:.4f}")
        history.append({"epoch": ep, "cur_sw_w": cur_sw_w, "cur_sp_w": cur_sp_w, "val": val_m})
        if val_m["macro_f1"] > best:
            best = val_m["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_m = evaluate(model, te_ld, device)
    out = {
        "best_val_f1": float(best), "test": test_m, "history": history,
        "config": vars(args),
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save({"state_dict": best_state, "args": vars(args), "test": test_m},
               os.path.join(args.out_dir, "best_model.pt"))
    print("=== TEST ===", test_m)


if __name__ == "__main__":
    main()
