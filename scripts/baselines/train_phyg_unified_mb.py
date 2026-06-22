"""PhyGProNet training with multi-band mech regularizer + per-recording z-score.

Adds two switches over train_phyg_unified.py:
  --multi_band_mech : use 5 band features (delta, theta, alpha, sigma, beta) instead of (sp, sw)
  --per_recording_norm : z-score EEG per recording using precomputed rec stats

Mech regularizer mapping when --multi_band_mech:
   delta  -> encourage N3 (hi delta = SWS)
   sigma  -> encourage N2 (spindle band)
   alpha  -> encourage W  (alpha rhythm awake)
   beta   -> encourage W  (alertness)
   theta  -> encourage REM (theta dominant) [informative; can be disabled with --theta_target=none]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")

from src.metrics import metrics
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1, level_targets
from src.data_sleepedf_packed import (
    open_packed as load_packed_arrays,
    SleepEDFPackedDataset,
    BAND_ORDER,
)


def fold_indices(packed, split_json_path, fold_id):
    with open(split_json_path) as f:
        split = json.load(f)
    fold = split["folds"][str(fold_id)]
    pkg_subjects = list(packed.subjects)
    prefix_to_sids = {}
    for sid_idx, full in enumerate(pkg_subjects):
        prefix_to_sids.setdefault(full[:5], []).append(sid_idx)

    def collect(subj_list):
        sids, missing = [], []
        for s in subj_list:
            if s in prefix_to_sids:
                sids.extend(prefix_to_sids[s])
            else:
                missing.append(s)
        return set(sids), missing

    tr_sids, _ = collect(fold["train"])
    va_sids, _ = collect(fold["val"])
    te_sids, _ = collect(fold["test"])
    all_sids = np.array(packed.sid, dtype=np.int32)
    idx = np.arange(packed.N, dtype=np.int64)
    tr_idx = idx[np.isin(all_sids, list(tr_sids))]
    va_idx = idx[np.isin(all_sids, list(va_sids))]
    te_idx = idx[np.isin(all_sids, list(te_sids))]
    return tr_idx, va_idx, te_idx


def compute_class_weights(packed, tr_idx, mode="sqrt_clipped"):
    if mode == "none":
        return None
    ys = np.asarray(packed.y[tr_idx]).astype(np.int64)
    bincnt = np.bincount(ys, minlength=5).astype(np.float32)
    if mode == "sqrt_clipped":
        cw = np.sqrt(bincnt.sum() / (5.0 * np.maximum(bincnt, 1))).astype(np.float32)
        cw = np.clip(cw, 0.8, 2.5)
        return cw
    raise ValueError(mode)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x30 = batch[0].to(device, non_blocking=True)
            y = batch[1]
            _, _, _, p5, _ = model(x30)
            ps.append(p5.argmax(-1).cpu().numpy())
            ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    ps = np.concatenate(ps); ys = np.concatenate(ys)
    return metrics(ys, ps, n_classes=5)


def proto_regularizers(model, extras, rep_margin, device):
    use_w_term = torch.zeros((), device=device)
    rep_term = torch.zeros((), device=device)
    if "C" in extras:
        C = extras["C"]
        Cn = F.normalize(C, dim=-1)
        sim = Cn @ Cn.t()
        K = C.shape[0]
        I = torch.eye(K, device=device)
        rep_term = F.relu(sim - I * 2 - rep_margin).pow(2).mean()
    if "assign1" in extras:
        assign1 = extras["assign1"]
        denom = assign1.sum(dim=0)
        use_w_term = (-denom.clamp_min(1e-8).log().mean())
    return use_w_term, rep_term


def mech_reg_legacy(model, extras, sp_z, sw_z, margin):
    """Original 2-band: spindle->N2, slowwave->N3 (uses 3-way proto_logits l1)."""
    assign1 = extras.get("assign1", None)
    if assign1 is None:
        return torch.zeros((), device=sp_z.device)
    eps = 1e-8
    A = assign1
    denom = A.sum(dim=0).clamp_min(eps)
    proto_sp = (A.t() @ sp_z) / denom
    proto_sw = (A.t() @ sw_z) / denom

    def to_weight(v):
        v = (v - v.mean()) / (v.std().clamp_min(1e-6))
        return torch.sigmoid(v)

    w_sp = to_weight(proto_sp)
    w_sw = to_weight(proto_sw)
    logits = extras.get("proto_logits_l1", None)
    if logits is None:
        logits = model.l1.proto_logits

    other_n2 = torch.stack([logits[:, 0], logits[:, 2]], dim=1).max(dim=1).values
    delta_n2 = logits[:, 1] - other_n2
    L_sp = (w_sp * F.softplus(margin - delta_n2)).mean()
    other_n3 = torch.stack([logits[:, 1], logits[:, 2]], dim=1).max(dim=1).values
    delta_n3 = logits[:, 0] - other_n3
    L_sw = (w_sw * F.softplus(margin - delta_n3)).mean()
    return L_sp + L_sw


# l1 proto_logits is (K, 3) with columns = [N3, N2, Other]
# In our tree, l1 is a 3-way decision at the top level: {N3, N2, Other}.
# So we cannot directly target W vs REM at this level. We use the *level-1* logit "Other" as a
# proxy: alpha & beta should encourage "Other" (which contains W). For sigma->N2, delta->N3 we
# act on columns 1 and 0 directly. Theta is informative (REM-ish) — we treat it like alpha (Other
# proxy) because REM lives in the "Other" branch. This is a conservative, leakage-safe extension.
def mech_reg_multiband(model, extras, bands_z, margin,
                       use_alpha_other=True, use_beta_other=True, use_theta_other=True):
    """bands_z: (B, 5) in order delta, theta, alpha, sigma, beta (z-scored)."""
    assign1 = extras.get("assign1", None)
    if assign1 is None:
        return torch.zeros((), device=bands_z.device)
    eps = 1e-8
    A = assign1                              # (B, K)
    denom = A.sum(dim=0).clamp_min(eps)      # (K,)
    proto_b = (A.t() @ bands_z) / denom.unsqueeze(1)  # (K, 5)

    def to_w(v):
        v = (v - v.mean()) / (v.std().clamp_min(1e-6))
        return torch.sigmoid(v)

    logits = extras.get("proto_logits_l1", None)
    if logits is None:
        logits = model.l1.proto_logits         # (K, 3) [N3, N2, Other]

    # Margins for each level-1 class
    def margin_loss_for(col_target, w_proto):
        other_cols = [c for c in range(3) if c != col_target]
        other_max = torch.stack([logits[:, c] for c in other_cols], dim=1).max(dim=1).values
        delta = logits[:, col_target] - other_max
        return (w_proto * F.softplus(margin - delta)).mean()

    # bands_z order = delta, theta, alpha, sigma, beta
    w_delta = to_w(proto_b[:, 0])
    w_theta = to_w(proto_b[:, 1])
    w_alpha = to_w(proto_b[:, 2])
    w_sigma = to_w(proto_b[:, 3])
    w_beta  = to_w(proto_b[:, 4])

    # delta -> N3 (col 0)
    L = margin_loss_for(0, w_delta)
    # sigma -> N2 (col 1)
    L = L + margin_loss_for(1, w_sigma)
    # alpha -> Other (col 2)  (W lives in Other)
    if use_alpha_other:
        L = L + 0.5 * margin_loss_for(2, w_alpha)
    if use_beta_other:
        L = L + 0.5 * margin_loss_for(2, w_beta)
    if use_theta_other:
        # theta is more REM-ish but REM also lives in Other; keep small weight
        L = L + 0.25 * margin_loss_for(2, w_theta)
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--fold_id", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--class_weights", choices=["none", "sqrt_clipped"], default="sqrt_clipped")
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--out_ch", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--K1", type=int, default=192)
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--kernel_sizes", type=int, nargs="+", default=[9, 19, 39])
    ap.add_argument("--use_pg", action="store_true", default=True)
    ap.add_argument("--no_pg", action="store_true")

    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--use_w", type=float, default=0.02)
    ap.add_argument("--rep_w", type=float, default=0.01)
    ap.add_argument("--rep_margin", type=float, default=0.2)
    ap.add_argument("--mech_w", type=float, default=0.1)
    ap.add_argument("--mech_margin", type=float, default=1.0)

    # NEW
    ap.add_argument("--multi_band_mech", action="store_true",
                    help="Use 5-band PSD features for mech reg")
    ap.add_argument("--per_recording_norm", action="store_true",
                    help="Z-score EEG per recording using stored rec stats")
    ap.add_argument("--band_stats", default=None,
                    help="Path to band_stats_fold{F}.json (default: in packed_dir)")
    ap.add_argument("--rec_stats", default=None,
                    help="Path to recording_stats.json (default: in packed_dir)")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    pg_flag = bool(args.use_pg) and not args.no_pg
    device = args.device

    band_stats = None
    rec_stats = None
    if args.multi_band_mech:
        bs_path = args.band_stats or os.path.join(args.packed_dir, f"band_stats_fold{args.fold_id}.json")
        with open(bs_path) as f:
            band_stats = json.load(f)
        print(f"[band_stats] {bs_path}")
    if args.per_recording_norm:
        rs_path = args.rec_stats or os.path.join(args.packed_dir, "recording_stats.json")
        with open(rs_path) as f:
            rec_stats = json.load(f)
        print(f"[rec_stats] {rs_path}  n_sids={len(rec_stats)}")

    packed = load_packed_arrays(args.packed_dir, load_bands=args.multi_band_mech)
    tr_idx, va_idx, te_idx = fold_indices(packed, args.split_json, args.fold_id)
    yy = np.asarray(packed.y)
    print(f"[split] fold={args.fold_id}  N_train={len(tr_idx)}  N_val={len(va_idx)}  N_test={len(te_idx)}")
    for nm, ids in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        c = np.bincount(yy[ids], minlength=5).tolist()
        print(f"   {nm} class dist (W/N1/N2/N3/REM): {c}")

    ds_kw = dict(
        want_mech=not args.multi_band_mech,
        multi_band=args.multi_band_mech,
        band_stats=band_stats,
        per_recording_norm=args.per_recording_norm,
        rec_stats=rec_stats,
    )
    tr_ds = SleepEDFPackedDataset(packed, tr_idx, **ds_kw)
    va_ds = SleepEDFPackedDataset(packed, va_idx, **ds_kw)
    te_ds = SleepEDFPackedDataset(packed, te_idx, **ds_kw)
    pf = 2 if args.num_workers > 0 else None
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                       prefetch_factor=pf, pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                       prefetch_factor=pf)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                       prefetch_factor=pf)

    cw = compute_class_weights(packed, tr_idx, mode=args.class_weights)
    print(f"[class_weights={args.class_weights}] {cw}")
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device) if cw is not None else None

    model = InceptionTreeProtoL1(
        z_dim=args.z_dim, blocks=args.blocks, out_ch=args.out_ch,
        head_hidden=args.head_hidden, K1=args.K1, tau=args.tau, kernel_sizes=tuple(args.kernel_sizes)
    ).to(device)
    if hasattr(model, "l1") and hasattr(model.l1, "use_pg"):
        model.l1.use_pg = pg_flag

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1 = -1.0
    best_state = None
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(tr_ld, desc=f"Fold{args.fold_id} Ep{ep}/{args.epochs}", leave=False):
            if args.multi_band_mech:
                x30, y, bands = batch
                bands = bands.to(device, non_blocking=True)
                sp = None; sw = None
            else:
                x30, y, sp, sw = batch
                sp = sp.to(device, non_blocking=True)
                sw = sw.to(device, non_blocking=True)
            x30 = x30.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            z, (l1, l2, l3), (p1, p2, p3), p5, extras = model(x30)

            logp5 = (p5.clamp_min(1e-8)).log()
            if args.label_smoothing > 0:
                ls = args.label_smoothing
                K = 5
                with torch.no_grad():
                    target = torch.full_like(logp5, ls / (K - 1))
                    target.scatter_(1, y.unsqueeze(1), 1.0 - ls)
                if cw_t is not None:
                    w = cw_t[y].unsqueeze(1)
                    loss_main = -(w * target * logp5).sum(dim=1).mean()
                else:
                    loss_main = -(target * logp5).sum(dim=1).mean()
            else:
                loss_main = F.nll_loss(logp5, y, weight=cw_t)

            y1, y2, y3 = level_targets(y)
            L1 = F.cross_entropy(l1, y1)
            m2 = (y != 2) & (y != 3)
            L2 = F.cross_entropy(l2[m2], y2[m2]) if m2.any() else torch.zeros((), device=device)
            m3 = (y == 0) | (y == 1)
            L3 = F.cross_entropy(l3[m3], y3[m3]) if m3.any() else torch.zeros((), device=device)
            loss_aux = (L1 + L2 + L3) / 3.0

            L_use, L_rep = proto_regularizers(model, extras, args.rep_margin, device)
            loss_reg = args.use_w * L_use + args.rep_w * L_rep

            if args.multi_band_mech:
                L_mech = mech_reg_multiband(model, extras, bands, args.mech_margin)
            else:
                L_mech = mech_reg_legacy(model, extras, sp, sw, args.mech_margin)
            loss_mech = args.mech_w * L_mech

            loss = loss_main + args.aux * loss_aux + loss_reg + loss_mech

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        sched.step()
        val_m = evaluate(model, va_ld, device)
        print(f"[fold{args.fold_id} ep{ep}] loss={np.mean(losses):.4f}  "
              f"val_F1={val_m['macro_f1']:.4f}  val_BalAcc={val_m['bal_acc']:.4f}  val_Kappa={val_m['kappa']:.4f}")
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val": val_m})

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "args": vars(args), "val": val_m, "epoch": ep}, best_path)

    model.load_state_dict(best_state)
    test_m = evaluate(model, te_ld, device)
    print("\n=== TEST (best val macro-F1) ===")
    print(test_m)

    out = {
        "fold_id": args.fold_id,
        "best_val_f1": float(best_f1),
        "test": test_m,
        "config": vars(args),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
