"""PhyGProNet training under UNIFIED 5-fold subject-CV protocol.

Matches TinySleepNet v2 / "strong baseline" protocol:
- subject-level 5-fold split loaded from JSON (5-char prefix grouping)
- AdamW + CosineAnnealingLR
- 50 epochs (configurable)
- monitor val macro-F1, save best ckpt
- class weights (sqrt-scaled clipped) computed from TRAIN fold only
- label smoothing CE on main logit; PhyG aux/reg/mech losses retained
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")

from src.metrics import metrics
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1, level_targets
from src.data_sleepedf_packed import (
    open_packed as load_packed_arrays,
    SleepEDFPackedDataset,
)


def fold_indices(packed, split_json_path, fold_id):
    """Map 5-char subject prefixes (split JSON) -> sample indices in packed memmap."""
    with open(split_json_path) as f:
        split = json.load(f)
    fold = split["folds"][str(fold_id)]
    pkg_subjects = list(packed.subjects)  # full names like "SC4001"
    # Map: 5-char prefix -> list of full names + sid indices
    prefix_to_sids = {}
    for sid_idx, full in enumerate(pkg_subjects):
        prefix_to_sids.setdefault(full[:5], []).append(sid_idx)

    def collect(subj_list):
        sids = []
        missing = []
        for s in subj_list:
            if s in prefix_to_sids:
                sids.extend(prefix_to_sids[s])
            else:
                missing.append(s)
        return set(sids), missing

    tr_sids, tr_miss = collect(fold["train"])
    va_sids, va_miss = collect(fold["val"])
    te_sids, te_miss = collect(fold["test"])
    if tr_miss or va_miss or te_miss:
        print(f"[WARN] missing subjects: train={tr_miss}, val={va_miss}, test={te_miss}")

    all_sids = np.array(packed.sid, dtype=np.int32)
    idx = np.arange(packed.N, dtype=np.int64)
    tr_idx = idx[np.isin(all_sids, list(tr_sids))]
    va_idx = idx[np.isin(all_sids, list(va_sids))]
    te_idx = idx[np.isin(all_sids, list(te_sids))]

    # sanity: no overlap
    assert len(set(tr_idx) & set(va_idx)) == 0
    assert len(set(tr_idx) & set(te_idx)) == 0
    assert len(set(va_idx) & set(te_idx)) == 0
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
    """Reuse from original PhyG trainer."""
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


def mech_regularizer_level1(model, extras, sp_z, sw_z, margin):
    """Reuse: spindle->N2, slowwave->N3."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--fold_id", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--class_weights", choices=["none", "sqrt_clipped"], default="sqrt_clipped")
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=4)

    # model
    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--out_ch", type=int, default=32)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--K1", type=int, default=96)
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--kernel_sizes", type=int, nargs="+", default=[9, 19, 39])
    ap.add_argument("--use_pg", action="store_true", default=True)
    ap.add_argument("--no_pg", action="store_true")

    # losses
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--use_w", type=float, default=0.02)
    ap.add_argument("--rep_w", type=float, default=0.01)
    ap.add_argument("--rep_margin", type=float, default=0.2)
    ap.add_argument("--mech_w", type=float, default=0.1)  # default to 0.1 in unified
    ap.add_argument("--mech_margin", type=float, default=1.0)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    pg_flag = bool(args.use_pg) and not args.no_pg
    device = args.device

    # Data
    packed = load_packed_arrays(args.packed_dir)
    tr_idx, va_idx, te_idx = fold_indices(packed, args.split_json, args.fold_id)

    # Class distribution audit
    yy = np.asarray(packed.y)
    print(f"[split] fold={args.fold_id}  N_train={len(tr_idx)}  N_val={len(va_idx)}  N_test={len(te_idx)}")
    for nm, ids in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        c = np.bincount(yy[ids], minlength=5).tolist()
        print(f"   {nm} class dist (W/N1/N2/N3/REM): {c}")

    tr_ds = SleepEDFPackedDataset(packed, tr_idx, want_mech=True)
    va_ds = SleepEDFPackedDataset(packed, va_idx, want_mech=True)
    te_ds = SleepEDFPackedDataset(packed, te_idx, want_mech=True)
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

    # Class weights from train only
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
        for x30, y, sp, sw in tqdm(tr_ld, desc=f"Fold{args.fold_id} Ep{ep}/{args.epochs}", leave=False):
            x30 = x30.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)

            z, (l1, l2, l3), (p1, p2, p3), p5, extras = model(x30)
            # main: cross-entropy with class weights + label smoothing on log(p5)
            # p5 is probability (softmax already applied inside model); convert to logit space for CE
            # Use NLL on log(p5) with class weights — label smoothing applied manually
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

            # aux
            y1, y2, y3 = level_targets(y)
            L1 = F.cross_entropy(l1, y1)
            m2 = (y != 2) & (y != 3)
            L2 = F.cross_entropy(l2[m2], y2[m2]) if m2.any() else torch.zeros((), device=device)
            m3 = (y == 0) | (y == 1)
            L3 = F.cross_entropy(l3[m3], y3[m3]) if m3.any() else torch.zeros((), device=device)
            loss_aux = (L1 + L2 + L3) / 3.0

            L_use, L_rep = proto_regularizers(model, extras, args.rep_margin, device)
            loss_reg = args.use_w * L_use + args.rep_w * L_rep
            L_mech = mech_regularizer_level1(model, extras, sp, sw, args.mech_margin)
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

    # Final test with best ckpt
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
