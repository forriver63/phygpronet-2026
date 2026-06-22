"""Prototype-Physiology Alignment v3.

Improvements over v2:
- Filter inactive prototypes (assign < threshold).
- Per-class "specialization": for class c, prototype k is c-specialized
  if normalized affinity p(c|k) / p(c) > 1 (i.e., prototype prefers c
  more than class prior would predict).
- Per-class "dominant" prototype defined via specialization score, not raw argmax.
- Alignment score: among "spindle-loaded" top-N prototypes, what fraction are
  N2-specialized? Same for "slowwave-loaded" → N3.
- Scatter plot annotated with theoretical N2/N3 regions.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.data_sleepedf_packed import open_packed, SleepEDFPackedDataset


def fold_indices_from_json(packed, split_json, fold_id, partition):
    with open(split_json) as f:
        sj = json.load(f)
    fold = sj["folds"][str(fold_id)]
    subjs = fold[partition]
    prefix_to_sids = {}
    for sid_idx, full in enumerate(packed.subjects):
        prefix_to_sids.setdefault(full[:5], []).append(sid_idx)
    sids = set()
    for s in subjs:
        sids.update(prefix_to_sids.get(s, []))
    all_sids = np.array(packed.sid, dtype=np.int32)
    return np.arange(packed.N, dtype=np.int64)[np.isin(all_sids, list(sids))]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--fold_id", type=int, required=True)
    ap.add_argument("--partition", choices=["train","val","test"], default="test")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--active_thresh", type=float, default=0.001,
                    help="frac of total assignment below which prototype is 'inactive'")
    ap.add_argument("--topN", type=int, default=20, help="top-N for spindle/slowwave loaded")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    packed = open_packed(args.packed_dir, readonly=True)
    idx = fold_indices_from_json(packed, args.split_json, args.fold_id, args.partition)
    print(f"[split] fold{args.fold_id} {args.partition}: N={len(idx)}")

    ds = SleepEDFPackedDataset(packed, idx, want_mech=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    a = ckpt.get("args", {})
    model = InceptionTreeProtoL1(
        z_dim=int(a.get("z_dim",256)),
        blocks=int(a.get("blocks",6)),
        out_ch=int(a.get("out_ch",32)),
        head_hidden=int(a.get("head_hidden",256)),
        K1=int(a.get("K1",96)),
        tau=float(a.get("tau",3.5)),
    ).to(device)
    if hasattr(model, "l1") and hasattr(model.l1, "use_pg"):
        model.l1.use_pg = bool(a.get("use_pg", True))
    model.load_state_dict(sd, strict=True)
    model.eval()
    K = model.l1.proto_logits.shape[0]
    print(f"K={K} prototypes; mech_w={a.get('mech_w')}; ckpt_val_F1={ckpt.get('val',{}).get('macro_f1', 'N/A')}")

    sum_A  = torch.zeros(K, device=device)
    sum_sp = torch.zeros(K, device=device)
    sum_sw = torch.zeros(K, device=device)
    sum_Ac = torch.zeros(K, 5, device=device)
    n_total = 0
    class_total = torch.zeros(5, device=device)

    for x30, y, sp, sw in dl:
        x30 = x30.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sp = sp.to(device, non_blocking=True)
        sw = sw.to(device, non_blocking=True)
        _, _, _, _, extras = model(x30)
        A = extras["assign1"]
        sum_A  += A.sum(dim=0)
        sum_sp += A.t().mv(sp)
        sum_sw += A.t().mv(sw)
        y_oh = F.one_hot(y, num_classes=5).float()
        sum_Ac += A.t() @ y_oh
        class_total += y_oh.sum(dim=0)
        n_total += y.numel()

    eps = 1e-8
    proto_sp = (sum_sp / sum_A.clamp_min(eps)).cpu().numpy()
    proto_sw = (sum_sw / sum_A.clamp_min(eps)).cpu().numpy()
    proto_assign = sum_A.cpu().numpy()
    proto_class_counts = sum_Ac.cpu().numpy()
    class_total_np = class_total.cpu().numpy()
    class_prior = class_total_np / n_total

    # Active = used at least active_thresh fraction
    active_mask = proto_assign >= args.active_thresh * proto_assign.sum()
    n_active = int(active_mask.sum())

    # p(c|k) = counts[k,c] / sum_c counts[k,c]
    p_c_given_k = proto_class_counts / proto_class_counts.sum(axis=1, keepdims=True).clip(min=eps)
    # specialization score for class c: p(c|k) / p(c)  (higher = more specialized)
    spec_score = p_c_given_k / class_prior.clip(min=eps)
    # dominant class by specialization (only for active)
    dom_spec = spec_score.argmax(axis=1)

    proto_logits = model.l1.proto_logits.detach().cpu().numpy()

    # Top-N spindle-loaded & slowwave-loaded prototypes (active only)
    sp_rank = np.argsort(-proto_sp)
    sw_rank = np.argsort(-proto_sw)
    sp_rank_active = [int(i) for i in sp_rank if active_mask[i]][:args.topN]
    sw_rank_active = [int(i) for i in sw_rank if active_mask[i]][:args.topN]

    # Alignment score: fraction of top-N spindle-loaded that are N2-specialized
    def specialized_in(c, idx_list):
        return float(sum(1 for k in idx_list if dom_spec[k] == c) / max(len(idx_list), 1))

    align = {
        "top_spindle_is_N2": specialized_in(2, sp_rank_active),
        "top_slowwave_is_N3": specialized_in(3, sw_rank_active),
        # also check: top-N for each class & their average spindle/slowwave
    }

    # For each class, top-5 most specialized prototypes
    per_class_top5 = {}
    for c in range(5):
        idxs = [int(i) for i in np.argsort(-spec_score[:, c]) if active_mask[i]][:5]
        per_class_top5[["W","N1","N2","N3","REM"][c]] = {
            "proto_ids": idxs,
            "spec_scores": [float(spec_score[i, c]) for i in idxs],
            "mean_sp": [float(proto_sp[i]) for i in idxs],
            "mean_sw": [float(proto_sw[i]) for i in idxs],
            "class_frac": [float(p_c_given_k[i, c]) for i in idxs],
        }

    # Correlations (active only for cleaner story)
    def corr(x, y):
        x = (x - x.mean()) / (x.std()+1e-12)
        y = (y - y.mean()) / (y.std()+1e-12)
        return float((x*y).mean())

    A_idx = np.where(active_mask)[0]
    sp_a = proto_sp[A_idx]
    sw_a = proto_sw[A_idx]
    L_a = proto_logits[A_idx]
    corr_sp_log = [corr(sp_a, L_a[:, j]) for j in range(3)]
    corr_sw_log = [corr(sw_a, L_a[:, j]) for j in range(3)]

    np.savez(os.path.join(args.out_dir, "proto_stats_v3.npz"),
             sp=proto_sp, sw=proto_sw, assign=proto_assign,
             class_counts=proto_class_counts, p_c_given_k=p_c_given_k,
             spec_score=spec_score, dom_spec=dom_spec, active_mask=active_mask,
             class_prior=class_prior, proto_logits=proto_logits)

    report = {
        "K": int(K),
        "n_active": n_active,
        "n_total_samples": int(n_total),
        "class_prior": [float(p) for p in class_prior],
        "ckpt_val_macro_f1": float(ckpt.get("val",{}).get("macro_f1", -1)),
        "alignment_score": align,
        "corr_active_with_spindle": corr_sp_log,
        "corr_active_with_slowwave": corr_sw_log,
        "per_class_top5_specialized": per_class_top5,
        "active_dominant_distribution_by_spec": np.bincount(dom_spec[active_mask], minlength=5).tolist(),
    }
    with open(os.path.join(args.out_dir, "report_v3.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        cmap = ["#888","#1f77b4","#2ca02c","#d62728","#9467bd"]
        cls_names = ["W","N1","N2","N3","REM"]

        # Inactive: small gray dots
        inactive = ~active_mask
        ax.scatter(proto_sp[inactive], proto_sw[inactive], s=8, c="#ddd",
                   alpha=0.4, label=f"inactive ({inactive.sum()})")

        # Active: colored by spec-based dominant
        for c in range(5):
            mask = active_mask & (dom_spec == c)
            if mask.sum() == 0:
                continue
            ax.scatter(proto_sp[mask], proto_sw[mask],
                       s=30 + 250 * proto_assign[mask] / proto_assign.max(),
                       c=cmap[c], alpha=0.8, edgecolors="black", linewidths=0.4,
                       label=f"{cls_names[c]}-spec ({mask.sum()})")

        # Annotate theoretical regions
        ax.axhspan(0.2, 1.0, xmin=0, xmax=0.45, alpha=0.1, color="red", label="_N3 theory")
        ax.axhspan(-1.0, -0.2, xmin=0.5, xmax=1.0, alpha=0.1, color="green", label="_N2 theory")
        ax.axhline(0, c="black", lw=0.5, alpha=0.3)
        ax.axvline(0, c="black", lw=0.5, alpha=0.3)
        ax.set_xlabel("Mean Spindle Intensity (per prototype)")
        ax.set_ylabel("Mean Slowwave Intensity (per prototype)")
        ax.set_title(f"Prototype-Physiology v3 (fold{args.fold_id} {args.partition}, K={K}, active={n_active})")
        ax.legend(title="Spec-based dominant", loc="upper right")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(args.out_dir, "proto_scatter_v3.png"), dpi=150, bbox_inches="tight")
        print("[plot]", os.path.join(args.out_dir, "proto_scatter_v3.png"))
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
