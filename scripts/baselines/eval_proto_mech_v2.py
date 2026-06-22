"""Prototype-Physiology Alignment (Exp A — improved).

For each of K prototypes:
- mean spindle intensity over samples assigned to it
- mean slowwave intensity
- dominant class (argmax frequency of W/N1/N2/N3/REM)
- per-class assignment counts
- proto_logits[:,j] for j in (N3, N2, Other) raw

Output: JSON + numpy arrays + scatter plot.
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


def fold_indices_from_json(packed, split_json_path, fold_id, partition):
    with open(split_json_path) as f:
        split = json.load(f)
    fold = split["folds"][str(fold_id)]
    subjs = fold[partition]
    pkg_subjects = list(packed.subjects)
    prefix_to_sids = {}
    for sid_idx, full in enumerate(pkg_subjects):
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
    print(f"K={K} prototypes, mech_w={a.get('mech_w')}, use_pg={a.get('use_pg', True)}")

    sum_A  = torch.zeros(K, device=device)
    sum_sp = torch.zeros(K, device=device)
    sum_sw = torch.zeros(K, device=device)
    # per-class assignment count: (K, 5)
    sum_Ac = torch.zeros(K, 5, device=device)

    for x30, y, sp, sw in dl:
        x30 = x30.to(device, non_blocking=True)
        y   = y.to(device, non_blocking=True)
        sp  = sp.to(device, non_blocking=True)
        sw  = sw.to(device, non_blocking=True)
        _, _, _, _, extras = model(x30)
        A = extras["assign1"]  # (B,K)
        sum_A  += A.sum(dim=0)
        sum_sp += A.t().mv(sp)
        sum_sw += A.t().mv(sw)
        # accumulate per-class
        y_oh = F.one_hot(y, num_classes=5).float()  # (B,5)
        sum_Ac += A.t() @ y_oh   # (K,5)

    eps = 1e-8
    proto_sp = (sum_sp / sum_A.clamp_min(eps)).cpu().numpy()
    proto_sw = (sum_sw / sum_A.clamp_min(eps)).cpu().numpy()
    proto_assign = sum_A.cpu().numpy()
    proto_class_counts = sum_Ac.cpu().numpy()  # (K,5)
    proto_class_frac = proto_class_counts / proto_class_counts.sum(axis=1, keepdims=True).clip(min=eps)
    dominant_class = proto_class_counts.argmax(axis=1)

    proto_logits = model.l1.proto_logits.detach().cpu().numpy()  # (K, 3) -> [N3, N2, Other]

    # correlations
    def corr(x, y):
        x = (x - x.mean()) / (x.std()+1e-12)
        y = (y - y.mean()) / (y.std()+1e-12)
        return float((x*y).mean())
    corr_sp_logits = [corr(proto_sp, proto_logits[:,j]) for j in range(3)]
    corr_sw_logits = [corr(proto_sw, proto_logits[:,j]) for j in range(3)]

    np.savez(os.path.join(args.out_dir, "proto_stats.npz"),
             sp=proto_sp, sw=proto_sw, assign=proto_assign,
             class_counts=proto_class_counts, class_frac=proto_class_frac,
             dominant_class=dominant_class, proto_logits=proto_logits)

    report = {
        "K": int(K),
        "n_eval_samples": int(len(idx)),
        "proto_sp_range": [float(proto_sp.min()), float(proto_sp.max())],
        "proto_sw_range": [float(proto_sw.min()), float(proto_sw.max())],
        "corr_with_spindle_logits": corr_sp_logits,
        "corr_with_slowwave_logits": corr_sw_logits,
        "dominant_class_distribution": np.bincount(dominant_class, minlength=5).tolist(),
        "top10_by_assignment": [int(i) for i in proto_assign.argsort()[::-1][:10]],
        "top10_n2_specialized": [int(i) for i in proto_class_frac[:,2].argsort()[::-1][:10]],
        "top10_n3_specialized": [int(i) for i in proto_class_frac[:,3].argsort()[::-1][:10]],
        "top10_n1_specialized": [int(i) for i in proto_class_frac[:,1].argsort()[::-1][:10]],
        "top10_spindle_loaded": [int(i) for i in proto_sp.argsort()[::-1][:10]],
        "top10_slowwave_loaded": [int(i) for i in proto_sw.argsort()[::-1][:10]],
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\n[saved]", args.out_dir)

    # Make scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        cmap = ["#666","#1f77b4","#2ca02c","#d62728","#9467bd"]
        cls_names = ["W","N1","N2","N3","REM"]
        for c in range(5):
            mask = (dominant_class == c)
            ax.scatter(proto_sp[mask], proto_sw[mask],
                       s=20+150*(proto_assign[mask]/proto_assign.max()),
                       c=cmap[c], alpha=0.7, label=cls_names[c])
        ax.set_xlabel("Mean Spindle Intensity (per prototype)")
        ax.set_ylabel("Mean Slowwave Intensity (per prototype)")
        ax.set_title(f"Prototype-Physiology Map (fold{args.fold_id} {args.partition}, K={K})")
        ax.legend(title="Dominant class")
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(args.out_dir, "proto_scatter.png"), dpi=150, bbox_inches="tight")
        print("[plot]", os.path.join(args.out_dir, "proto_scatter.png"))
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
