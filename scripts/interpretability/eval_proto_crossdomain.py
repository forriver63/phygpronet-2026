"""Exp C: Cross-Domain Prototype Consistency.

Given fold0 ckpt, run forward on:
  - Sleep-EDF test (fold0 partition)
  - SHHS (all)
  - MASS (all) — skipped if data dir empty
  - ISRUC (all)

For each domain, accumulate per-prototype stats:
  - assignment count
  - per-class assignment count (5 classes)
  - per-prototype class fraction p(c|k, domain)

Cross-domain metrics:
  - Correlation of per-prototype assignment patterns across domains
  - Top-K agreement: among top-K N2-specialized prototypes on Sleep-EDF,
    what fraction are also top-K N2-specialized on SHHS?
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
sys.path.insert(0, "/root/autodl-tmp/code/PhyGProNet-Repro")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.data_sleepedf_packed import open_packed, SleepEDFPackedDataset
from phygpronet.data.mass_npz import MASSNpzDataset
from phygpronet.data.shhs_npz import SHHSNpzDataset
from phygpronet.data.isruc_npz import ISRUCNpzDataset


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
def collect_proto_stats(model, loader, device, K, expects_mech=False):
    """Iterate loader; return (sum_A, sum_Ac, n_total, class_total) as numpy."""
    sum_A  = torch.zeros(K, device=device)
    sum_Ac = torch.zeros(K, 5, device=device)
    class_total = torch.zeros(5, device=device)
    n_total = 0
    for batch in loader:
        if expects_mech:
            x30, y, _, _ = batch
        else:
            x30, y = batch
        x30 = x30.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        if x30.dim() == 2:
            x30 = x30.unsqueeze(1)  # (B, 3000) -> (B,1,3000)
        elif x30.dim() == 3 and x30.shape[1] != 1:
            # if (B, T, 1) -> (B,1,T)
            if x30.shape[2] == 1:
                x30 = x30.transpose(1, 2)
        _, _, _, _, extras = model(x30)
        A = extras["assign1"]
        sum_A += A.sum(dim=0)
        y_oh = F.one_hot(y, num_classes=5).float()
        sum_Ac += A.t() @ y_oh
        class_total += y_oh.sum(dim=0)
        n_total += y.numel()
    return (sum_A.cpu().numpy(), sum_Ac.cpu().numpy(),
            int(n_total), class_total.cpu().numpy())


def specialization_score(class_counts, class_prior, eps=1e-8):
    """Return (K, 5) specialization: p(c|k) / p(c)."""
    p_c_given_k = class_counts / class_counts.sum(axis=1, keepdims=True).clip(min=eps)
    return p_c_given_k / class_prior.clip(min=eps)


def top_k_agreement(sl_edf_spec, other_spec, c, K):
    """Fraction of top-K prototypes for class c that overlap between domains."""
    top_edf = set(np.argsort(-sl_edf_spec[:, c])[:K].tolist())
    top_oth = set(np.argsort(-other_spec[:, c])[:K].tolist())
    return len(top_edf & top_oth) / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--fold_id", type=int, default=0)
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--mass_dir", default="/root/autodl-tmp/data/mass_npz")
    ap.add_argument("--isruc_dir", default="/root/autodl-tmp/data/isruc_npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    print(f"K={K}")

    results = {}

    # Sleep-EDF fold0 test
    print("[domain] Sleep-EDF test")
    packed = open_packed(args.packed_dir, readonly=True)
    idx = fold_indices_from_json(packed, args.split_json, args.fold_id, "test")
    ds = SleepEDFPackedDataset(packed, idx, want_mech=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    sumA, sumAc, n, ct = collect_proto_stats(model, dl, device, K, expects_mech=True)
    results["sleepedf"] = {"sum_A": sumA, "sum_Ac": sumAc, "n": n, "class_total": ct}

    # SHHS
    if os.path.isdir(args.shhs_dir) and os.listdir(args.shhs_dir):
        print("[domain] SHHS")
        ds = SHHSNpzDataset(args.shhs_dir)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        sumA, sumAc, n, ct = collect_proto_stats(model, dl, device, K, expects_mech=False)
        results["shhs"] = {"sum_A": sumA, "sum_Ac": sumAc, "n": n, "class_total": ct}

    # MASS
    if os.path.isdir(args.mass_dir) and os.listdir(args.mass_dir):
        print("[domain] MASS")
        ds = MASSNpzDataset(args.mass_dir)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        sumA, sumAc, n, ct = collect_proto_stats(model, dl, device, K, expects_mech=False)
        results["mass"] = {"sum_A": sumA, "sum_Ac": sumAc, "n": n, "class_total": ct}

    # ISRUC
    if os.path.isdir(args.isruc_dir) and os.listdir(args.isruc_dir):
        print("[domain] ISRUC")
        ds = ISRUCNpzDataset(args.isruc_dir)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        sumA, sumAc, n, ct = collect_proto_stats(model, dl, device, K, expects_mech=False)
        results["isruc"] = {"sum_A": sumA, "sum_Ac": sumAc, "n": n, "class_total": ct}

    # Compute spec scores per domain
    spec_per_domain = {}
    for dn, r in results.items():
        prior = r["class_total"] / r["n"]
        spec_per_domain[dn] = specialization_score(r["sum_Ac"], prior)

    # Cross-domain consistency
    cls_names = ["W","N1","N2","N3","REM"]
    report = {"K": int(K), "domains": {dn: int(r["n"]) for dn, r in results.items()}}

    if "sleepedf" in spec_per_domain:
        edf = spec_per_domain["sleepedf"]
        # Top-K agreement for each class, across each non-EDF domain
        topK_values = [5, 10, 20]
        report["top_k_agreement"] = {}
        for k_val in topK_values:
            report["top_k_agreement"][f"top{k_val}"] = {}
            for dn in spec_per_domain:
                if dn == "sleepedf": continue
                agree = {cls_names[c]: top_k_agreement(edf, spec_per_domain[dn], c, k_val)
                         for c in range(5)}
                report["top_k_agreement"][f"top{k_val}"][f"edf_vs_{dn}"] = agree

        # Per-prototype assignment count correlation
        report["assignment_correlation"] = {}
        for dn in results:
            if dn == "sleepedf": continue
            x = results["sleepedf"]["sum_A"] / results["sleepedf"]["sum_A"].sum()
            y = results[dn]["sum_A"] / results[dn]["sum_A"].sum()
            xn = (x - x.mean()) / (x.std()+1e-12)
            yn = (y - y.mean()) / (y.std()+1e-12)
            report["assignment_correlation"][f"edf_vs_{dn}"] = float((xn*yn).mean())

        # Per-class specialization correlation (across domains)
        report["spec_score_correlation"] = {}
        for dn in spec_per_domain:
            if dn == "sleepedf": continue
            corrs = {}
            for c in range(5):
                x = edf[:, c]
                y = spec_per_domain[dn][:, c]
                xn = (x - x.mean()) / (x.std()+1e-12)
                yn = (y - y.mean()) / (y.std()+1e-12)
                corrs[cls_names[c]] = float((xn*yn).mean())
            report["spec_score_correlation"][f"edf_vs_{dn}"] = corrs

    # Save
    np.savez(os.path.join(args.out_dir, "proto_crossdomain.npz"),
             **{f"{dn}_sumA": r["sum_A"] for dn, r in results.items()},
             **{f"{dn}_sumAc": r["sum_Ac"] for dn, r in results.items()},
             **{f"{dn}_class_total": r["class_total"] for dn, r in results.items()},
             **{f"{dn}_spec": spec_per_domain[dn] for dn in spec_per_domain})
    with open(os.path.join(args.out_dir, "report_crossdomain.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # Visualization: alignment heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if "sleepedf" in spec_per_domain:
            other_domains = [d for d in spec_per_domain if d != "sleepedf"]
            if other_domains:
                # heatmap of top-10 agreement
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                rows = other_domains
                data = np.zeros((len(rows), 5))
                for i, d in enumerate(rows):
                    for c in range(5):
                        data[i, c] = top_k_agreement(spec_per_domain["sleepedf"], spec_per_domain[d], c, 10)
                im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
                ax.set_xticks(range(5)); ax.set_xticklabels(cls_names)
                ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"SleepEDF vs {d.upper()}" for d in rows])
                for i in range(len(rows)):
                    for j in range(5):
                        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", color="black")
                ax.set_title("Top-10 prototype agreement per class (cross-domain)")
                fig.colorbar(im, ax=ax, label="overlap fraction")
                fig.savefig(os.path.join(args.out_dir, "topk_agreement.png"), dpi=150, bbox_inches="tight")
                print("[plot]", os.path.join(args.out_dir, "topk_agreement.png"))
    except Exception as e:
        print(f"[plot] skipped: {e}")


if __name__ == "__main__":
    main()
