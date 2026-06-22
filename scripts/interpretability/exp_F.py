"""Exp F: Per-class top prototype consistency across 5 folds.

For each fold ckpt, compute per-class top-K prototypes (by spec_score)
on Sleep-EDF test partition. Then measure cross-fold overlap.
"""
import sys, os, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.data_sleepedf_packed import open_packed, SleepEDFPackedDataset


def fold_idx(packed, split_json, fold_id, part):
    with open(split_json) as f: sj = json.load(f)
    subjs = sj["folds"][str(fold_id)][part]
    p2sid = {}
    for i, full in enumerate(packed.subjects):
        p2sid.setdefault(full[:5], []).append(i)
    sids = set()
    for s in subjs: sids.update(p2sid.get(s, []))
    return np.arange(packed.N, dtype=np.int64)[np.isin(np.array(packed.sid), list(sids))]


@torch.no_grad()
def compute_spec(ckpt, packed, idx, device):
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(
        z_dim=int(a.get("z_dim",256)),blocks=int(a.get("blocks",6)),out_ch=int(a.get("out_ch",32)),
        head_hidden=int(a.get("head_hidden",256)),K1=int(a.get("K1",96)),tau=float(a.get("tau",3.5)),
    ).to(device)
    if hasattr(m,"l1") and hasattr(m.l1,"use_pg"): m.l1.use_pg=bool(a.get("use_pg",True))
    m.load_state_dict(s["state_dict"], strict=True); m.eval()
    K = m.l1.proto_logits.shape[0]
    ds = SleepEDFPackedDataset(packed, idx, want_mech=True)
    dl = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True)
    sum_Ac = torch.zeros(K, 5, device=device); class_total = torch.zeros(5, device=device)
    n = 0
    for x30, y, _, _ in dl:
        x30 = x30.to(device); y = y.to(device)
        _,_,_,_,extras = m(x30)
        A = extras["assign1"]
        y_oh = F.one_hot(y, 5).float()
        sum_Ac += A.t() @ y_oh
        class_total += y_oh.sum(0)
        n += y.numel()
    p_c_given_k = sum_Ac.cpu().numpy() / sum_Ac.sum(1, keepdim=True).cpu().numpy().clip(min=1e-8)
    prior = (class_total / n).cpu().numpy()
    spec = p_c_given_k / prior.clip(min=1e-8)
    return spec, K


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = open_packed("/root/autodl-tmp/data/sleepedf_memmap_v2_mech", readonly=True)
    split_json = "/root/autodl-tmp/splits/sleepedf_subject_5fold_seed42.json"
    out_dir = "/root/autodl-tmp/runs/expF_proto_consistency"
    os.makedirs(out_dir, exist_ok=True)

    specs = []
    Ks = []
    for fold in range(5):
        ckpt = f"/root/autodl-tmp/runs/phyg_unified_5fold/fold{fold}/best_model.pt"
        if not os.path.exists(ckpt):
            print(f"[skip] fold{fold} missing", flush=True); continue
        idx = fold_idx(packed, split_json, fold, "test")
        print(f"[fold{fold}] N_test={len(idx)}", flush=True)
        spec, K = compute_spec(ckpt, packed, idx, device)
        specs.append(spec); Ks.append(K)

    # Cross-fold top-K agreement per class
    cls_names = ["W","N1","N2","N3","REM"]
    K = Ks[0]
    report = {"K": int(K), "n_folds": len(specs), "topK_agreement": {}}

    for topK in [5, 10, 20]:
        report["topK_agreement"][f"top{topK}"] = {}
        for c in range(5):
            top_sets = [set(np.argsort(-s[:, c])[:topK].tolist()) for s in specs]
            # pairwise overlap (Jaccard)
            jaccards = []
            for i in range(len(top_sets)):
                for j in range(i+1, len(top_sets)):
                    inter = len(top_sets[i] & top_sets[j])
                    union = len(top_sets[i] | top_sets[j])
                    jaccards.append(inter / max(union, 1))
            report["topK_agreement"][f"top{topK}"][cls_names[c]] = {
                "mean_jaccard": float(np.mean(jaccards)),
                "min_jaccard": float(np.min(jaccards)),
                "max_jaccard": float(np.max(jaccards)),
            }

    # Stable prototypes per class (appear in top-K of ALL folds)
    report["stable_per_class_top10"] = {}
    for c in range(5):
        top_sets = [set(np.argsort(-s[:, c])[:10].tolist()) for s in specs]
        intersection = set.intersection(*top_sets) if top_sets else set()
        report["stable_per_class_top10"][cls_names[c]] = sorted(list(intersection))

    np.savez(os.path.join(out_dir, "spec_per_fold.npz"),
             **{f"spec_fold{i}": s for i, s in enumerate(specs)})
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
