"""Exp 1: transfer-aware checkpoint selection — per-epoch evaluator.

For one fold directory containing epoch_01.pt..epoch_50.pt:
  - Load each ckpt
  - Compute val Sleep-EDF macro-F1 (from ckpt['val'])
  - Compute alignment metrics on val set:
      * spindle_N2_frac  — top-N spindle-loaded prototypes, frac N2-dominant
      * slowwave_N3_frac — top-N slowwave-loaded prototypes, frac N3-dominant
      * proto_entropy    — usage entropy across active prototypes
      * within_class_edge_mass — PG-R adjacency within-class mass fraction (if use_pg)
  - Compute SHHS zero-shot macro-F1
  - Output rows to <out_csv>

Usage:
  python eval_exp1_perckpt.py \
    --fold_dir /root/autodl-tmp/runs/exp1_saveall/fold0 \
    --fold_id 0 \
    --split_json /root/autodl-tmp/splits/sleepedf_subject_5fold_seed42.json \
    --packed_dir /root/autodl-tmp/data/sleepedf_memmap_v2_mech \
    --shhs_dir /root/autodl-tmp/data/shhs_npz \
    --out_csv /root/autodl-tmp/runs/exp1_saveall/fold0/perepoch_eval.csv
"""
from __future__ import annotations
import os, sys, json, glob, argparse, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
sys.path.insert(0, "/root/autodl-tmp/code/PhyGProNet-Repro")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.data_sleepedf_packed import open_packed, SleepEDFPackedDataset


def fold_indices(packed, split_json, fold_id, partition):
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


def build_model(ckpt_path, device):
    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(
        z_dim=int(a.get("z_dim", 256)),
        blocks=int(a.get("blocks", 6)),
        out_ch=int(a.get("out_ch", 32)),
        head_hidden=int(a.get("head_hidden", 256)),
        K1=int(a.get("K1", 96)),
        tau=float(a.get("tau", 3.5)),
    ).to(device)
    if hasattr(m, "l1") and hasattr(m.l1, "use_pg"):
        m.l1.use_pg = bool(a.get("use_pg", True))
    m.load_state_dict(s["state_dict"], strict=True)
    m.eval()
    return m, s


@torch.no_grad()
def eval_val_and_alignment(model, dl, device, topN=20, active_thresh=0.001):
    K = model.l1.proto_logits.shape[0]
    sum_A  = torch.zeros(K, device=device)
    sum_sp = torch.zeros(K, device=device)
    sum_sw = torch.zeros(K, device=device)
    sum_Ac = torch.zeros(K, 5, device=device)
    class_total = torch.zeros(5, device=device)
    preds, ys = [], []
    for x30, y, sp, sw in dl:
        x30 = x30.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sp = sp.to(device, non_blocking=True)
        sw = sw.to(device, non_blocking=True)
        # model returns: z, (l1,l2,l3), (p1,p2,p3), p5, extras
        _, _, _, p5, extras = model(x30)
        logits5 = p5  # already softmax probs; argmax invariant
        A = extras.get("assign1") if isinstance(extras, dict) else None
        if A is not None:
            sum_A  += A.sum(dim=0)
            sum_sp += A.t().mv(sp)
            sum_sw += A.t().mv(sw)
            oh = F.one_hot(y, num_classes=5).float()
            sum_Ac += A.t().mm(oh)
            class_total += oh.sum(dim=0)
        preds.append(logits5.argmax(-1).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    val_f1 = float(f1_score(y_true, y_pred, average="macro"))

    sum_A_np = sum_A.cpu().numpy()
    total_A = sum_A_np.sum() + 1e-12
    active_mask = (sum_A_np / total_A) >= active_thresh
    n_active = int(active_mask.sum())

    if A is not None and n_active > 0:
        # per-prototype average mech load
        avg_sp = (sum_sp / sum_A.clamp_min(1e-8)).cpu().numpy()
        avg_sw = (sum_sw / sum_A.clamp_min(1e-8)).cpu().numpy()
        # per-prototype class probability
        p_c_given_k = (sum_Ac / sum_A.clamp_min(1e-8).unsqueeze(1)).cpu().numpy()
        p_c = (class_total / class_total.sum().clamp_min(1e-8)).cpu().numpy()
        spec = p_c_given_k / (p_c[None, :] + 1e-12)
        dom_cls = spec.argmax(axis=1)
        active_idx = np.where(active_mask)[0]
        # top-N by spindle load, slowwave load (descending)
        order_sp = np.argsort(-avg_sp[active_idx])[:topN]
        order_sw = np.argsort(-avg_sw[active_idx])[:topN]
        top_sp_dom = dom_cls[active_idx[order_sp]]
        top_sw_dom = dom_cls[active_idx[order_sw]]
        spindle_N2_frac = float((top_sp_dom == 2).mean())  # N2=2
        slowwave_N3_frac = float((top_sw_dom == 3).mean())  # N3=3
        # entropy of usage across active prototypes
        usage = sum_A_np[active_mask] / sum_A_np[active_mask].sum()
        proto_entropy = float(-(usage * np.log(usage + 1e-12)).sum() / np.log(max(n_active, 2)))

        # pearson-like normalized dot-product correlations between proto N3-logit vs slowwave
        # (using proto_logits attribute on the head — Level-1 logits over [N3, N2, Other])
        try:
            proto_logits = model.l1.proto_logits.detach().cpu().numpy()
            corr_sw_n3 = float(np.corrcoef(avg_sw[active_mask], proto_logits[active_mask, 0])[0, 1])
            corr_sp_n2 = float(np.corrcoef(avg_sp[active_mask], proto_logits[active_mask, 1])[0, 1])
        except Exception:
            corr_sw_n3 = corr_sp_n2 = float("nan")

        # within-class edge mass via PG-R adjacency
        within_frac = float("nan")
        if hasattr(model.l1, "pg") and model.l1.pg is not None:
            try:
                proto_vec = model.l1.proto.detach()
                Aadj = model.l1.pg.build_adj(proto_vec).cpu().numpy()
                mass = 0.0; w_in = 0.0
                for i in range(K):
                    for j in range(K):
                        w = Aadj[i, j]
                        if w > 1e-8:
                            mass += w
                            if dom_cls[i] == dom_cls[j]:
                                w_in += w
                within_frac = float(w_in / mass) if mass > 0 else float("nan")
            except Exception:
                pass
    else:
        spindle_N2_frac = slowwave_N3_frac = proto_entropy = corr_sw_n3 = corr_sp_n2 = within_frac = float("nan")
        n_active = 0

    return {
        "val_f1": val_f1,
        "n_active": n_active,
        "spindle_N2_frac": spindle_N2_frac,
        "slowwave_N3_frac": slowwave_N3_frac,
        "proto_entropy": proto_entropy,
        "corr_sw_n3": corr_sw_n3,
        "corr_sp_n2": corr_sp_n2,
        "within_class_edge_mass": within_frac,
    }


class SHHSAllDataset(Dataset):
    """Iterate all SHHS subjects sequentially."""
    def __init__(self, shhs_dir, eeg_ch=0, max_subjects=None):
        files = sorted(glob.glob(os.path.join(shhs_dir, "shhs1-*.npz")))
        if max_subjects is not None:
            files = files[:max_subjects]
        self.files = files
        self.eeg_ch = eeg_ch
        self.index = []
        for fi, f in enumerate(files):
            with np.load(f, mmap_mode="r") as z:
                n = z["y"].shape[0]
            self.index.extend([(fi, i) for i in range(n)])
        self._cache = {}
        self._cache_limit = 4

    def _load(self, fi):
        if fi not in self._cache:
            if len(self._cache) >= self._cache_limit:
                self._cache.pop(next(iter(self._cache)))
            with np.load(self.files[fi]) as z:
                self._cache[fi] = {"x": z["x"][:], "y": z["y"][:]}
        return self._cache[fi]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        fi, si = self.index[idx]
        d = self._load(fi)
        x = d["x"][si, self.eeg_ch, :].astype(np.float32)
        x = resample_poly(x, 100, 125)
        if x.shape[0] != 3000:
            x = (x[:3000] if x.shape[0] > 3000 else np.pad(x, (0, 3000 - x.shape[0])))
        return torch.from_numpy(x[None, :].astype(np.float32)), int(d["y"][si])


@torch.no_grad()
def eval_shhs_zs(model, shhs_dl, device):
    preds, ys = [], []
    for x, y in shhs_dl:
        x = x.to(device, non_blocking=True)
        _, _, _, p5, _ = model(x)
        preds.append(p5.argmax(-1).cpu().numpy())
        ys.append(np.asarray(y))
    p = np.concatenate(preds); t = np.concatenate(ys)
    return {
        "shhs_zs_f1": float(f1_score(t, p, average="macro")),
        "shhs_zs_bal_acc": float(balanced_accuracy_score(t, p)),
        "shhs_zs_kappa": float(cohen_kappa_score(t, p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", required=True)
    ap.add_argument("--fold_id", type=int, required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--topN", type=int, default=20)
    ap.add_argument("--skip_shhs", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    packed = open_packed(args.packed_dir, readonly=True)
    val_idx = fold_indices(packed, args.split_json, args.fold_id, "val")
    val_ds = SleepEDFPackedDataset(packed, val_idx, want_mech=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[val] fold{args.fold_id} N={len(val_idx)}")

    if not args.skip_shhs:
        shhs_ds = SHHSAllDataset(args.shhs_dir)
        shhs_dl = DataLoader(shhs_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
        print(f"[shhs] N samples={len(shhs_ds)}")

    ckpts = sorted(glob.glob(os.path.join(args.fold_dir, "epoch_*.pt")))
    print(f"[main] found {len(ckpts)} ckpts in {args.fold_dir}")

    fields = ["epoch", "val_f1", "n_active", "spindle_N2_frac", "slowwave_N3_frac",
              "proto_entropy", "corr_sw_n3", "corr_sp_n2", "within_class_edge_mass",
              "shhs_zs_f1", "shhs_zs_bal_acc", "shhs_zs_kappa"]
    rows = []
    for ck in ckpts:
        ep = int(os.path.basename(ck).replace("epoch_", "").replace(".pt", ""))
        print(f"\n[eval] epoch={ep}")
        model, _ = build_model(ck, device)
        align = eval_val_and_alignment(model, val_dl, device, topN=args.topN)
        if not args.skip_shhs:
            zs = eval_shhs_zs(model, shhs_dl, device)
        else:
            zs = {"shhs_zs_f1": float("nan"), "shhs_zs_bal_acc": float("nan"), "shhs_zs_kappa": float("nan")}
        row = {"epoch": ep}; row.update(align); row.update(zs)
        rows.append(row)
        print(f"  val_F1={row['val_f1']:.4f}  sp_N2={row['spindle_N2_frac']:.2f}  sw_N3={row['slowwave_N3_frac']:.2f}  "
              f"ent={row['proto_entropy']:.3f}  edge_in={row['within_class_edge_mass']:.3f}  "
              f"shhs_zs={row['shhs_zs_f1']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[DONE] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
