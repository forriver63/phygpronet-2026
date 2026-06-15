"""Evaluate PhyG ckpt on MASS or ISRUC zero-shot."""
import sys, os, glob, argparse, json
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1


def load_model(ckpt, device):
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(z_dim=int(a.get("z_dim",256)), blocks=int(a.get("blocks",6)),
        out_ch=int(a.get("out_ch",32)), head_hidden=int(a.get("head_hidden",256)),
        K1=int(a.get("K1",96)), tau=float(a.get("tau",3.5))).to(device)
    if hasattr(m, "l1") and hasattr(m.l1, "use_pg"): m.l1.use_pg = bool(a.get("use_pg", True))
    m.load_state_dict(s["state_dict"], strict=True)
    m.eval()
    return m


@torch.no_grad()
def predict_batch(model, x_np, device, bs=256):
    """x_np: (N, 3000) at 100Hz. Returns predictions."""
    x_t = torch.from_numpy(x_np[:, None, :].astype(np.float32)).to(device, non_blocking=True)
    preds = []
    for i in range(0, x_t.shape[0], bs):
        _, _, _, p5, _ = model(x_t[i:i+bs])
        preds.append(p5.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def eval_mass(model, mass_dir, device):
    files = sorted(glob.glob(os.path.join(mass_dir, "*.npz")))
    preds_all, y_all = [], []
    for f in files:
        with np.load(f) as z:
            data = z["data"]  # (N, 27, 7680) at 256Hz
            y = z["labels"]
        # use channel 0 (assume Fpz-Cz-equivalent); resample 256→100Hz
        x = data[:, 0, :].astype(np.float32)
        x = resample_poly(x, 100, 256, axis=-1)  # (N, 3000)
        if x.shape[1] != 3000:
            x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
        preds_all.append(predict_batch(model, x, device))
        y_all.append(y)
    return np.concatenate(preds_all), np.concatenate(y_all)


def eval_isruc(model, isruc_dir, device):
    """ISRUC: 3 subsets, each is dict {subject_id: {edf_data, labels}}.
    edf_data: list of (13, 6000) arrays at 200Hz. resample to 100Hz (3000 samples)."""
    preds_all, y_all = [], []
    for sub in ["sup0", "sup1", "sup2"]:
        seg_p = os.path.join(isruc_dir, f"isruc_segmented_{sub}.npz")
        lab_p = os.path.join(isruc_dir, f"isruc_labels_{sub}.npz")
        if not os.path.exists(seg_p): continue
        z_seg = np.load(seg_p, allow_pickle=True)
        z_lab = np.load(lab_p, allow_pickle=True)
        labels = z_lab["labels"]
        sids = sorted(z_seg.files)
        offset = 0
        for sid in sids:
            d = z_seg[sid].item()
            edf_list = d["edf_data"]
            n = len(edf_list)
            # use channel 1 (C3-A2) per the existing ISRUCNpzDataset default; scale by 1e-5
            xs = np.stack([edf_list[i][1].astype(np.float32) for i in range(n)], axis=0) * 1e-5
            x = resample_poly(xs, 100, 200, axis=-1)
            if x.shape[1] != 3000:
                x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
            preds_all.append(predict_batch(model, x, device))
            y_sub = []
            for i in range(n):
                v = int(str(labels[offset+i]).strip())
                y_sub.append(4 if v == 5 else v)
            y_all.append(np.array(y_sub, dtype=np.int64))
            offset += n
    return np.concatenate(preds_all), np.concatenate(y_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", choices=["mass", "isruc"], required=True)
    ap.add_argument("--mass_dir", default="/root/autodl-tmp/data/mass_npz")
    ap.add_argument("--isruc_dir", default="/root/autodl-tmp/data/isruc_npz")
    args = ap.parse_args()
    device = "cuda"
    m = load_model(args.ckpt, device)
    if args.target == "mass":
        p, y = eval_mass(m, args.mass_dir, device)
    else:
        p, y = eval_isruc(m, args.isruc_dir, device)
    pf = f1_score(y, p, average=None, labels=list(range(5)))
    print(f"[{args.target}] macro_F1={f1_score(y,p,average='macro'):.4f} bal_acc={balanced_accuracy_score(y,p):.4f} kappa={cohen_kappa_score(y,p):.4f}")
    print(f"per_f1=[W={pf[0]:.3f},N1={pf[1]:.3f},N2={pf[2]:.3f},N3={pf[3]:.3f},REM={pf[4]:.3f}]")


if __name__ == "__main__":
    main()
