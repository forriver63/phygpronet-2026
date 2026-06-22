"""ZS eval for PhyG multi-band+per-rec-norm model: SHHS, MASS, ISRUC.

Applies per-recording z-score on each foreign recording (computed on that recording only — no leakage).
"""
import os, sys, glob, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.metrics import metrics


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    model = InceptionTreeProtoL1(
        z_dim=a.get("z_dim", 256), blocks=a.get("blocks", 6),
        out_ch=a.get("out_ch", 32), head_hidden=a.get("head_hidden", 256),
        K1=a.get("K1", 192), tau=a.get("tau", 3.5),
    ).to(device)
    if hasattr(model, "l1") and hasattr(model.l1, "use_pg"):
        model.l1.use_pg = bool(a.get("use_pg", True))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, a


def preprocess_shhs(x, eeg_ch=0):
    """x: (N,6,3750)@125Hz -> (N,1,3000)@100Hz."""
    x = x[:, eeg_ch, :]
    x = resample_poly(x, up=4, down=5, axis=-1)
    return x[:, None, :].astype(np.float32)


def preprocess_mass(x):
    """MASS npz is already (N,1,3000)@100Hz typically. We try ch0 if shape mismatched."""
    if x.ndim == 3:
        if x.shape[1] != 1:
            x = x[:, :1, :]
        return x.astype(np.float32)
    elif x.ndim == 2:
        return x[:, None, :].astype(np.float32)
    raise ValueError(f"bad MASS shape {x.shape}")


def per_recording_zscore(x):
    """x: (N,1,L). Z-score with that recording's own mean/std."""
    mu = float(x.mean())
    sd = float(x.std()) + 1e-8
    return ((x - mu) / sd).astype(np.float32)


@torch.no_grad()
def eval_dir(model, files, device, preproc, per_rec_norm, batch_size=128, eeg_ch=0):
    yt, yp = [], []
    for fp in files:
        d = np.load(fp)
        x = d["x"]
        y = d["y"].astype(int)
        if preproc == "shhs":
            x = preprocess_shhs(x, eeg_ch=eeg_ch)
        else:
            x = preprocess_mass(x)
        if per_rec_norm:
            x = per_recording_zscore(x)
        N = x.shape[0]
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(x[i:i+batch_size]).to(device)
            _, _, _, p5, _ = model(xb)
            yp.extend(p5.argmax(-1).cpu().numpy().tolist())
            yt.extend(y[i:i+batch_size].tolist())
    yt = np.array(yt); yp = np.array(yp)
    m = metrics(yt, yp, n_classes=5)
    cm = confusion_matrix(yt, yp, labels=[0,1,2,3,4])
    return m, cm, yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--mass_dir", default="/root/autodl-tmp/data/mass_npz")
    ap.add_argument("--isruc_dir", default="")
    ap.add_argument("--no_per_rec_norm", action="store_true",
                    help="disable per-recording z-score even if model was trained with it")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, a = load_model(args.ckpt, device)
    per_rec = bool(a.get("per_recording_norm", False)) and not args.no_per_rec_norm
    print(f"[model] K1={a.get('K1')} mech_w={a.get('mech_w')} multi_band={a.get('multi_band_mech')} per_rec_norm={per_rec}")

    out = {"ckpt": args.ckpt, "per_rec_norm": per_rec}

    if args.shhs_dir and os.path.isdir(args.shhs_dir):
        files = sorted(glob.glob(os.path.join(args.shhs_dir, "*.npz")))
        print(f"[SHHS] {len(files)} files")
        m, cm, _, _ = eval_dir(model, files, device, "shhs", per_rec, args.batch_size)
        print(f"[SHHS] F1={m['macro_f1']:.4f}  BalAcc={m['bal_acc']:.4f}  per_f1={m['per_f1']}")
        out["shhs"] = {**m, "cm": cm.tolist()}

    if args.mass_dir and os.path.isdir(args.mass_dir):
        files = sorted(glob.glob(os.path.join(args.mass_dir, "*.npz")))
        print(f"[MASS] {len(files)} files")
        m, cm, _, _ = eval_dir(model, files, device, "mass", per_rec, args.batch_size)
        print(f"[MASS] F1={m['macro_f1']:.4f}  BalAcc={m['bal_acc']:.4f}  per_f1={m['per_f1']}")
        out["mass"] = {**m, "cm": cm.tolist()}

    if args.isruc_dir and os.path.isdir(args.isruc_dir):
        files = sorted(glob.glob(os.path.join(args.isruc_dir, "*.npz")))
        print(f"[ISRUC] {len(files)} files")
        m, cm, _, _ = eval_dir(model, files, device, "mass", per_rec, args.batch_size)
        print(f"[ISRUC] F1={m['macro_f1']:.4f}  BalAcc={m['bal_acc']:.4f}  per_f1={m['per_f1']}")
        out["isruc"] = {**m, "cm": cm.tolist()}

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
