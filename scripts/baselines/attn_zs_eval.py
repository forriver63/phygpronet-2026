"""Evaluate AttnSleep 5-fold ckpts on MASS/ISRUC zero-shot.
Reuses PhyGProNet-Repro dataset loaders for consistency.
"""
import argparse, json, os, sys
sys.path.insert(0, "/root/autodl-tmp/code/毕业设计/AttnSleep-main/AttnSleep-main")
sys.path.insert(0, "/root/autodl-tmp/code/PhyGProNet-Repro")

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

from model.model import AttnSleep


def load_dataset(target, root):
    if target == "mass":
        from phygpronet.data.mass_npz import MASSNpzDataset
        return MASSNpzDataset(root)
    if target == "shhs":
        from phygpronet.data.shhs_npz import SHHSNpzDataset
        return SHHSNpzDataset(root)
    if target == "isruc":
        from phygpronet.data.isruc_npz import ISRUCNpzDataset
        return ISRUCNpzDataset(root)
    raise ValueError(target)


def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            out = model(x)
            ps.append(out.argmax(-1).cpu().numpy())
            ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    return np.concatenate(ps), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", required=True, choices=["mass", "shhs", "isruc"])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AttnSleep().to(device)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state["state_dict"]
    model.load_state_dict(sd)

    ds = load_dataset(args.target, args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2, shuffle=False)
    preds, labels = evaluate(model, loader, device)

    f1 = f1_score(labels, preds, average="macro")
    ba = balanced_accuracy_score(labels, preds)
    kp = cohen_kappa_score(labels, preds)
    per = f1_score(labels, preds, average=None, labels=[0, 1, 2, 3, 4])

    print(f"[{args.target}] macro-F1={f1:.4f} bal_acc={ba:.4f} kappa={kp:.4f}")
    print(f"  per-class: {[round(float(v),3) for v in per]}")

    np.savez(os.path.join(args.out_dir, f"{args.target}_preds.npz"),
             preds=preds, labels=labels)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"f1": float(f1), "bal_acc": float(ba), "kappa": float(kp),
                   "per_class": [float(v) for v in per]}, f, indent=2)


if __name__ == "__main__":
    main()
