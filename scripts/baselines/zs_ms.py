"""Eval multi-scale PhyG ckpt on SHHS zero-shot."""
import sys, os, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
sys.path.insert(0, "/root/autodl-tmp/code/PhyGProNet-Repro")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from phygpronet.data.shhs_npz import SHHSNpzDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state["state_dict"]
    a = state.get("args", {})
    model = InceptionTreeProtoL1(
        z_dim=int(a.get("z_dim", 256)),
        blocks=int(a.get("blocks", 6)),
        out_ch=int(a.get("out_ch", 32)),
        head_hidden=int(a.get("head_hidden", 256)),
        K1=int(a.get("K1", 96)),
        tau=float(a.get("tau", 3.5)),
        kernel_sizes=tuple(a.get("kernel_sizes", [9, 19, 39])),
    ).to(device)
    if hasattr(model, "l1") and hasattr(model.l1, "use_pg"):
        model.l1.use_pg = bool(a.get("use_pg", True))
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[loaded] kernel_sizes={a.get('kernel_sizes')} K1={a.get('K1')}", flush=True)

    ds = SHHSNpzDataset(args.data_root)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device).float()
            _, _, _, p5, _ = model(x)
            ps.append(p5.argmax(-1).cpu().numpy())
            ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    p = np.concatenate(ps); y = np.concatenate(ys)
    f1 = float(f1_score(y, p, average="macro"))
    ba = float(balanced_accuracy_score(y, p))
    kp = float(cohen_kappa_score(y, p))
    per = f1_score(y, p, average=None, labels=[0,1,2,3,4]).tolist()
    print(f"[SHHS zs] F1={f1:.4f} bal_acc={ba:.4f} kappa={kp:.4f}")
    print(f"  per: W={per[0]:.3f} N1={per[1]:.3f} N2={per[2]:.3f} N3={per[3]:.3f} REM={per[4]:.3f}")

    np.savez(os.path.join(args.out_dir, "shhs_preds.npz"), preds=p, labels=y)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f_:
        json.dump({"macro_f1": f1, "bal_acc": ba, "kappa": kp, "per_class": per}, f_, indent=2)


if __name__ == "__main__":
    main()
