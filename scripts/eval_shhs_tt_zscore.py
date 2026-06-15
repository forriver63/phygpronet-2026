"""Exp 9: SHHS zs with test-time per-RECORDING z-score (NOT training-time).

For each SHHS recording, compute mean/std across all epochs in that recording,
z-score the EEG signal, then run inference. Compare to no-normalization baseline.
"""
import sys, os, glob, argparse
import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--zscore", choices=["none","per_record","per_epoch"], default="per_record")
    args = ap.parse_args()
    device = "cuda"
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(z_dim=int(a.get("z_dim",256)), blocks=int(a.get("blocks",6)),
        out_ch=int(a.get("out_ch",32)), head_hidden=int(a.get("head_hidden",256)),
        K1=int(a.get("K1",96)), tau=float(a.get("tau",3.5))).to(device)
    if hasattr(m, "l1") and hasattr(m.l1, "use_pg"): m.l1.use_pg = bool(a.get("use_pg", True))
    m.load_state_dict(s["state_dict"], strict=True)
    m.eval()

    files = sorted(glob.glob(os.path.join(args.shhs_dir, "shhs1-*.npz")))
    all_preds, all_labels = [], []
    with torch.no_grad():
        for f in files:
            with np.load(f) as z:
                x = z["x"][:, 0, :].astype(np.float32)   # (n_ep, 3750) eeg_ch=0
                y = z["y"][:].astype(np.int64)
            # resample 125->100
            x = resample_poly(x, 100, 125, axis=-1)
            if x.shape[1] != 3000:
                x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
            # apply normalization
            if args.zscore == "per_record":
                # z-score per recording using all epochs' samples
                mu = x.mean(); sd = x.std() + 1e-6
                x = (x - mu) / sd
            elif args.zscore == "per_epoch":
                mu = x.mean(axis=1, keepdims=True); sd = x.std(axis=1, keepdims=True) + 1e-6
                x = (x - mu) / sd
            # forward
            x_t = torch.from_numpy(x[:, None, :]).to(device, non_blocking=True)
            B = 256
            preds = []
            for i in range(0, x_t.shape[0], B):
                _, _, _, p5, _ = m(x_t[i:i+B])
                preds.append(p5.argmax(-1).cpu().numpy())
            all_preds.append(np.concatenate(preds))
            all_labels.append(y)
    p = np.concatenate(all_preds); t = np.concatenate(all_labels)
    pf = f1_score(t, p, average=None, labels=list(range(5)))
    print(f"[zscore={args.zscore}] macro_f1={f1_score(t,p,average='macro'):.4f} bal_acc={balanced_accuracy_score(t,p):.4f} kappa={cohen_kappa_score(t,p):.4f}")
    print(f"per_f1=[W={pf[0]:.3f},N1={pf[1]:.3f},N2={pf[2]:.3f},N3={pf[3]:.3f},REM={pf[4]:.3f}]")


if __name__ == "__main__":
    main()
