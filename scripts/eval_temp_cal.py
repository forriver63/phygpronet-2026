"""Exp 8: Temperature calibration.
Fit temperature T on Sleep-EDF val to minimize NLL; apply 1/T scaling to SHHS log-probs at inference.
PhyG outputs p5 (already softmax probs). Effective logit = log(p5). Divide by T, re-softmax.
"""
import sys, os, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.data_sleepedf_packed import open_packed, SleepEDFPackedDataset, subject_split_indices


def fit_T(logp5_val, y_val, n_iters=200, lr=0.05):
    """Fit T by minimizing NLL on val."""
    T = torch.nn.Parameter(torch.ones(1, device=logp5_val.device))
    opt = torch.optim.Adam([T], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        scaled = logp5_val / T.clamp_min(0.1)
        # re-normalize as log-probs
        logp = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
        loss = F.nll_loss(logp, y_val)
        loss.backward(); opt.step()
    return float(T.detach().cpu().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
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

    # 1) Compute val log-probs
    seed = int(a.get("seed", 42))
    packed = open_packed(args.packed_dir, readonly=True)
    _, va_idx, _ = subject_split_indices(packed, seed=seed)
    va_ds = SleepEDFPackedDataset(packed, va_idx, want_mech=True)
    va_ld = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=4)
    logps, ys = [], []
    with torch.no_grad():
        for x30, y, sp, sw in va_ld:
            x30 = x30.to(device); y = y.to(device)
            _, _, _, p5, _ = m(x30)
            logps.append(p5.clamp_min(1e-8).log()); ys.append(y)
    logps = torch.cat(logps); ys = torch.cat(ys)
    T = fit_T(logps.detach(), ys.detach())
    print(f"fitted T={T:.4f}")

    # 2) Apply to SHHS
    files = sorted(glob.glob(os.path.join(args.shhs_dir, "shhs1-*.npz")))
    all_preds_base, all_preds_T, all_y = [], [], []
    with torch.no_grad():
        for f in files:
            with np.load(f) as z:
                x = z["x"][:, 0, :].astype(np.float32)
                y = z["y"][:].astype(np.int64)
            x = resample_poly(x, 100, 125, axis=-1)
            if x.shape[1] != 3000:
                x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
            x_t = torch.from_numpy(x[:, None, :]).to(device, non_blocking=True)
            base_preds = []
            T_preds = []
            for i in range(0, x_t.shape[0], 256):
                _, _, _, p5, _ = m(x_t[i:i+256])
                base_preds.append(p5.argmax(-1).cpu().numpy())
                logp = p5.clamp_min(1e-8).log() / T
                T_preds.append(logp.argmax(-1).cpu().numpy())
            all_preds_base.append(np.concatenate(base_preds))
            all_preds_T.append(np.concatenate(T_preds))
            all_y.append(y)
    pb = np.concatenate(all_preds_base); pT = np.concatenate(all_preds_T); yy = np.concatenate(all_y)
    pf = f1_score(yy, pb, average=None, labels=list(range(5)))
    print(f"[no T] macro_F1={f1_score(yy,pb,average='macro'):.4f} per=[W={pf[0]:.3f},N1={pf[1]:.3f},N2={pf[2]:.3f},N3={pf[3]:.3f},REM={pf[4]:.3f}]")
    pf2 = f1_score(yy, pT, average=None, labels=list(range(5)))
    print(f"[T={T:.2f}] macro_F1={f1_score(yy,pT,average='macro'):.4f} per=[W={pf2[0]:.3f},N1={pf2[1]:.3f},N2={pf2[2]:.3f},N3={pf2[3]:.3f},REM={pf2[4]:.3f}]")


if __name__ == "__main__":
    main()
