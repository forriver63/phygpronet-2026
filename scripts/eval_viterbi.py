"""Exp 5: Viterbi smoothing post-hoc on SHHS zs predictions.

Loads PhyG ckpt, predicts per-recording on SHHS, applies Viterbi using transition
matrix estimated from Sleep-EDF train labels. Compares no-smoothing vs Viterbi.
"""
import sys, os, glob, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.data_sleepedf_packed import open_packed, subject_split_indices


def estimate_transition_matrix(packed, tr_idx):
    """Estimate 5x5 transition matrix from training labels grouped by recording."""
    # tr_idx are global sample indices into packed.y / packed.sid
    y_all = np.array(packed.y)[tr_idx]
    sid_all = np.array(packed.sid)[tr_idx]
    # build per-subject sequences (recordings)
    T = np.zeros((5, 5), dtype=np.float64)
    init = np.zeros(5, dtype=np.float64)
    for sid in np.unique(sid_all):
        m = sid_all == sid
        seq = y_all[m]  # already in chronological order if memmap was packed that way
        if len(seq) < 2: continue
        init[seq[0]] += 1
        for t in range(1, len(seq)):
            T[seq[t-1], seq[t]] += 1
    init = init / init.sum().clip(1e-8)
    T = T / T.sum(axis=1, keepdims=True).clip(1e-8)
    return T, init


def viterbi(emissions, T, init, smooth=1.0):
    """emissions: (N, 5) class probabilities per epoch (already prob).
    T: 5x5 transition matrix
    init: (5,) initial distribution
    smooth: blend factor — T_effective = T * smooth + uniform * (1-smooth)
    Returns predicted class per epoch.
    """
    N, K = emissions.shape
    # blend transition for smoothing strength control
    if smooth < 1.0:
        unif = np.ones_like(T) / K
        T = T * smooth + unif * (1.0 - smooth)
    log_e = np.log(emissions.clip(1e-12))
    log_T = np.log(T.clip(1e-12))
    log_init = np.log(init.clip(1e-12))
    delta = np.empty((N, K))
    psi = np.empty((N, K), dtype=np.int32)
    delta[0] = log_init + log_e[0]
    psi[0] = 0
    for t in range(1, N):
        # delta[t-1] + log_T -> (K_prev,K_cur)
        scores = delta[t-1, :, None] + log_T   # (K,K)
        delta[t] = scores.max(axis=0) + log_e[t]
        psi[t] = scores.argmax(axis=0)
    # backtrack
    path = np.empty(N, dtype=np.int32)
    path[-1] = delta[-1].argmax()
    for t in range(N-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--packed_dir", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--smooth", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    args = ap.parse_args()
    device = "cuda"

    # Load model
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(z_dim=int(a.get("z_dim",256)), blocks=int(a.get("blocks",6)),
        out_ch=int(a.get("out_ch",32)), head_hidden=int(a.get("head_hidden",256)),
        K1=int(a.get("K1",96)), tau=float(a.get("tau",3.5))).to(device)
    if hasattr(m, "l1") and hasattr(m.l1, "use_pg"): m.l1.use_pg = bool(a.get("use_pg", True))
    m.load_state_dict(s["state_dict"], strict=True)
    m.eval()

    # Estimate transition matrix from Sleep-EDF train of this seed
    seed = int(a.get("seed", 42))
    packed = open_packed(args.packed_dir, readonly=True)
    tr_idx, _, _ = subject_split_indices(packed, seed=seed)
    T, init = estimate_transition_matrix(packed, tr_idx)
    print(f"[T] estimated from {len(tr_idx)} train samples, diag={[f'{T[i,i]:.2f}' for i in range(5)]}")

    # Predict SHHS per recording
    files = sorted(glob.glob(os.path.join(args.shhs_dir, "shhs1-*.npz")))
    all_emissions, all_labels = [], []
    with torch.no_grad():
        for f in files:
            with np.load(f) as z:
                x = z["x"][:, 0, :].astype(np.float32)
                y = z["y"][:].astype(np.int64)
            x = resample_poly(x, 100, 125, axis=-1)
            if x.shape[1] != 3000:
                x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
            x_t = torch.from_numpy(x[:, None, :]).to(device, non_blocking=True)
            preds_prob = []
            B = 256
            for i in range(0, x_t.shape[0], B):
                _, _, _, p5, _ = m(x_t[i:i+B])
                preds_prob.append(p5.cpu().numpy())
            all_emissions.append(np.concatenate(preds_prob))
            all_labels.append(y)

    # Baseline (no Viterbi)
    base_preds = np.concatenate([e.argmax(axis=1) for e in all_emissions])
    labels = np.concatenate(all_labels)
    pf = f1_score(labels, base_preds, average=None, labels=list(range(5)))
    print(f"[no smoothing] macro_F1={f1_score(labels,base_preds,average='macro'):.4f} per=[W={pf[0]:.3f},N1={pf[1]:.3f},N2={pf[2]:.3f},N3={pf[3]:.3f},REM={pf[4]:.3f}]")

    # Viterbi at various smooth strengths
    for sm in args.smooth:
        if sm <= 0: continue
        smoothed_preds = []
        for emis in all_emissions:
            smoothed_preds.append(viterbi(emis, T, init, smooth=sm))
        smp = np.concatenate(smoothed_preds)
        pf = f1_score(labels, smp, average=None, labels=list(range(5)))
        print(f"[viterbi smooth={sm:.2f}] macro_F1={f1_score(labels,smp,average='macro'):.4f} bal_acc={balanced_accuracy_score(labels,smp):.4f} kappa={cohen_kappa_score(labels,smp):.4f} per=[W={pf[0]:.3f},N1={pf[1]:.3f},N2={pf[2]:.3f},N3={pf[3]:.3f},REM={pf[4]:.3f}]")


if __name__ == "__main__":
    main()
