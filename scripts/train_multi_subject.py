"""Multi (MultiChannelSleepNet) subject-level retraining on Sleep-EDF with SHHS zs eval.

Adapts MultiChannelSleepNet to single-channel PhyG data via channel replication.
Computes STFT on-the-fly. 3 seeds with subject 80/10/10 split.
"""
import os, sys, glob, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import spectrogram, resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
sys.path.insert(0, "/root/autodl-tmp")

from src.data_sleepedf_packed import open_packed, subject_split_indices

# Import Multi model & config
sys.path.insert(0, "/root/autodl-tmp/multi_repo")
from model import Transformer
from args import Config


def compute_tf(x_1d, fs=100):
    """Compute STFT TF features. Returns (29, 128) per sample."""
    # x_1d: (N, T) at fs Hz, 30s epoch
    # nperseg=200 (2s), noverlap=100 (1s), nfft=256
    f, t, S = spectrogram(x_1d, fs=fs, window='hamming', nperseg=200, noverlap=100, nfft=256, axis=-1)
    # f.shape (129,), t.shape (29,), S shape (N, 129, 29)
    S = S[:, :128, :]  # (N, 128, 29)
    S = np.log10(S + 1e-10).astype(np.float32)
    # normalize per sample
    S = (S - S.mean(axis=(1, 2), keepdims=True)) / (S.std(axis=(1, 2), keepdims=True) + 1e-6)
    return S.transpose(0, 2, 1)  # → (N, 29, 128) = (time, freq)


class TFDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # (N, 29, 128) single channel
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        # replicate single channel 3 times (Multi expects 3 ch)
        x = np.stack([self.X[i], self.X[i], self.X[i]], axis=0)  # (3, 29, 128)
        return torch.from_numpy(x), int(self.y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_encoder', type=int, default=8, help="reduced from 16 for speed")
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda"

    cfg = Config()
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_encoder = args.num_encoder
    cfg.num_encoder_multi = max(2, args.num_encoder // 4)

    # Load PhyG packed Sleep-EDF
    packed = open_packed("/root/autodl-tmp/data/sleepedf_memmap_v2_mech", readonly=True)
    tr_idx, va_idx, te_idx = subject_split_indices(packed, seed=args.seed)
    print(f"[split seed={args.seed}] tr={len(tr_idx)} va={len(va_idx)} te={len(te_idx)}", flush=True)

    def load_subset(idx):
        x = np.array(packed.X)[idx, 0, :]  # (n, 3000)
        y = np.array(packed.y)[idx]
        return x, y

    X_tr_raw, y_tr = load_subset(tr_idx)
    X_va_raw, y_va = load_subset(va_idx)
    X_te_raw, y_te = load_subset(te_idx)
    print(f"  raw shapes: tr={X_tr_raw.shape} va={X_va_raw.shape} te={X_te_raw.shape}", flush=True)

    # Compute TF in batches to control memory
    def batch_tf(x):
        B = 5000
        outs = []
        for i in range(0, x.shape[0], B):
            outs.append(compute_tf(x[i:i+B], fs=100))
        return np.concatenate(outs, axis=0)
    X_tr = batch_tf(X_tr_raw); del X_tr_raw
    X_va = batch_tf(X_va_raw); del X_va_raw
    X_te = batch_tf(X_te_raw); del X_te_raw
    print(f"  TF shapes: tr={X_tr.shape}", flush=True)

    tr_ld = DataLoader(TFDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    va_ld = DataLoader(TFDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False, num_workers=2)
    te_ld = DataLoader(TFDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = Transformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0; best_state = None
    for ep in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in tr_ld:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x); loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        # Val
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for x, y in va_ld:
                x = x.to(device); logits = model(x)
                vp.append(logits.argmax(-1).cpu().numpy()); vy.append(y.numpy())
        vp = np.concatenate(vp); vy = np.concatenate(vy)
        val_f1 = float(f1_score(vy, vp, average='macro'))
        print(f"[ep {ep}] loss={np.mean(losses):.4f} val_F1={val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # In-domain test
    model.eval(); tp, ty = [], []
    with torch.no_grad():
        for x, y in te_ld:
            x = x.to(device); logits = model(x)
            tp.append(logits.argmax(-1).cpu().numpy()); ty.append(y.numpy())
    tp = np.concatenate(tp); ty = np.concatenate(ty)
    test_pf = f1_score(ty, tp, average=None, labels=list(range(5)))
    test_macro = float(f1_score(ty, tp, average='macro'))
    test_bal = float(balanced_accuracy_score(ty, tp))
    test_kap = float(cohen_kappa_score(ty, tp))
    print(f"[TEST] macro_F1={test_macro:.4f} bal_acc={test_bal:.4f} kappa={test_kap:.4f}", flush=True)
    print(f"[TEST] per_f1=[W={test_pf[0]:.3f},N1={test_pf[1]:.3f},N2={test_pf[2]:.3f},N3={test_pf[3]:.3f},REM={test_pf[4]:.3f}]")

    # SHHS zs
    shhs_files = sorted(glob.glob("/root/autodl-tmp/data/shhs_npz/shhs1-*.npz"))
    sp_all, sy_all = [], []
    with torch.no_grad():
        for f in shhs_files:
            with np.load(f) as z:
                x = z['x'][:, 0, :].astype(np.float32)  # 125Hz, 3750
                y = z['y'][:]
            # 125 → 100 Hz
            x = resample_poly(x, 100, 125, axis=-1)
            if x.shape[1] != 3000:
                x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
            X_tf = compute_tf(x, fs=100)  # (N, 29, 128)
            X_tf3 = np.stack([X_tf, X_tf, X_tf], axis=1)  # (N, 3, 29, 128)
            B = args.batch_size
            preds = []
            for i in range(0, X_tf3.shape[0], B):
                xt = torch.from_numpy(X_tf3[i:i+B]).to(device)
                logits = model(xt)
                preds.append(logits.argmax(-1).cpu().numpy())
            sp_all.append(np.concatenate(preds))
            sy_all.append(y)
    sp = np.concatenate(sp_all); sy = np.concatenate(sy_all)
    shhs_pf = f1_score(sy, sp, average=None, labels=list(range(5)))
    shhs_macro = float(f1_score(sy, sp, average='macro'))
    shhs_bal = float(balanced_accuracy_score(sy, sp))
    shhs_kap = float(cohen_kappa_score(sy, sp))
    print(f"[SHHS zs] macro_F1={shhs_macro:.4f} bal_acc={shhs_bal:.4f} kappa={shhs_kap:.4f}", flush=True)
    print(f"[SHHS zs] per_f1=[W={shhs_pf[0]:.3f},N1={shhs_pf[1]:.3f},N2={shhs_pf[2]:.3f},N3={shhs_pf[3]:.3f},REM={shhs_pf[4]:.3f}]")

    results = {
        'seed': args.seed,
        'best_val_f1': best_f1,
        'test': {'macro_f1': test_macro, 'bal_acc': test_bal, 'kappa': test_kap, 'per_f1': [float(x) for x in test_pf]},
        'shhs_zs': {'macro_f1': shhs_macro, 'bal_acc': shhs_bal, 'kappa': shhs_kap, 'per_f1': [float(x) for x in shhs_pf]},
        'config': vars(args),
    }
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=2)
    torch.save({'state_dict': best_state, 'config': vars(args)}, os.path.join(args.out_dir, 'best_model.pt'))


if __name__ == '__main__':
    main()
