"""Ensemble TTA SHHS zero-shot for CBraMod (container B).

6 ckpts x (1 orig + 4 noise) = 30 log-prob predictions, median-aggregated.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
from scipy.signal import resample_poly

CBRAMOD_DIR = "/root/autodl-tmp/code/baselines-code - clean/CBraMod_Code/CBraMod-main"
sys.path.insert(0, CBRAMOD_DIR)
from models.cbramod import CBraMod


class SHHSCBraMod(Dataset):
    """SHHS @125Hz ch0 -> 200Hz -> (1,30,200) with V->1e4 scaling (matching CBraMod pretrain unit)."""
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "shhs1-*.npz")))
        self.index = []
        for fi, f in enumerate(self.files):
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
                self._cache[fi] = {"x": z["x"][:].astype(np.float32), "y": z["y"][:].astype(np.int64)}
        return self._cache[fi]
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        fi, si = self.index[idx]
        d = self._load(fi)
        x = d["x"][si, 0, :]
        x6 = resample_poly(x, 200, 125).astype(np.float32)
        if x6.shape[0] != 6000:
            x6 = x6[:6000] if x6.shape[0] > 6000 else np.pad(x6, (0, 6000-x6.shape[0]))
        x6 = (x6 * 10000.0).reshape(1, 30, 200).astype(np.float32)
        return torch.from_numpy(x6), int(d["y"][si])


def build_cbramod_from_ckpt(ckpt_path, device):
    sys.path.insert(0, "/root/autodl-tmp")
    from train_cbramod_old import CBraModSleep
    s = torch.load(ckpt_path, map_location="cpu")
    sd = s.get("state_dict", s.get("model", s))
    m = CBraModSleep(num_classes=5, ch_num=1, pretrained_path=None, dropout=0.1).to(device)
    msg = m.load_state_dict(sd, strict=False)
    return m, msg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts_glob", default="/root/autodl-tmp/runs/baselines_old/cbramod/seed*/best_model.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--n_noise", type=int, default=4)
    ap.add_argument("--noise_std", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts = sorted(glob.glob(args.ckpts_glob))
    print(f"[config] cbramod ckpts={len(ckpts)} n_noise={args.n_noise} std={args.noise_std}", flush=True)
    models = []
    for c in ckpts:
        m, msg = build_cbramod_from_ckpt(c, device)
        if len(msg.missing_keys) + len(msg.unexpected_keys) > 0:
            print(f"  [{os.path.basename(os.path.dirname(c))}] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
        m.eval()
        models.append(m)

    ds = SHHSCBraMod(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[data] N={len(ds)}", flush=True)

    torch.manual_seed(42)
    ps, ys = [], []
    for bi, (x, y) in enumerate(dl):
        x = x.to(device).float()
        inputs = [x]
        if args.n_noise > 0:
            std = x.reshape(x.shape[0], -1).std(dim=-1, keepdim=True).clamp_min(1e-6)
            shape_view = [x.shape[0]] + [1] * (x.dim() - 1)
            std_b = std.view(*shape_view)
            for _ in range(args.n_noise):
                inputs.append(x + torch.randn_like(x) * std_b * args.noise_std)
        all_logp = []
        for x_aug in inputs:
            for m in models:
                logits = m(x_aug)
                logp = torch.log_softmax(logits, dim=-1)
                all_logp.append(logp)
        stacked = torch.stack(all_logp, dim=0)
        agg = stacked.median(dim=0).values
        ps.append(agg.argmax(-1).cpu().numpy())
        ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
        if bi % 50 == 0:
            print(f"  batch {bi}/{len(dl)}", flush=True)

    p = np.concatenate(ps); yarr = np.concatenate(ys)
    f1 = float(f1_score(yarr, p, average="macro"))
    ba = float(balanced_accuracy_score(yarr, p))
    kp = float(cohen_kappa_score(yarr, p))
    per = f1_score(yarr, p, average=None, labels=[0,1,2,3,4]).tolist()
    print(f"[result] cbramod F1={f1:.4f} bal={ba:.4f} kappa={kp:.4f} per={[round(v,3) for v in per]}", flush=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f_:
        json.dump({
            "model": "cbramod", "macro_f1": f1, "bal_acc": ba, "kappa": kp,
            "per_class": per,
            "config": {**vars(args), "n_ckpts": len(ckpts)},
        }, f_, indent=2)


if __name__ == "__main__":
    main()
