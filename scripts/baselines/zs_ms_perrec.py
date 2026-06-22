"""SHHS zs with per-recording z-score (for multi-band ckpt)."""
import sys, os, json, argparse, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1


class SHHSPerRecZScore(Dataset):
    """Per-recording z-score normalized SHHS loader."""
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.index = []
        self.stats = {}   # fi -> (mu, sigma)
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
                x_full = z["x"][:, 0, :].astype(np.float32)  # (n, 3750) at 125Hz
                # Resample to 100Hz per epoch
                x_rs = resample_poly(x_full, 100, 125, axis=-1).astype(np.float32)
                # Compute per-recording stats from concat of all epochs
                flat = x_rs.reshape(-1)
                mu = float(flat.mean()); sigma = float(flat.std() + 1e-6)
                x_rs = (x_rs - mu) / sigma
                if x_rs.shape[-1] != 3000:
                    x_rs = x_rs[:, :3000] if x_rs.shape[-1] > 3000 else np.pad(x_rs, ((0,0),(0,3000-x_rs.shape[-1])))
                self._cache[fi] = {"x": x_rs, "y": z["y"][:].astype(np.int64), "mu": mu, "sigma": sigma}
        return self._cache[fi]

    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        fi, si = self.index[idx]
        d = self._load(fi)
        return torch.from_numpy(d["x"][si][None, :]), int(d["y"][si])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda"
    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    model = InceptionTreeProtoL1(
        z_dim=int(a.get("z_dim",256)),blocks=int(a.get("blocks",6)),out_ch=int(a.get("out_ch",32)),
        head_hidden=int(a.get("head_hidden",256)),K1=int(a.get("K1",96)),tau=float(a.get("tau",3.5)),
    ).to(dev)
    if hasattr(model,"l1") and hasattr(model.l1,"use_pg"):
        model.l1.use_pg = bool(a.get("use_pg", True))
    model.load_state_dict(s["state_dict"], strict=True); model.eval()
    print(f"[ckpt] mech_w={a.get('mech_w')} multi_band={a.get('multi_band_mech')} per_rec={a.get('per_recording_norm')}", flush=True)

    ds = SHHSPerRecZScore("/root/autodl-tmp/data/shhs_npz")
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    print(f"[data] N={len(ds)} (with per-recording z-score)", flush=True)
    ps, ys = [], []
    for x, y in dl:
        x = x.to(dev).float()
        _,_,_,p5,_ = model(x)
        ps.append(p5.argmax(-1).cpu().numpy())
        ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    p = np.concatenate(ps); y = np.concatenate(ys)
    f1 = float(f1_score(y, p, average="macro"))
    ba = float(balanced_accuracy_score(y, p))
    kp = float(cohen_kappa_score(y, p))
    per = f1_score(y, p, average=None, labels=[0,1,2,3,4]).tolist()
    print(f"[SHHS zs per-rec] F1={f1:.4f} bal={ba:.4f} kappa={kp:.4f}")
    print(f"  per: W={per[0]:.3f} N1={per[1]:.3f} N2={per[2]:.3f} N3={per[3]:.3f} REM={per[4]:.3f}")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f_:
        json.dump({"macro_f1": f1, "bal_acc": ba, "kappa": kp, "per_class": per,
                   "norm": "per_recording_zscore"}, f_, indent=2)


if __name__ == "__main__":
    main()
