"""Quick SHHS zs eval with per-class F1 output."""
import sys, os, glob, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1


class SHHSAll(Dataset):
    def __init__(self, shhs_dir, eeg_ch=0):
        self.files = sorted(glob.glob(os.path.join(shhs_dir, "shhs1-*.npz")))
        self.eeg_ch = eeg_ch
        self.index = []
        for fi, f in enumerate(self.files):
            with np.load(f, mmap_mode="r") as z:
                self.index.extend([(fi, i) for i in range(z["y"].shape[0])])
        self._cache = {}
    def __len__(self): return len(self.index)
    def _load(self, fi):
        if fi not in self._cache:
            if len(self._cache) >= 4: self._cache.pop(next(iter(self._cache)))
            with np.load(self.files[fi]) as z:
                self._cache[fi] = {"x": z["x"][:], "y": z["y"][:]}
        return self._cache[fi]
    def __getitem__(self, idx):
        fi, si = self.index[idx]
        d = self._load(fi)
        x = d["x"][si, self.eeg_ch, :].astype(np.float32)
        x = resample_poly(x, 100, 125)
        if x.shape[0] != 3000:
            x = (x[:3000] if x.shape[0] > 3000 else np.pad(x, (0, 3000 - x.shape[0])))
        return torch.from_numpy(x[None, :].astype(np.float32)), int(d["y"][si])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
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
    ds = SHHSAll(args.shhs_dir)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    preds, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            _, _, _, p5, _ = m(x)
            preds.append(p5.argmax(-1).cpu().numpy()); ys.append(np.asarray(y))
    p = np.concatenate(preds); t = np.concatenate(ys)
    pf = f1_score(t, p, average=None, labels=list(range(5)))
    print(f"macro_f1={f1_score(t,p,average='macro'):.4f} bal_acc={balanced_accuracy_score(t,p):.4f} kappa={cohen_kappa_score(t,p):.4f}")
    print(f"per_f1=[W={pf[0]:.3f},N1={pf[1]:.3f},N2={pf[2]:.3f},N3={pf[3]:.3f},REM={pf[4]:.3f}]")

if __name__ == "__main__":
    main()
