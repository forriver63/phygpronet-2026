"""PhyG SHHS few-shot — fast and minimal. Subject-level sample, full finetune."""
import os, sys, glob, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")
from src.model_inception_tree_protoL1 import InceptionTreeProtoL1


class SHHSSubj(Dataset):
    def __init__(self, shhs_dir, subjects):
        files = sorted(glob.glob(os.path.join(shhs_dir, "shhs1-*.npz")))
        keep = set(subjects)
        files = [f for f in files if os.path.basename(f).replace(".npz","") in keep]
        self.X, self.y = [], []
        for f in files:
            with np.load(f) as z:
                x = z["x"][:, 0, :].astype(np.float32)   # 125Hz, 3750
                x = resample_poly(x, 100, 125, axis=-1)   # 100Hz, 3000
                if x.shape[1] != 3000:
                    x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
                self.X.append(x); self.y.append(z["y"][:])
        self.X = np.concatenate(self.X)
        self.y = np.concatenate(self.y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.X[i][None,:].astype(np.float32)), int(self.y[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--fs_seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    device = "cuda"
    random.seed(args.fs_seed); np.random.seed(args.fs_seed); torch.manual_seed(args.fs_seed); torch.cuda.manual_seed_all(args.fs_seed)

    subjects = [os.path.basename(f).replace(".npz","") for f in sorted(glob.glob(os.path.join(args.shhs_dir, "shhs1-*.npz")))]
    rng = random.Random(args.fs_seed); rng.shuffle(subjects)
    support = subjects[:args.k]; query = subjects[args.k:]
    print(f"[fs] shhs k={args.k} fs={args.fs_seed} support={len(support)} query={len(query)}", flush=True)

    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(z_dim=int(a.get("z_dim",256)), blocks=int(a.get("blocks",6)),
        out_ch=int(a.get("out_ch",32)), head_hidden=int(a.get("head_hidden",256)),
        K1=int(a.get("K1",96)), tau=float(a.get("tau",3.5))).to(device)
    if hasattr(m, "l1") and hasattr(m.l1, "use_pg"): m.l1.use_pg = bool(a.get("use_pg", True))
    m.load_state_dict(s["state_dict"], strict=True)

    qry = SHHSSubj(args.shhs_dir, query)
    qry_ld = DataLoader(qry, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    def ev():
        m.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in qry_ld:
                x = x.to(device, non_blocking=True)
                _, _, _, p5, _ = m(x)
                ps.append(p5.argmax(-1).cpu().numpy()); ys.append(np.asarray(y))
        p = np.concatenate(ps); t = np.concatenate(ys)
        pf = f1_score(t, p, average=None, labels=list(range(5)))
        return {"macro_f1": float(f1_score(t, p, average="macro")), "bal_acc": float(balanced_accuracy_score(t, p)),
                "kappa": float(cohen_kappa_score(t, p)), "per_f1": [float(x) for x in pf]}

    zs = ev(); print(f"[zs] {zs}", flush=True)

    sup = SHHSSubj(args.shhs_dir, support)
    sup_ld = DataLoader(sup, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-5)
    m.train()
    for ep in range(args.epochs):
        for x, y in sup_ld:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True).long()
            _, _, _, p5, _ = m(x)
            loss = F.nll_loss(p5.clamp_min(1e-8).log(), y)
            opt.zero_grad(); loss.backward(); opt.step()
    fs = ev(); print(f"[fs] {fs}", flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fp:
        json.dump({"ckpt": args.ckpt, "target": "shhs", "k": args.k, "fs_seed": args.fs_seed,
                   "n_support_subjects": len(support), "n_query_subjects": len(query),
                   "zero_shot": zs, "few_shot": fs}, fp, indent=2)
    print(f"[DONE] {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
