"""PhyG few-shot fine-tuning on MASS or ISRUC.

For ISRUC: subjects are dict keys in segmented_supX.npz. Subject pool = union across sup0/1/2.
For MASS: subjects = npz filenames.

Protocol:
  - Sample k support subjects (fixed by --fs_seed)
  - Fine-tune all params for --epochs
  - Eval macro-F1 on remaining (query) subjects
"""
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


# ----- MASS dataset -----
class MASSSubjectDataset(Dataset):
    def __init__(self, mass_dir, subjects):
        files = sorted(glob.glob(os.path.join(mass_dir, "*.npz")))
        keep = set(subjects)
        self.files = [f for f in files if os.path.basename(f).replace(".npz", "") in keep]
        self.index = []
        for fi, f in enumerate(self.files):
            with np.load(f, mmap_mode="r") as z:
                n = z["labels"].shape[0]
            self.index.extend([(fi, i) for i in range(n)])
        self._cache = {}
    def _load(self, fi):
        if fi not in self._cache:
            if len(self._cache) >= 3: self._cache.pop(next(iter(self._cache)))
            with np.load(self.files[fi]) as z:
                # downsample 256→100Hz once on load
                x = z["data"][:, 0, :].astype(np.float32)  # (N, 7680)
                x = resample_poly(x, 100, 256, axis=-1)  # (N, 3000)
                if x.shape[1] != 3000:
                    x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
                self._cache[fi] = {"x": x, "y": z["labels"][:]}
        return self._cache[fi]
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        fi, si = self.index[idx]
        d = self._load(fi)
        return torch.from_numpy(d["x"][si][None, :].astype(np.float32)), int(d["y"][si])


def list_mass_subjects(mass_dir):
    return [os.path.basename(f).replace(".npz", "") for f in sorted(glob.glob(os.path.join(mass_dir, "*.npz")))]


# ----- ISRUC dataset -----
class ISRUCSubjectDataset(Dataset):
    def __init__(self, isruc_dir, subject_keys):
        # subject_keys: list of (sub_subset, sid) tuples
        self.subject_keys = subject_keys
        self.cache = {}
        # build index
        self.entries = []
        for sub, sid in subject_keys:
            seg_p = os.path.join(isruc_dir, f"isruc_segmented_{sub}.npz")
            lab_p = os.path.join(isruc_dir, f"isruc_labels_{sub}.npz")
            z_seg = np.load(seg_p, allow_pickle=True)
            z_lab = np.load(lab_p, allow_pickle=True)
            labels = z_lab["labels"]
            sids = sorted(z_seg.files)
            offset = 0
            for s2 in sids:
                d = z_seg[s2].item()
                n = len(d["edf_data"])
                if s2 == sid:
                    # load and cache this subject
                    xs = np.stack([d["edf_data"][i][1].astype(np.float32) for i in range(n)], axis=0) * 1e-5
                    xs = resample_poly(xs, 100, 200, axis=-1)
                    if xs.shape[1] != 3000:
                        xs = xs[:, :3000] if xs.shape[1] > 3000 else np.pad(xs, ((0,0),(0,3000-xs.shape[1])))
                    ys = []
                    for i in range(n):
                        v = int(str(labels[offset+i]).strip())
                        ys.append(4 if v == 5 else v)
                    self.cache[(sub, sid)] = {"x": xs, "y": np.array(ys, dtype=np.int64)}
                    for i in range(n):
                        self.entries.append(((sub, sid), i))
                offset += n
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        key, i = self.entries[idx]
        d = self.cache[key]
        return torch.from_numpy(d["x"][i][None, :].astype(np.float32)), int(d["y"][i])


def list_isruc_subjects(isruc_dir):
    out = []
    for sub in ["sup0", "sup1", "sup2"]:
        seg_p = os.path.join(isruc_dir, f"isruc_segmented_{sub}.npz")
        if not os.path.exists(seg_p): continue
        z_seg = np.load(seg_p, allow_pickle=True)
        for sid in sorted(z_seg.files):
            out.append((sub, sid))
    return out


def load_model(ckpt, device):
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(z_dim=int(a.get("z_dim",256)), blocks=int(a.get("blocks",6)),
        out_ch=int(a.get("out_ch",32)), head_hidden=int(a.get("head_hidden",256)),
        K1=int(a.get("K1",96)), tau=float(a.get("tau",3.5))).to(device)
    if hasattr(m, "l1") and hasattr(m.l1, "use_pg"): m.l1.use_pg = bool(a.get("use_pg", True))
    m.load_state_dict(s["state_dict"], strict=True)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        _, _, _, p5, _ = model(x)
        ps.append(p5.argmax(-1).cpu().numpy()); ys.append(np.asarray(y))
    p = np.concatenate(ps); t = np.concatenate(ys)
    pf = f1_score(t, p, average=None, labels=list(range(5)))
    return {
        "macro_f1": float(f1_score(t, p, average="macro")),
        "bal_acc": float(balanced_accuracy_score(t, p)),
        "kappa": float(cohen_kappa_score(t, p)),
        "per_f1": [float(x) for x in pf],
    }


def fine_tune(model, loader, epochs, lr, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True).long()
            _, _, _, p5, _ = model(x)
            loss = F.nll_loss(p5.clamp_min(1e-8).log(), y)
            opt.zero_grad(); loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", choices=["mass", "isruc"], required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--fs_seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    device = "cuda"
    random.seed(args.fs_seed); np.random.seed(args.fs_seed); torch.manual_seed(args.fs_seed); torch.cuda.manual_seed_all(args.fs_seed)

    if args.target == "mass":
        subjects = list_mass_subjects(args.target_dir)
    else:
        subjects = list_isruc_subjects(args.target_dir)
    rng = random.Random(args.fs_seed)
    pool = list(subjects); rng.shuffle(pool)
    support = pool[:args.k]; query = pool[args.k:]
    print(f"[fewshot] target={args.target} k={args.k} fs_seed={args.fs_seed} support={len(support)} query={len(query)}", flush=True)

    model = load_model(args.ckpt, device)

    DSCls = MASSSubjectDataset if args.target == "mass" else ISRUCSubjectDataset
    qry_ds = DSCls(args.target_dir, query)
    qry_ld = DataLoader(qry_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    zs = evaluate(model, qry_ld, device)
    print(f"[zs] {zs}", flush=True)

    sup_ds = DSCls(args.target_dir, support)
    sup_ld = DataLoader(sup_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    fine_tune(model, sup_ld, args.epochs, args.lr, device)

    fs = evaluate(model, qry_ld, device)
    print(f"[fs] {fs}", flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "ckpt": args.ckpt, "target": args.target, "k": args.k, "fs_seed": args.fs_seed,
            "n_support_subjects": len(support), "n_query_subjects": len(query),
            "zero_shot": zs, "few_shot": fs,
        }, f, indent=2)
    print(f"[DONE] wrote {args.out_json}")


if __name__ == "__main__":
    main()
