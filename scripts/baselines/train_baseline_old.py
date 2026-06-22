"""Generic baseline trainer under PhyG OLD protocol.

Protocol:
- Sleep-EDF-78, subject-wise random 80/10/10 split per seed
- Input (B, 1, 3000) at 100Hz, 5 classes
- AdamW lr=3e-4, wd=1e-4, batch=128, epochs=20
- No scheduler, no class weights, no label smoothing
- Monitor val macro-F1, save best ckpt
- Eval: Sleep-EDF in-domain test F1 + SHHS zero-shot
"""
import os, sys, json, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SleepEDFEpochs(Dataset):
    """Loads (B, 1, 3000) tensors from a list of npz files."""
    def __init__(self, files):
        self.xs, self.ys = [], []
        for f in files:
            with np.load(f) as z:
                x = z["x"].astype(np.float32)
                y = z["y"].astype(np.int64)
            self.xs.append(x); self.ys.append(y)
        self.x = np.concatenate(self.xs, 0)
        self.y = np.concatenate(self.ys, 0)
        if self.x.ndim == 2:
            self.x = self.x[:, None, :]
        elif self.x.ndim == 3 and self.x.shape[1] != 1:
            if self.x.shape[-1] == 3000:
                pass
            else:
                self.x = self.x[:, :1, :]
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.x[i]), int(self.y[i])


class SHHSEpochs(Dataset):
    """Lazy SHHS, 125Hz→100Hz resample, ch0."""
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
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
        x = resample_poly(x, 100, 125).astype(np.float32)
        if x.shape[0] != 3000:
            x = x[:3000] if x.shape[0] > 3000 else np.pad(x, (0, 3000-x.shape[0]))
        return torch.from_numpy(x[None, :]), int(d["y"][si])


def random_subject_split(data_dir, seed, val_frac=0.1, test_frac=0.1):
    """Subject-wise random 80/10/10 split, grouping by first 5 chars of filename."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    subj_map = {}
    for f in files:
        sid = os.path.basename(f)[:5]
        subj_map.setdefault(sid, []).append(f)
    subjects = sorted(subj_map.keys())
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    n_test = max(1, int(round(len(subjects) * test_frac)))
    n_val  = max(1, int(round(len(subjects) * val_frac)))
    test_subj = perm[:n_test].tolist()
    val_subj  = perm[n_test:n_test+n_val].tolist()
    train_subj = perm[n_test+n_val:].tolist()
    tr_files = [f for s in train_subj for f in subj_map[s]]
    va_files = [f for s in val_subj for f in subj_map[s]]
    te_files = [f for s in test_subj for f in subj_map[s]]
    return tr_files, va_files, te_files, len(train_subj), len(val_subj), len(test_subj)


def load_model(model_name, device):
    """Return (model, forward_fn) where forward_fn(model, x) -> logits (B,5)."""
    if model_name == "attnsleep":
        sys.path.insert(0, "/root/autodl-tmp/code/毕业设计/AttnSleep-main/AttnSleep-main")
        from model.model import AttnSleep
        m = AttnSleep().to(device)
        def fwd(model, x): return model(x.float())
        return m, fwd
    if model_name == "tinysleep":
        sys.path.insert(0, "/root/autodl-tmp/code/Tinysleepnet-pytorch-main")
        from network import TinySleepNet
        cfg = {"n_rnn_units":128,"sampling_rate":100.0,"input_size":3000,"n_classes":5,
               "l2_weight_decay":1e-3,"seq_length":1,"batch_size":128,
               "class_weights":[1.,1.,1.,1.,1.],"adam_beta_1":0.9,"adam_beta_2":0.999,
               "adam_epsilon":1e-8,"clip_grad_value":5.0,"learning_rate":3e-4,
               "n_epochs":20}
        m = TinySleepNet(cfg, N_channels=1).to(device)
        def fwd(model, x):
            bs = x.shape[0]
            state = (torch.zeros(1, bs, 128, device=device), torch.zeros(1, bs, 128, device=device))
            model.config["batch_size"] = bs; model.config["seq_length"] = 1
            out, _ = model(x.float(), state)
            return out
        return m, fwd
    raise ValueError(model_name)


def evaluate(model, loader, fwd, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = fwd(model, x)
            ps.append(logits.argmax(-1).cpu().numpy())
            ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    p = np.concatenate(ps); y = np.concatenate(ys)
    return {
        "macro_f1": float(f1_score(y, p, average="macro")),
        "bal_acc": float(balanced_accuracy_score(y, p)),
        "kappa": float(cohen_kappa_score(y, p)),
        "per_f1": f1_score(y, p, average=None, labels=[0,1,2,3,4]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["attnsleep","tinysleep"])
    ap.add_argument("--data_dir", required=True, help="Sleep-EDF npz dir")
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_shhs", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    set_seed(args.seed)
    device = args.device

    tr_files, va_files, te_files, n_tr, n_va, n_te = random_subject_split(args.data_dir, args.seed)
    print(f"[split seed={args.seed}] train={n_tr} val={n_va} test={n_te} subjects")

    tr_ds = SleepEDFEpochs(tr_files); va_ds = SleepEDFEpochs(va_files); te_ds = SleepEDFEpochs(te_files)
    print(f"[samples] tr={len(tr_ds)} va={len(va_ds)} te={len(te_ds)}")
    for nm, ds in [("train",tr_ds),("val",va_ds),("test",te_ds)]:
        c = np.bincount(ds.y, minlength=5).tolist()
        print(f"  {nm} W/N1/N2/N3/REM: {c}")

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, fwd = load_model(args.model, device)
    print(f"[model {args.model}] params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True).long()
            logits = fwd(model, x)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        vm = evaluate(model, va_dl, fwd, device)
        print(f"[ep{ep}/{args.epochs}] loss={np.mean(losses):.4f} val_F1={vm['macro_f1']:.4f} val_BA={vm['bal_acc']:.4f}", flush=True)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val": vm})
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "val": vm, "epoch": ep}, best_path)

    # Final test
    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test_m = evaluate(model, te_dl, fwd, device)
    print(f"\n[TEST seed={args.seed}] F1={test_m['macro_f1']:.4f}", flush=True)

    out = {"seed": args.seed, "best_val_f1": float(best_f1), "test": test_m,
           "config": vars(args), "history": history, "train_time_s": time.time()-t0}

    # SHHS zero-shot
    if not args.skip_shhs and os.path.isdir(args.shhs_dir):
        shhs_ds = SHHSEpochs(args.shhs_dir)
        shhs_dl = DataLoader(shhs_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        shhs_m = evaluate(model, shhs_dl, fwd, device)
        print(f"[SHHS zs seed={args.seed}] F1={shhs_m['macro_f1']:.4f}", flush=True)
        out["shhs_zs"] = shhs_m

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
