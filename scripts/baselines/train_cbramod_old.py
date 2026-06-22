#!/usr/bin/env python3
"""CBraMod fine-tune under PhyG OLD protocol.

- Sleep-EDF-78, subject-wise random 80/10/10 split per seed (subject = filename[:5])
- Epoch-level input: (B, 1, 30, 200) = single 30s epoch resampled 100->200Hz
- AdamW lr=3e-4, wd=1e-4, batch=128, epochs=20
- No scheduler, no class weights, no label smoothing
- Monitor val macro-F1, save best ckpt
- Eval: Sleep-EDF test F1 + SHHS zero-shot F1
"""
import os, sys, json, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

CBRAMOD_DIR = "/root/autodl-tmp/code/baselines-code - clean/CBraMod_Code/CBraMod-main"
sys.path.insert(0, CBRAMOD_DIR)
from models.cbramod import CBraMod  # noqa: E402


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------- Sleep-EDF memmap ----------
class SleepEDFMemmap:
    def __init__(self, root):
        meta = json.load(open(os.path.join(root, "meta.json")))
        self.N = meta["N"]; self.L = meta["L"]
        self.X = np.memmap(os.path.join(root, "X30.dat"), dtype=np.float32, mode="r", shape=(self.N, self.L))
        self.y = np.memmap(os.path.join(root, "y.dat"), dtype=np.int64, mode="r", shape=(self.N,))
        self.subj_files = {}
        for e in meta["files"]:
            sid = e["file"][:5]  # 'SC400' groups SC4001/SC4002 (one subject, two nights)
            self.subj_files.setdefault(sid, []).append((e["start"], e["n"]))

    def all_subjects(self):
        return sorted(self.subj_files.keys())


class EDFEpochDataset(Dataset):
    """Per-epoch dataset: x (1, 30, 200), y int."""
    def __init__(self, mm: SleepEDFMemmap, subjects):
        self.mm = mm
        self.indices = []  # global indices into X
        for sid in subjects:
            for (s0, n) in mm.subj_files[sid]:
                self.indices.extend(range(s0, s0 + n))
        self.indices = np.array(self.indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        gi = int(self.indices[idx])
        x = np.array(self.mm.X[gi], copy=True)  # (3000,) @ 100Hz
        y = int(self.mm.y[gi])
        # resample 100 -> 200 Hz -> 6000 samples
        x6 = resample_poly(x, 200, 100).astype(np.float32)
        if x6.shape[0] != 6000:
            x6 = x6[:6000] if x6.shape[0] > 6000 else np.pad(x6, (0, 6000 - x6.shape[0]))
        # CBraMod pretrain unit: V -> 100uV (x10000)
        x6 = x6 * 10000.0
        # reshape to (1, 30, 200)
        x6 = x6.reshape(1, 30, 200)
        return torch.from_numpy(x6), torch.tensor(y, dtype=torch.long)


def random_subject_split(mm, seed, val_frac=0.1, test_frac=0.1):
    subjects = mm.all_subjects()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    n_test = max(1, int(round(len(subjects) * test_frac)))
    n_val = max(1, int(round(len(subjects) * val_frac)))
    test_subj = perm[:n_test].tolist()
    val_subj = perm[n_test:n_test + n_val].tolist()
    train_subj = perm[n_test + n_val:].tolist()
    return train_subj, val_subj, test_subj


# ---------- SHHS per-epoch ----------
class SHHSEpochDataset(Dataset):
    """SHHS ch0, resample 125->200Hz -> (1,30,200)."""
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.index = []  # (fi, k)
        for fi, f in enumerate(self.files):
            with np.load(f, mmap_mode="r") as z:
                n_ep = z["y"].shape[0]
            self.index.extend([(fi, k) for k in range(n_ep)])
        self._cache = {}
        self._cache_order = []
        self._cache_limit = 4

    def _load(self, fi):
        if fi in self._cache:
            return self._cache[fi]
        if len(self._cache) >= self._cache_limit:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        with np.load(self.files[fi]) as z:
            x = z["x"][:, 0, :].astype(np.float32)  # (N, 3750) @125Hz
            y = z["y"][:].astype(np.int64)
        # CBraMod pretrain unit: V -> 100uV (x10000)
        x = x * 10000.0
        # resample 125 -> 200 -> 6000
        x_rs = np.empty((x.shape[0], 6000), dtype=np.float32)
        for i in range(x.shape[0]):
            xi = resample_poly(x[i], 200, 125).astype(np.float32)
            if xi.shape[0] != 6000:
                xi = xi[:6000] if xi.shape[0] > 6000 else np.pad(xi, (0, 6000 - xi.shape[0]))
            x_rs[i] = xi
        self._cache[fi] = (x_rs, y)
        self._cache_order.append(fi)
        return x_rs, y

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, k = self.index[idx]
        x, y = self._load(fi)
        xi = x[k].reshape(1, 30, 200)
        return torch.from_numpy(xi.copy()), torch.tensor(int(y[k]), dtype=torch.long)


# ---------- model ----------
class CBraModSleep(nn.Module):
    """CBraMod backbone + classifier head for 5-class single-channel sleep staging."""
    def __init__(self, num_classes=5, ch_num=1, pretrained_path=None, dropout=0.1):
        super().__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8,
        )
        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[pretrained] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        self.backbone.proj_out = nn.Identity()
        self.ch_num = ch_num
        feat_dim = ch_num * 30 * 200
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, ch, 30, 200)
        B = x.size(0)
        feats = self.backbone(x)  # (B, ch, 30, 200)
        feats = feats.reshape(B, -1)
        h = self.head(feats)
        out = self.classifier(h)
        return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_y.append(y.numpy())
        all_p.append(pred)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    return {
        "macro_f1": float(f1_score(y, p, average="macro")),
        "bal_acc": float(balanced_accuracy_score(y, p)),
        "kappa": float(cohen_kappa_score(y, p)),
        "per_f1": f1_score(y, p, average=None, labels=[0, 1, 2, 3, 4]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf_root", default="/root/autodl-tmp/data/sleepedf_memmap_v2_mech")
    ap.add_argument("--shhs_dir", default="/root/autodl-tmp/data/shhs_npz")
    ap.add_argument("--pretrained", default=os.path.join(CBRAMOD_DIR, "pretrained_weights/pretrained_weights.pth"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--skip_shhs", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mm = SleepEDFMemmap(args.edf_root)
    train_subj, val_subj, test_subj = random_subject_split(mm, args.seed)
    print(f"[split seed={args.seed}] subj train={len(train_subj)} val={len(val_subj)} test={len(test_subj)}", flush=True)

    tr_ds = EDFEpochDataset(mm, train_subj)
    va_ds = EDFEpochDataset(mm, val_subj)
    te_ds = EDFEpochDataset(mm, test_subj)
    print(f"[epochs] tr={len(tr_ds)} va={len(va_ds)} te={len(te_ds)}", flush=True)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=(args.num_workers > 0))
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0))

    model = CBraModSleep(num_classes=5, ch_num=1, pretrained_path=args.pretrained, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model CBraMod] total params: {n_params/1e6:.2f}M  trainable: {n_train/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history = []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        t_ep = time.time()
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        vm = evaluate(model, va_dl, device)
        dt = time.time() - t_ep
        print(f"[ep{ep}/{args.epochs}] loss={np.mean(losses):.4f} val_F1={vm['macro_f1']:.4f} val_BA={vm['bal_acc']:.4f} ({dt:.1f}s)", flush=True)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val": vm, "time_s": dt})
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "val": vm, "epoch": ep}, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test_m = evaluate(model, te_dl, device)
    print(f"\n[TEST seed={args.seed}] F1={test_m['macro_f1']:.4f} BA={test_m['bal_acc']:.4f} kappa={test_m['kappa']:.4f}", flush=True)

    out = {"seed": args.seed, "best_val_f1": float(best_f1), "test": test_m,
           "config": vars(args), "history": history, "train_time_s": time.time() - t0}

    if not args.skip_shhs and os.path.isdir(args.shhs_dir):
        print("[SHHS] building dataset...", flush=True)
        shhs_ds = SHHSEpochDataset(args.shhs_dir)
        print(f"[SHHS] epochs={len(shhs_ds)}", flush=True)
        shhs_dl = DataLoader(shhs_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        shhs_m = evaluate(model, shhs_dl, device)
        print(f"[SHHS zs seed={args.seed}] F1={shhs_m['macro_f1']:.4f} BA={shhs_m['bal_acc']:.4f}", flush=True)
        out["shhs_zs"] = shhs_m

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
