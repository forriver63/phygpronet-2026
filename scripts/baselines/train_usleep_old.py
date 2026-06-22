#!/usr/bin/env python3
"""USleep trainer under PhyG OLD protocol.

- Sleep-EDF-78, subject-wise random 80/10/10 split per seed
- Sequence-level input (B, S=35, 1, T=3000)
- AdamW lr=3e-4, wd=1e-4, batch=128 (sequences), epochs=20
- No scheduler, no class weights, no label smoothing
- Monitor val per-epoch macro-F1, save best ckpt
- Eval: Sleep-EDF in-domain test F1 + SHHS zero-shot (per-epoch)
"""
import os, sys, json, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
from braindecode.models import USleep


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------- Sleep-EDF: memmap-backed subject dataset ----------
class SleepEDFMemmap:
    """Holds the global memmap arrays + per-subject (start, n) index."""
    def __init__(self, root):
        meta = json.load(open(os.path.join(root, "meta.json")))
        self.N = meta["N"]; self.L = meta["L"]
        self.X = np.memmap(os.path.join(root, "X30.dat"), dtype=np.float32, mode="r", shape=(self.N, self.L))
        self.y = np.memmap(os.path.join(root, "y.dat"), dtype=np.int64, mode="r", shape=(self.N,))
        # subject id by filename prefix (SC4001, SC4002, ...) - first 5 chars
        self.subj_files = {}  # sid_str -> list of (start, n)
        for e in meta["files"]:
            sid = e["file"][:5]  # 'SC4001'
            self.subj_files.setdefault(sid, []).append((e["start"], e["n"]))

    def all_subjects(self):
        return sorted(self.subj_files.keys())

    def subject_epochs(self, sid):
        """Return list of (start, n) blocks for this subject (one subject has 1-2 recordings)."""
        return self.subj_files[sid]


class EDFSeqDataset(Dataset):
    """Yield (S, 1, T) sequence + (S,) labels.

    For each recording block (one PSG file = one continuous block),
    slide windows of seq_len with stride=seq_len; if last window doesn't fit,
    add one final window aligned to the tail.
    """
    def __init__(self, mm: SleepEDFMemmap, subjects, seq_len=35):
        self.mm = mm
        self.seq_len = seq_len
        self.items = []  # (start_idx in global memmap, n_in_seq)
        for sid in subjects:
            for (s0, n) in mm.subject_epochs(sid):
                if n < 1:
                    continue
                # main strided windows
                k = 0
                while k + seq_len <= n:
                    self.items.append((s0 + k, seq_len))
                    k += seq_len
                # tail window aligned to end
                if k < n:
                    # if recording shorter than seq_len, pad start by replicating last
                    if n < seq_len:
                        # take the available n epochs; we'll pad in __getitem__
                        self.items.append((s0, n))
                    else:
                        tail_start = s0 + n - seq_len
                        self.items.append((tail_start, seq_len))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s, n = self.items[idx]
        if n == self.seq_len:
            x = np.array(self.mm.X[s:s+n], copy=True)  # (S, T)
            y = np.array(self.mm.y[s:s+n], copy=True)
        else:
            # short recording; pad at end with last epoch
            x = np.array(self.mm.X[s:s+n], copy=True)
            y = np.array(self.mm.y[s:s+n], copy=True)
            pad = self.seq_len - n
            x = np.concatenate([x, np.tile(x[-1:], (pad, 1))], axis=0)
            y = np.concatenate([y, np.full((pad,), -1, dtype=np.int64)], axis=0)
        # to (S, 1, T)
        x = x[:, None, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def random_subject_split(mm: SleepEDFMemmap, seed, val_frac=0.1, test_frac=0.1):
    subjects = mm.all_subjects()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(subjects)
    n_test = max(1, int(round(len(subjects) * test_frac)))
    n_val = max(1, int(round(len(subjects) * val_frac)))
    test_subj = perm[:n_test].tolist()
    val_subj = perm[n_test:n_test+n_val].tolist()
    train_subj = perm[n_test+n_val:].tolist()
    return train_subj, val_subj, test_subj


# ---------- SHHS sequence dataset ----------
class SHHSSeqDataset(Dataset):
    """Lazy SHHS; ch0; resample 125->100Hz; build seqs of seq_len epochs."""
    def __init__(self, root, seq_len=35):
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.seq_len = seq_len
        # Index per-file epoch counts (need y length only)
        self.items = []  # list of (file_idx, start_in_file, n)
        for fi, f in enumerate(self.files):
            with np.load(f, mmap_mode="r") as z:
                n_ep = z["y"].shape[0]
            k = 0
            while k + seq_len <= n_ep:
                self.items.append((fi, k, seq_len))
                k += seq_len
            if k < n_ep:
                if n_ep < seq_len:
                    self.items.append((fi, 0, n_ep))
                else:
                    self.items.append((fi, n_ep - seq_len, seq_len))
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
            x = z["x"][:, 0, :].astype(np.float32)  # (N, 3750)
            y = z["y"][:].astype(np.int64)
        # resample to 3000 along time
        x_rs = np.empty((x.shape[0], 3000), dtype=np.float32)
        for i in range(x.shape[0]):
            xi = resample_poly(x[i], 100, 125).astype(np.float32)
            if xi.shape[0] != 3000:
                xi = xi[:3000] if xi.shape[0] > 3000 else np.pad(xi, (0, 3000 - xi.shape[0]))
            x_rs[i] = xi
        self._cache[fi] = (x_rs, y)
        self._cache_order.append(fi)
        return x_rs, y

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fi, s, n = self.items[idx]
        x, y = self._load(fi)
        if n == self.seq_len:
            xs = x[s:s+n]
            ys = y[s:s+n]
        else:
            xs = x[s:s+n]
            ys = y[s:s+n]
            pad = self.seq_len - n
            xs = np.concatenate([xs, np.tile(xs[-1:], (pad, 1))], axis=0)
            ys = np.concatenate([ys, np.full((pad,), -1, dtype=np.int64)], axis=0)
        xs = xs[:, None, :].astype(np.float32)
        return torch.from_numpy(xs), torch.from_numpy(ys.astype(np.int64))


# ---------- eval ----------
def _set_bn_train(module):
    # Keep BN in train mode so it uses batch stats (USleep with brief training
    # produces degenerate running stats; eval-mode BN washes out the input).
    import torch.nn as _nn
    for m in module.modules():
        if isinstance(m, (_nn.BatchNorm1d, _nn.BatchNorm2d, _nn.BatchNorm3d, _nn.SyncBatchNorm)):
            m.train()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    _set_bn_train(model)
    all_y, all_p = [], []
    for x_seq, y_seq in loader:
        x_seq = x_seq.to(device, non_blocking=True)
        logits = model(x_seq)  # (B, K, S)
        pred = logits.argmax(dim=1).cpu().numpy()  # (B, S)
        y = y_seq.numpy()
        # mask out padding labels (-1)
        mask = y != -1
        all_y.append(y[mask])
        all_p.append(pred[mask])
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
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seq_len", type=int, default=35)
    ap.add_argument("--num_workers", type=int, default=2)
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

    tr_ds = EDFSeqDataset(mm, train_subj, seq_len=args.seq_len)
    va_ds = EDFSeqDataset(mm, val_subj, seq_len=args.seq_len)
    te_ds = EDFSeqDataset(mm, test_subj, seq_len=args.seq_len)
    print(f"[seqs] tr={len(tr_ds)} va={len(va_ds)} te={len(te_ds)}", flush=True)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = USleep(n_chans=1, sfreq=100.0, n_outputs=5,
                   n_times=3000, depth=12, with_skip_connection=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model USleep] params: {n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history = []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        t_ep = time.time()
        for x_seq, y_seq in tr_dl:
            x_seq = x_seq.to(device, non_blocking=True)
            y_seq = y_seq.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x_seq)  # (B, K, S)
                B, K, S = logits.shape
                loss = ce(logits.permute(0, 2, 1).reshape(B*S, K), y_seq.reshape(B*S))
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
        shhs_ds = SHHSSeqDataset(args.shhs_dir, seq_len=args.seq_len)
        print(f"[SHHS] seqs={len(shhs_ds)}", flush=True)
        shhs_dl = DataLoader(shhs_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        shhs_m = evaluate(model, shhs_dl, device)
        print(f"[SHHS zs seed={args.seed}] F1={shhs_m['macro_f1']:.4f} BA={shhs_m['bal_acc']:.4f}", flush=True)
        out["shhs_zs"] = shhs_m

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_dir}/results.json", flush=True)


if __name__ == "__main__":
    main()
