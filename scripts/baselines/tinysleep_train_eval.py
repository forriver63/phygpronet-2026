"""TinySleepNet 5-fold train + zero-shot eval on SHHS / ISRUC.

Trains on Sleep-EDF (sleepedf_tinysleep_npz, 78 subjects, 5-fold subject split),
then evaluates each fold ckpt zero-shot on SHHS and ISRUC.
"""
import os, sys, glob, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
from scipy.signal import resample_poly

sys.path.insert(0, "/root/autodl-tmp/code/Tinysleepnet-pytorch-main")
from network import TinySleepNet

SLEEPEDF_DIR = "/root/autodl-tmp/data/sleepedf_tinysleep_npz"
SHHS_DIR = "/root/autodl-tmp/data/shhs_npz"
ISRUC_DIR = "/root/autodl-tmp/data/isruc_npz"
OUT_ROOT = "/root/autodl-tmp/runs"

CFG = {
    "n_epochs": 100,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9, "adam_beta_2": 0.999, "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "n_rnn_units": 128, "sampling_rate": 100.0,
    "input_size": 3000, "n_classes": 5,
    "l2_weight_decay": 1e-3,
    "seq_length": 20,
    "batch_size": 15,
    "class_weights": [1., 1.5, 1., 1., 1.],
}


# ---- Sleep-EDF dataset (epoch-shuffled, since we'll use seq_length=1 for simplicity) ----
class SleepEDFEpochs(Dataset):
    """Returns one 30s epoch at a time."""
    def __init__(self, files):
        self.xs, self.ys = [], []
        for f in files:
            with np.load(f) as z:
                x = z["x"].astype(np.float32)  # (n, 1, 3000)
                y = z["y"].astype(np.int64)
            self.xs.append(x); self.ys.append(y)
        self.x = np.concatenate(self.xs, 0)
        self.y = np.concatenate(self.ys, 0)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.x[i]), int(self.y[i])


# ---- SHHS / ISRUC datasets (epoch-level, with resample) ----
class SHHSEpochs(Dataset):
    def __init__(self, files, eeg_ch=0):
        self.files = files
        self.eeg_ch = eeg_ch
        self.index = []
        for fi, f in enumerate(files):
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
        x = d["x"][si, self.eeg_ch, :]
        x = resample_poly(x, 100, 125).astype(np.float32)
        if x.shape[0] != 3000:
            x = x[:3000] if x.shape[0] > 3000 else np.pad(x, (0, 3000 - x.shape[0]))
        return torch.from_numpy(x[None, :]), int(d["y"][si])


class ISRUCEpochs(Dataset):
    def __init__(self, root, eeg_ch=1, unit_scale=1e-5):
        self.eeg_ch = eeg_ch
        self.unit_scale = unit_scale
        self.entries = []
        self._seg_cache = {}
        self._cur = {}
        for sub in ["sup0", "sup1", "sup2"]:
            seg = os.path.join(root, f"isruc_segmented_{sub}.npz")
            lab = os.path.join(root, f"isruc_labels_{sub}.npz")
            if not (os.path.exists(seg) and os.path.exists(lab)):
                continue
            z_seg = np.load(seg, allow_pickle=True)
            z_lab = np.load(lab, allow_pickle=True)
            labels = z_lab["labels"]
            sids = sorted(z_seg.files)  # str-sort
            offset = 0
            for sid in sids:
                d = z_seg[sid].item()
                n = len(d["edf_data"])
                for i in range(n):
                    raw = labels[offset + i]
                    v = int(str(raw).strip())
                    y = 4 if v == 5 else v
                    self.entries.append((sub, sid, i, y))
                offset += n
            self._seg_cache[sub] = z_seg
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        sub, sid, i, y = self.entries[idx]
        key = (sub, sid)
        if key not in self._cur:
            self._cur.clear()
            self._cur[key] = self._seg_cache[sub][sid].item()["edf_data"]
        epoch = self._cur[key][i]
        x = np.asarray(epoch[self.eeg_ch], dtype=np.float32) * self.unit_scale
        x = resample_poly(x, 100, 200).astype(np.float32)
        if x.shape[0] != 3000:
            x = x[:3000] if x.shape[0] > 3000 else np.pad(x, (0, 3000 - x.shape[0]))
        return torch.from_numpy(x[None, :]), int(y)


# ---- Training (epoch-level, no LSTM seq context for simplicity → seq_length=1) ----
def build_model(device, seq_length=1, batch_size=128):
    cfg = dict(CFG); cfg["seq_length"] = seq_length; cfg["batch_size"] = batch_size
    net = TinySleepNet(cfg, N_channels=1).to(device)
    return net, cfg


def train_one_fold(train_files, valid_files, test_files, out_dir, device, n_epochs=100, fold_id=0, log=print):
    os.makedirs(out_dir, exist_ok=True)
    BS = 128
    SEQ = 1  # treat each epoch independently — simpler + avoids weight collapse from weighted CE
    net, cfg = build_model(device, seq_length=SEQ, batch_size=BS)

    tr_ds = SleepEDFEpochs(train_files)
    va_ds = SleepEDFEpochs(valid_files) if valid_files else None
    te_ds = SleepEDFEpochs(test_files)

    # Compute class weights from training set (tame: sqrt-scaled, clipped tight)
    bincnt = np.bincount(tr_ds.y, minlength=5).astype(np.float32)
    cw = np.sqrt(bincnt.sum() / (5.0 * np.maximum(bincnt, 1))).astype(np.float32)
    cw = np.clip(cw, 0.8, 2.5)
    log(f"fold{fold_id} class counts: {bincnt.tolist()}  weights: {cw.tolist()}")

    tr_dl = DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=BS, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=BS, shuffle=False, num_workers=2, drop_last=False, pin_memory=True) if va_ds else None

    cw_t = torch.from_numpy(cw).to(device)
    ce = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-5, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_va_f1 = -1.0
    best_path = os.path.join(out_dir, "best_model.pt")
    t0 = time.time()
    for ep in range(n_epochs):
        net.train()
        tr_loss = 0; tr_n = 0
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            bs = x.shape[0]
            state = (torch.zeros(1, bs, 128, device=device), torch.zeros(1, bs, 128, device=device))
            # network forwards (bs*seq, 1, 3000) with seq=1 → just (bs,1,3000) works since seq_length=1 in cfg
            net.config["batch_size"] = bs  # adjust on-the-fly
            net.config["seq_length"] = 1
            out, _ = net(x, state)
            loss = ce(out, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["clip_grad_value"])
            opt.step()
            tr_loss += loss.item() * bs; tr_n += bs
        sched.step()

        # validate
        net.eval()
        with torch.no_grad():
            def _eval(dl):
                ps, ys = [], []
                for x, y in dl:
                    x = x.to(device)
                    bs = x.shape[0]
                    state = (torch.zeros(1, bs, 128, device=device), torch.zeros(1, bs, 128, device=device))
                    net.config["batch_size"] = bs; net.config["seq_length"] = 1
                    out, _ = net(x, state)
                    ps.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
                ps = np.concatenate(ps); ys = np.concatenate(ys)
                return f1_score(ys, ps, average="macro", zero_division=0), ps, ys
            va_f1 = -1
            if va_dl is not None:
                va_f1, _, _ = _eval(va_dl)
            te_f1, te_p, te_y = _eval(te_dl)
        elapsed = time.time() - t0
        log(f"fold{fold_id} ep{ep+1}/{n_epochs} trL={tr_loss/tr_n:.4f} vaF1={va_f1:.4f} teF1={te_f1:.4f} ({elapsed:.0f}s)")
        score = va_f1 if va_dl is not None else te_f1
        if score > best_va_f1:
            best_va_f1 = score
            torch.save(net.state_dict(), best_path)

    # Final test using best ckpt
    net.load_state_dict(torch.load(best_path))
    net.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in te_dl:
            x = x.to(device); bs = x.shape[0]
            state = (torch.zeros(1, bs, 128, device=device), torch.zeros(1, bs, 128, device=device))
            net.config["batch_size"] = bs; net.config["seq_length"] = 1
            out, _ = net(x, state)
            ps.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
    ps = np.concatenate(ps); ys = np.concatenate(ys)
    return {
        "fold": fold_id,
        "best_val_f1": float(best_va_f1),
        "test_macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "test_per_class_f1": f1_score(ys, ps, average=None, labels=[0,1,2,3,4], zero_division=0).tolist(),
        "test_acc": float((ps == ys).mean()),
        "test_kappa": float(cohen_kappa_score(ys, ps)),
        "test_bal_acc": float(balanced_accuracy_score(ys, ps)),
        "n_test": int(len(ys)),
        "ckpt": best_path,
        "train_time_s": time.time() - t0,
    }


def eval_zs(ckpt_path, dataset, device, batch_size=128, max_samples=None):
    net, cfg = build_model(device, seq_length=1, batch_size=batch_size)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    if max_samples is not None and len(dataset) > max_samples:
        # take stratified-ish subsample by step
        idx = np.linspace(0, len(dataset) - 1, max_samples).astype(int)
        dataset = torch.utils.data.Subset(dataset, idx.tolist())
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device); bs = x.shape[0]
            state = (torch.zeros(1, bs, 128, device=device), torch.zeros(1, bs, 128, device=device))
            net.config["batch_size"] = bs; net.config["seq_length"] = 1
            out, _ = net(x, state)
            ps.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
    ps = np.concatenate(ps); ys = np.concatenate(ys)
    return {
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "per_class_f1": f1_score(ys, ps, average=None, labels=[0,1,2,3,4], zero_division=0).tolist(),
        "acc": float((ps == ys).mean()),
        "kappa": float(cohen_kappa_score(ys, ps)),
        "bal_acc": float(balanced_accuracy_score(ys, ps)),
        "n": int(len(ys)),
        "pred_dist": np.bincount(ps, minlength=5).tolist(),
        "true_dist": np.bincount(ys, minlength=5).tolist(),
    }


def subject_id_from_file(p):
    base = os.path.basename(p)
    # SC4xxXY.npz / ST7xxXY.npz — use first 5 chars to identify subject (e.g. SC401)
    return base[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--mode", type=str, default="train_and_eval", choices=["train", "eval", "train_and_eval"])
    ap.add_argument("--out", type=str, default=f"{OUT_ROOT}/tinysleep_5fold_v2")
    ap.add_argument("--shhs_max", type=int, default=50000)  # cap for speed
    ap.add_argument("--isruc_max", type=int, default=0)  # 0=no cap
    args = ap.parse_args()

    folds_to_run = [int(x) for x in args.folds.split(",")]
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    # ---- Build 5-fold split over subjects ----
    files = sorted(glob.glob(os.path.join(SLEEPEDF_DIR, "*.npz")))
    subj_map = {}
    for f in files:
        sid = subject_id_from_file(f)
        subj_map.setdefault(sid, []).append(f)
    subjects = sorted(subj_map.keys())
    print(f"Found {len(subjects)} subjects, {len(files)} files")
    rng = np.random.RandomState(42)
    perm = rng.permutation(subjects)
    fold_subjects = np.array_split(perm, 5)
    for i, fs in enumerate(fold_subjects):
        print(f"  Fold {i}: {len(fs)} test subjects")

    # ---- Train each fold ----
    train_results = {}
    for fold in folds_to_run:
        test_subj = set(fold_subjects[fold].tolist())
        train_subj = [s for s in subjects if s not in test_subj]
        # hold out 10% as valid
        n_va = max(1, round(len(train_subj) * 0.10))
        rng2 = np.random.RandomState(100 + fold)
        valid_subj = set(rng2.choice(train_subj, size=n_va, replace=False).tolist())
        train_only = [s for s in train_subj if s not in valid_subj]

        tr_files = [f for s in train_only for f in subj_map[s]]
        va_files = [f for s in valid_subj for f in subj_map[s]]
        te_files = [f for s in test_subj for f in subj_map[s]]
        fold_out = os.path.join(args.out, f"fold{fold}")
        print(f"\n===== FOLD {fold}: train={len(tr_files)} valid={len(va_files)} test={len(te_files)} files =====")
        if args.mode in ("train", "train_and_eval"):
            res = train_one_fold(tr_files, va_files, te_files, fold_out, device,
                                 n_epochs=args.epochs, fold_id=fold)
            train_results[fold] = res
            with open(os.path.join(fold_out, "train_metrics.json"), "w") as fh:
                json.dump(res, fh, indent=2)
            print(f"FOLD {fold} TEST macro F1 = {res['test_macro_f1']:.4f}, per-class {res['test_per_class_f1']}")
        else:
            with open(os.path.join(fold_out, "train_metrics.json")) as fh:
                train_results[fold] = json.load(fh)

    # ---- Zero-shot eval ----
    if args.mode in ("eval", "train_and_eval"):
        print("\n===== Building ZS datasets =====")
        shhs_files = sorted(glob.glob(os.path.join(SHHS_DIR, "shhs1-*.npz")))
        print(f"SHHS: {len(shhs_files)} files")
        # only use first ~80 files for speed
        shhs_files = shhs_files[:80]
        print(f"SHHS reduced to {len(shhs_files)} files (for speed)")
        shhs_ds = SHHSEpochs(shhs_files, eeg_ch=0)
        print(f"SHHS dataset n={len(shhs_ds)}")
        isruc_ds = ISRUCEpochs(ISRUC_DIR, eeg_ch=1, unit_scale=1e-5)
        print(f"ISRUC dataset n={len(isruc_ds)}")

        zs_results = {"shhs": {}, "isruc": {}}
        for fold in folds_to_run:
            ckpt = os.path.join(args.out, f"fold{fold}", "best_model.pt")
            if not os.path.exists(ckpt):
                print(f"SKIP fold {fold} (no ckpt)"); continue
            print(f"\n--- Fold {fold} ZS eval ---")
            r_shhs = eval_zs(ckpt, shhs_ds, device, max_samples=args.shhs_max)
            r_isr = eval_zs(ckpt, isruc_ds, device, max_samples=args.isruc_max or None)
            print(f"  SHHS  macro F1={r_shhs['macro_f1']:.4f}  per-class={r_shhs['per_class_f1']}")
            print(f"        pred dist={r_shhs['pred_dist']}  true dist={r_shhs['true_dist']}")
            print(f"  ISRUC macro F1={r_isr['macro_f1']:.4f}  per-class={r_isr['per_class_f1']}")
            print(f"        pred dist={r_isr['pred_dist']}  true dist={r_isr['true_dist']}")
            zs_results["shhs"][f"fold{fold}"] = r_shhs
            zs_results["isruc"][f"fold{fold}"] = r_isr
            out_dir = f"{OUT_ROOT}/zs_tinysleep_fold{fold}_shhs"
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
                json.dump(r_shhs, fh, indent=2)
            out_dir = f"{OUT_ROOT}/zs_tinysleep_fold{fold}_isruc"
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
                json.dump(r_isr, fh, indent=2)

        # summary
        summary = {"in_domain": {}, "zs": {}, "train": {}}
        for f in folds_to_run:
            if f in train_results:
                summary["in_domain"][f"fold{f}"] = {
                    "macro_f1": train_results[f]["test_macro_f1"],
                    "per_class_f1": train_results[f]["test_per_class_f1"],
                    "acc": train_results[f]["test_acc"],
                    "kappa": train_results[f]["test_kappa"],
                }
                summary["train"][f"fold{f}"] = {"train_time_s": train_results[f]["train_time_s"]}
        summary["zs"] = zs_results
        with open(os.path.join(args.out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to {args.out}/summary.json")


if __name__ == "__main__":
    main()
