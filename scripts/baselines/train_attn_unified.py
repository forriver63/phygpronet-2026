"""AttnSleep retrain under UNIFIED 5-fold subject-CV protocol.

Matches strong-baseline recipe (= same as TinySleepNet v2 / PhyG unified):
- subject 5-fold split loaded from JSON
- AdamW + CosineAnnealingLR
- 50 epochs, batch 128, lr 3e-4 (or per-arg)
- monitor val macro-F1, save best ckpt
- class weights (sqrt-scaled clipped) from TRAIN only
- label smoothing 0.05
"""
import os, sys, json, argparse, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/毕业设计/AttnSleep-main/AttnSleep-main")
from model.model import AttnSleep


class SleepEDFEpochs(Dataset):
    def __init__(self, files):
        self.xs, self.ys = [], []
        for f in files:
            with np.load(f) as z:
                x = z["x"].astype(np.float32)
                y = z["y"].astype(np.int64)
            self.xs.append(x); self.ys.append(y)
        self.x = np.concatenate(self.xs, 0)
        self.y = np.concatenate(self.ys, 0)
        # ensure shape (N, 1, 3000)
        if self.x.ndim == 2:
            self.x = self.x[:, None, :]
        elif self.x.ndim == 3 and self.x.shape[1] != 1:
            self.x = self.x.transpose(0, 2, 1) if self.x.shape[-1] != 3000 else self.x[:, :1, :]

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(self.y[i])


def subject_id_5char(path):
    return os.path.basename(path)[:5]


def fold_files(data_dir, split_json_path, fold_id):
    with open(split_json_path) as f:
        split = json.load(f)
    fold = split["folds"][str(fold_id)]
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    by_prefix = {}
    for f in files:
        by_prefix.setdefault(subject_id_5char(f), []).append(f)
    def col(subj_list):
        out = []
        for s in subj_list:
            out.extend(by_prefix.get(s, []))
        return sorted(out)
    return col(fold["train"]), col(fold["val"]), col(fold["test"])


def compute_class_weights(ys, mode="sqrt_clipped"):
    if mode == "none": return None
    bincnt = np.bincount(ys, minlength=5).astype(np.float32)
    cw = np.sqrt(bincnt.sum() / (5.0 * np.maximum(bincnt, 1))).astype(np.float32)
    return np.clip(cw, 0.8, 2.5)


def evaluate(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device).float())
            ps.append(out.argmax(-1).cpu().numpy())
            ys.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
    p = np.concatenate(ps); yy = np.concatenate(ys)
    return {
        "macro_f1": float(f1_score(yy, p, average="macro")),
        "bal_acc": float(balanced_accuracy_score(yy, p)),
        "kappa": float(cohen_kappa_score(yy, p)),
        "per_f1": f1_score(yy, p, average=None, labels=[0,1,2,3,4]).tolist(),
    }


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/autodl-tmp/data/sleepedf_attnsleep_npz")
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--fold_id", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--class_weights", choices=["none", "sqrt_clipped"], default="sqrt_clipped")
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_files, va_files, te_files = fold_files(args.data_dir, args.split_json, args.fold_id)
    print(f"[split] fold={args.fold_id}  files train={len(tr_files)} val={len(va_files)} test={len(te_files)}")

    tr_ds = SleepEDFEpochs(tr_files)
    va_ds = SleepEDFEpochs(va_files)
    te_ds = SleepEDFEpochs(te_files)

    # class dist
    for nm, ds in [("train", tr_ds), ("val", va_ds), ("test", te_ds)]:
        c = np.bincount(ds.y, minlength=5).tolist()
        print(f"   {nm} class dist W/N1/N2/N3/REM: {c}")

    cw = compute_class_weights(tr_ds.y, mode=args.class_weights)
    print(f"[class_weights={args.class_weights}] {cw}")
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device) if cw is not None else None

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, drop_last=True, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AttnSleep().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=args.label_smoothing)

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tr_dl:
            x = x.to(device).float(); y = y.to(device).long()
            out = model(x)
            loss = ce(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val_m = evaluate(model, va_dl, device)
        print(f"[ep{ep}] loss={np.mean(losses):.4f}  val_F1={val_m['macro_f1']:.4f}  "
              f"val_BalAcc={val_m['bal_acc']:.4f}  val_Kappa={val_m['kappa']:.4f}", flush=True)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val": val_m})

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "val": val_m, "epoch": ep}, best_path)

    # final test
    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test_m = evaluate(model, te_dl, device)
    print("\n=== TEST (best val macro-F1) ===")
    print(test_m)

    out = {"fold_id": args.fold_id, "best_val_f1": float(best_f1), "test": test_m,
           "config": vars(args), "history": history}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {args.out_dir}/results.json")


if __name__ == "__main__":
    main()
