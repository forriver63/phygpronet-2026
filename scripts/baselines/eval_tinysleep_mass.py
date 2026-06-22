"""Zero-shot eval of TinySleepNet (5 folds, trained on Sleep-EDF) on MASS-SS3."""
import os, sys, json, glob, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

sys.path.insert(0, "/root/autodl-tmp/code/Tinysleepnet-pytorch-main")
sys.path.insert(0, "/root/autodl-tmp/code/PhyGProNet-Repro")
sys.path.insert(0, "/root/autodl-tmp")

from network import TinySleepNet
from phygpronet.data.mass_npz import MASSNpzDataset

CFG = {
    "n_epochs": 100,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9, "adam_beta_2": 0.999, "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "n_rnn_units": 128, "sampling_rate": 100.0,
    "input_size": 3000, "n_classes": 5,
    "l2_weight_decay": 1e-3,
    "seq_length": 1,
    "batch_size": 128,
    "class_weights": [1., 1.5, 1., 1., 1.],
}

CKPT_ROOT = "/root/autodl-tmp/runs/tinysleep_5fold_v2"
OUT_ROOT  = "/root/autodl-tmp/runs"
MASS_DIR  = "/root/autodl-tmp/data/mass_npz"

def build_model(device, batch_size=128):
    cfg = dict(CFG); cfg["batch_size"] = batch_size; cfg["seq_length"] = 1
    net = TinySleepNet(cfg, N_channels=1).to(device)
    return net

def eval_one(ckpt, ds, device, bs=128):
    net = build_model(device, batch_size=bs)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)
    ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device); b = x.shape[0]
            state = (torch.zeros(1, b, 128, device=device), torch.zeros(1, b, 128, device=device))
            net.config["batch_size"] = b; net.config["seq_length"] = 1
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Building MASS dataset (eeg_ch=20, 256->100Hz)...")
    t0 = time.time()
    ds = MASSNpzDataset(MASS_DIR, eeg_ch=20)
    print(f"MASS n={len(ds)} epochs, n_subjects={len(ds.files)} (built in {time.time()-t0:.1f}s)")

    all_results = {}
    for fold in range(5):
        ckpt = os.path.join(CKPT_ROOT, f"fold{fold}", "best_model.pt")
        if not os.path.exists(ckpt):
            print(f"SKIP fold{fold}: ckpt missing"); continue
        print(f"\n--- Fold {fold} ZS MASS eval ---")
        t1 = time.time()
        r = eval_one(ckpt, ds, device, bs=128)
        print(f"  macro F1={r['macro_f1']:.4f}  bal_acc={r['bal_acc']:.4f}  kappa={r['kappa']:.4f}")
        print(f"  per-class F1: W={r['per_class_f1'][0]:.3f} N1={r['per_class_f1'][1]:.3f} N2={r['per_class_f1'][2]:.3f} N3={r['per_class_f1'][3]:.3f} REM={r['per_class_f1'][4]:.3f}")
        print(f"  pred dist={r['pred_dist']}  true dist={r['true_dist']}  ({time.time()-t1:.0f}s)")
        out_dir = os.path.join(OUT_ROOT, f"zs_tinysleep_fold{fold}_mass")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
            json.dump(r, fh, indent=2)
        all_results[f"fold{fold}"] = r

    # aggregate
    if all_results:
        macros = [r["macro_f1"] for r in all_results.values()]
        bals   = [r["bal_acc"] for r in all_results.values()]
        kappas = [r["kappa"] for r in all_results.values()]
        pcs    = np.array([r["per_class_f1"] for r in all_results.values()])  # (folds, 5)
        agg = {
            "n_folds": len(all_results),
            "macro_f1_mean": float(np.mean(macros)), "macro_f1_std": float(np.std(macros)),
            "bal_acc_mean": float(np.mean(bals)),   "bal_acc_std": float(np.std(bals)),
            "kappa_mean": float(np.mean(kappas)),   "kappa_std": float(np.std(kappas)),
            "per_class_f1_mean": pcs.mean(0).tolist(),
            "per_class_f1_std":  pcs.std(0).tolist(),
            "folds": all_results,
        }
        with open(os.path.join(OUT_ROOT, "zs_tinysleep_mass_summary.json"), "w") as fh:
            json.dump(agg, fh, indent=2)
        print("\n========== AGGREGATE ==========")
        print(f"Macro F1: {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
        print(f"Bal acc:  {agg['bal_acc_mean']:.4f} +/- {agg['bal_acc_std']:.4f}")
        print(f"Kappa:    {agg['kappa_mean']:.4f} +/- {agg['kappa_std']:.4f}")
        print(f"Per-class F1 mean: {[f'{v:.3f}' for v in agg['per_class_f1_mean']]}")
        print(f"Per-class F1 std:  {[f'{v:.3f}' for v in agg['per_class_f1_std']]}")

if __name__ == "__main__":
    main()
