"""Statistical analysis for paper:
- Paired t-test PhyG vs each baseline on Sleep-EDF F1 + SHHS zs (6 seeds)
- Confusion matrix figure (PhyG vs best baseline)
- N1 misclassification analysis (PhyG)
"""
import os, json, glob
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_seeds(model_name, ckpt_dir):
    """Load (test_f1, shhs_zs) for each seed from results.json."""
    tests, shhs = [], []
    for seed in [42, 43, 44, 45, 46, 47]:
        f = os.path.join(ckpt_dir, f"seed{seed}", "results.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        tests.append(r["test"]["macro_f1"])
        shhs.append(r.get("shhs_zs", {}).get("macro_f1", np.nan))
    return tests, shhs


def main():
    out_dir = "/root/autodl-tmp/runs/analysis_paper"
    os.makedirs(out_dir, exist_ok=True)

    # ====== Paired t-test: PhyG vs each baseline ======
    print("=" * 60); print("Paired t-test (6 seeds, Sleep-EDF test F1)")
    print("=" * 60)

    base = "/root/autodl-tmp/runs/baselines_old"
    models = {
        "phyg": "/root/autodl-tmp/runs/baselines_old/phyg_seed_results.json",  # placeholder
        "attnsleep": f"{base}/attnsleep",
        "tinysleep": f"{base}/tinysleep",
        "usleep": f"{base}/usleep",
        "deepsleep": f"{base}/deepsleep",
        "cbramod": f"{base}/cbramod",
        "labram": f"{base}/labram",
    }

    # PhyG 6 seeds from old protocol K96_t35 — already computed elsewhere
    phyg_tests = [0.7632, 0.7356, 0.7257, 0.7263, 0.7484, 0.7652]
    phyg_shhs  = [0.3889, 0.5326, 0.5264, 0.4404, 0.4883, 0.5423]

    rows = [{"model": "PhyG", "edf_mean": np.mean(phyg_tests), "edf_std": np.std(phyg_tests, ddof=1),
             "shhs_mean": np.mean(phyg_shhs), "shhs_std": np.std(phyg_shhs, ddof=1),
             "edf_p_vs_phyg": "—", "shhs_p_vs_phyg": "—"}]

    for name, d in models.items():
        if name == "phyg": continue
        tests, shhs = load_seeds(name, d)
        if not tests:
            print(f"[skip] {name} no seeds"); continue
        edf_p = stats.ttest_rel(phyg_tests, tests).pvalue if len(tests) == 6 else np.nan
        shhs_clean = [x for x in shhs if not np.isnan(x)]
        if len(shhs_clean) == 6:
            shhs_p = stats.ttest_rel(phyg_shhs, shhs_clean).pvalue
        else:
            shhs_p = np.nan
        rows.append({
            "model": name, "edf_mean": np.mean(tests), "edf_std": np.std(tests, ddof=1),
            "shhs_mean": np.mean(shhs_clean), "shhs_std": np.std(shhs_clean, ddof=1) if shhs_clean else np.nan,
            "edf_p_vs_phyg": float(edf_p), "shhs_p_vs_phyg": float(shhs_p) if not np.isnan(shhs_p) else "—",
        })
        print(f"{name:12s}: EDF mean={np.mean(tests):.4f}±{np.std(tests, ddof=1):.4f} p={edf_p:.3g}  "
              f"SHHS mean={np.mean(shhs_clean):.4f}±{np.std(shhs_clean, ddof=1) if shhs_clean else np.nan:.4f} p={shhs_p if not np.isnan(shhs_p) else 'nan':.3g}")

    with open(os.path.join(out_dir, "paired_ttest.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # ====== Confusion matrix: PhyG seed42 vs DeepSleep seed42 (SHHS) ======
    print("\n" + "=" * 60); print("Confusion matrices (SHHS zs, seed42)")
    print("=" * 60)

    def load_preds(npz_path):
        d = np.load(npz_path)
        # Common keys: preds/labels or pred/label
        keys = list(d.keys())
        if "preds" in keys: return d["preds"], d["labels"]
        if "pred" in keys: return d["pred"], d["label"]
        return None, None

    def conf_mat(preds, labels, n=5):
        cm = np.zeros((n, n), dtype=np.float64)
        for p, l in zip(preds, labels):
            cm[l, p] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-8)
        return cm, cm_norm

    cls = ["W", "N1", "N2", "N3", "REM"]

    # PhyG seed42 SHHS
    phyg_npz = "/root/autodl-tmp/runs/zs_K96_t35_seed42_shhs/shhs_preds.npz"
    deep_npz = "/root/autodl-tmp/runs/baselines_old/deepsleep/seed42/shhs_preds.npz"  # may not exist
    # Try multiple paths
    for path in [deep_npz, "/root/autodl-tmp/runs/baselines_old/deepsleep/seed42/results.json"]:
        if os.path.exists(path):
            print(f"  found {path}")

    if os.path.exists(phyg_npz):
        p_pred, p_label = load_preds(phyg_npz)
        cm, cm_norm = conf_mat(p_pred, p_label)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(5)); ax.set_xticklabels(cls)
        ax.set_yticks(range(5)); ax.set_yticklabels(cls)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("PhyG seed42 — SHHS zero-shot confusion matrix (row-normalized)")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax)
        fig.savefig(os.path.join(out_dir, "cm_phyg_shhs_seed42.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved cm_phyg_shhs_seed42.png")

        # ====== N1 misclassification analysis ======
        print("\nN1 misclassification (PhyG seed42 SHHS):")
        n1_mask = (p_label == 1)
        n1_preds = p_pred[n1_mask]
        if len(n1_preds) > 0:
            from collections import Counter
            counts = Counter(n1_preds.tolist())
            total = len(n1_preds)
            n1_dist = {cls[i]: counts.get(i, 0) / total for i in range(5)}
            print(f"  Total N1 epochs: {total}")
            for c in range(5):
                print(f"  → {cls[c]}: {counts.get(c, 0)} ({n1_dist[cls[c]]*100:.1f}%)")
            with open(os.path.join(out_dir, "n1_misclass.json"), "w") as f:
                json.dump({"total": total, "by_class": n1_dist}, f, indent=2)

    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
