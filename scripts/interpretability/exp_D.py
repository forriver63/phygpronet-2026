"""Exp D: Decision path visualization.

Pick N example 30s EEG epochs per class (W/N1/N2/N3/REM).
For each example, run fold0 ckpt, output:
- Raw EEG waveform (3000 samples)
- Level-1/2/3 probabilities
- Top-3 active prototypes (by assignment weight)
- Final 5-class prediction
- Whether prediction matches ground truth

Save figures as PNGs.
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1
from src.data_sleepedf_packed import open_packed, SleepEDFPackedDataset


def fold_idx(packed, split_json, fold_id, part):
    with open(split_json) as f: sj = json.load(f)
    subjs = sj["folds"][str(fold_id)][part]
    p2sid = {}
    for i, full in enumerate(packed.subjects):
        p2sid.setdefault(full[:5], []).append(i)
    sids = set()
    for s in subjs: sids.update(p2sid.get(s, []))
    return np.arange(packed.N, dtype=np.int64)[np.isin(np.array(packed.sid), list(sids))]


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = open_packed("/root/autodl-tmp/data/sleepedf_memmap_v2_mech", readonly=True)
    split_json = "/root/autodl-tmp/splits/sleepedf_subject_5fold_seed42.json"
    out_dir = "/root/autodl-tmp/runs/expD_decision_path"
    os.makedirs(out_dir, exist_ok=True)
    n_per_class = 4

    # Load ckpt
    ckpt = "/root/autodl-tmp/runs/phyg_unified_5fold/fold0/best_model.pt"
    s = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(
        z_dim=int(a.get("z_dim",256)),blocks=int(a.get("blocks",6)),out_ch=int(a.get("out_ch",32)),
        head_hidden=int(a.get("head_hidden",256)),K1=int(a.get("K1",96)),tau=float(a.get("tau",3.5)),
    ).to(device)
    if hasattr(m,"l1") and hasattr(m.l1,"use_pg"): m.l1.use_pg=bool(a.get("use_pg",True))
    m.load_state_dict(s["state_dict"], strict=True); m.eval()
    K = m.l1.proto_logits.shape[0]
    print(f"[ckpt loaded] K={K}", flush=True)

    # Pick examples per class from fold0 test
    idx = fold_idx(packed, split_json, 0, "test")
    yy = np.asarray(packed.y)[idx]
    rng = np.random.RandomState(42)
    picks = {}  # class -> list of sample idx
    cls_names = ["W","N1","N2","N3","REM"]
    for c in range(5):
        candidates = idx[yy == c]
        rng.shuffle(candidates)
        picks[c] = candidates[:n_per_class].tolist()

    # Run inference per example
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = []
    for c in range(5):
        for ex_i, j in enumerate(picks[c]):
            x = torch.from_numpy(np.array(packed.X[j])).unsqueeze(0).to(device).float()
            y_true = int(packed.y[j])
            sp_val = float(packed.sp[j]); sw_val = float(packed.sw[j])

            z, (l1, l2, l3), (p1, p2, p3), p5, extras = m(x)
            A = extras["assign1"][0].cpu().numpy()  # (K,)
            top3_proto = np.argsort(-A)[:3]
            top3_weights = A[top3_proto]

            y_pred = int(p5.argmax(-1).item())
            p5_np = p5[0].cpu().numpy()

            # Figure
            fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                     gridspec_kw={"height_ratios": [2, 1]})
            # Waveform
            wave = x[0,0].cpu().numpy()
            t = np.arange(3000) / 100.0
            axes[0].plot(t, wave, lw=0.5, color="black")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("EEG (µV)")
            corr = "✓" if y_pred == y_true else "✗"
            axes[0].set_title(
                f"Class={cls_names[y_true]}, Pred={cls_names[y_pred]} [{corr}], "
                f"spindle={sp_val:.2f}, slowwave={sw_val:.2f}"
            )
            axes[0].grid(alpha=0.3)

            # Bar: top-3 prototypes
            axes[1].bar(range(3), top3_weights, color=["red","orange","gold"])
            axes[1].set_xticks(range(3))
            axes[1].set_xticklabels([f"p{int(top3_proto[i])} ({top3_weights[i]:.2f})" for i in range(3)])
            axes[1].set_ylabel("Assignment weight")
            axes[1].set_title("Top-3 active prototypes")
            axes[1].grid(alpha=0.3)

            # Add p5 text
            p5_text = "  ".join(f"{cls_names[ii]}={p5_np[ii]:.2f}" for ii in range(5))
            fig.text(0.5, -0.01, f"P5: {p5_text}", ha="center", fontsize=9)

            tag = f"{cls_names[y_true]}_ex{ex_i}_pred{cls_names[y_pred]}"
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)

            results.append({
                "class": cls_names[y_true], "predicted": cls_names[y_pred],
                "correct": y_pred == y_true,
                "spindle": sp_val, "slowwave": sw_val,
                "top3_protos": top3_proto.tolist(),
                "top3_weights": top3_weights.tolist(),
                "p5": p5_np.tolist(),
            })

    with open(os.path.join(out_dir, "examples.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {len(results)} examples to {out_dir}")


if __name__ == "__main__":
    main()
