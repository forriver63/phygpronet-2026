"""Exp E: visualize PG-R (prototype graph refinement) structure.

Loads fold0 ckpt, extracts:
- Prototype vectors (K, D)
- Adjacency matrix (K, K) via cosine + topk
- Per-prototype dominant class (from Exp A's spec score)

Visualizes:
- Heatmap of adjacency (sorted by dominant class)
- 2D embedding (t-SNE / PCA) of prototypes, colored by class
- Per-class community structure
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts")
sys.path.insert(0, "/root/autodl-tmp/code/proroga/scripts/src")

from src.model_inception_tree_protoL1 import InceptionTreeProtoL1


@torch.no_grad()
def main():
    out_dir = "/root/autodl-tmp/runs/expE_pg_graph"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fold0 ckpt
    ckpt_path = "/root/autodl-tmp/runs/phyg_unified_5fold/fold0/best_model.pt"
    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = s.get("args", {})
    m = InceptionTreeProtoL1(
        z_dim=int(a.get("z_dim",256)), blocks=int(a.get("blocks",6)),
        out_ch=int(a.get("out_ch",32)), head_hidden=int(a.get("head_hidden",256)),
        K1=int(a.get("K1",96)), tau=float(a.get("tau",3.5)),
    ).to(device)
    if hasattr(m, "l1") and hasattr(m.l1, "use_pg"):
        m.l1.use_pg = bool(a.get("use_pg", True))
    m.load_state_dict(s["state_dict"], strict=True)
    m.eval()

    K = m.l1.proto_logits.shape[0]
    proto_vec = m.l1.proto.detach().cpu()     # (K, D)
    proto_logits = m.l1.proto_logits.detach().cpu().numpy()  # (K, 3) [N3, N2, Other]
    print(f"K={K}, D={proto_vec.shape[1]}")

    # Build adjacency
    if hasattr(m.l1, "pg") and m.l1.pg is not None:
        A = m.l1.pg.build_adj(proto_vec.to(device)).cpu().numpy()
        topk = m.l1.pg.topk
    else:
        # Reconstruct manually
        v = F.normalize(proto_vec, dim=-1)
        sim = v @ v.t()
        sim.fill_diagonal_(-1e9)
        topk = 12
        vals, idx = torch.topk(sim, k=topk, dim=1)
        w = torch.softmax(vals / 0.2, dim=1)
        A = torch.zeros(K, K)
        A.scatter_(1, idx, w)
        A = A + torch.eye(K)
        A = (A / A.sum(1, keepdim=True).clamp_min(1e-8)).numpy()

    print(f"adjacency: shape={A.shape}, mean nnz/row={(A>0.001).sum()/K:.1f}")

    # Compute per-prototype class via spec (from previous Exp A, fold0)
    # Use proto_logits argmax as proxy (already class-aligned via mech reg)
    cls_names = ["N3", "N2", "Other(W/N1/REM)"]  # Level-1 labels
    dom_cls = proto_logits.argmax(axis=1)
    print(f"dominant class distribution: {np.bincount(dom_cls, minlength=3).tolist()}")

    # Sort by dominant class for heatmap clarity
    order = np.argsort(dom_cls)
    A_sorted = A[order][:, order]
    dom_sorted = dom_cls[order]

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # 1. Adjacency heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(A_sorted, cmap="viridis", aspect="auto",
                       norm=LogNorm(vmin=0.001, vmax=A_sorted.max()))
        # Add class boundary lines
        boundaries = np.where(np.diff(dom_sorted) != 0)[0]
        for b in boundaries:
            ax.axhline(b + 0.5, color="white", lw=0.5)
            ax.axvline(b + 0.5, color="white", lw=0.5)
        ax.set_xlabel("Prototype j (sorted by class)")
        ax.set_ylabel("Prototype i")
        ax.set_title(f"PG-R Adjacency (K={K}, topk={topk}, sorted by Level-1 class)")
        fig.colorbar(im, ax=ax, label="edge weight")
        fig.savefig(os.path.join(out_dir, "pg_adjacency.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] pg_adjacency.png")

        # 2. Compute community fraction: how much edge mass stays within same class?
        within_class_mass = 0.0
        total_mass = 0.0
        for i in range(K):
            for j in range(K):
                w = A[i, j]
                if w > 1e-8:
                    total_mass += w
                    if dom_cls[i] == dom_cls[j]:
                        within_class_mass += w
        within_frac = within_class_mass / total_mass
        print(f"Within-class edge mass: {within_frac:.3f}")

        # 3. PCA / t-SNE 2D embedding
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(proto_vec.numpy())
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        colors = ["#d62728", "#2ca02c", "#888"]  # N3 red, N2 green, Other gray
        for c in range(3):
            mask = dom_cls == c
            ax.scatter(coords[mask, 0], coords[mask, 1], s=60, c=colors[c],
                       alpha=0.7, edgecolor="black", linewidth=0.4, label=f"{cls_names[c]} (n={mask.sum()})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("Prototype vectors (PCA), colored by Level-1 dominant class")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(out_dir, "pg_pca.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] pg_pca.png")

        # 4. Save summary
        report = {
            "K": int(K),
            "topk": int(topk),
            "dom_cls_count": np.bincount(dom_cls, minlength=3).tolist(),
            "within_class_edge_mass_fraction": float(within_frac),
            "pca_explained_var": pca.explained_variance_ratio_.tolist(),
        }
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
    except Exception as e:
        print(f"[plot error] {e}")


if __name__ == "__main__":
    main()
