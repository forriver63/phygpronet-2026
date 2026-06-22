# src/proto_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoGraphRefiner(nn.Module):
    """
    Light GCN-style refinement on prototype logits.

    Inputs:
      proto_vec:    (K, D)    prototype centers
      proto_logits: (K, C)    per-prototype class logits (C=3 for Level-1)
      node_feat:    (K, F)    optional node features (e.g., mech stats / usage), can be None

    Output:
      refined_logits: (K, C)
    """
    def __init__(self, D: int, C: int, Fdim: int = 0, hidden: int = 64, topk: int = 12, tau_edge: float = 0.2):
        super().__init__()
        self.topk = topk
        self.tau_edge = tau_edge

        in_dim = C + (Fdim if Fdim > 0 else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, C),
        )

    @torch.no_grad()
    def build_adj(self, proto_vec: torch.Tensor) -> torch.Tensor:
        """
        Build row-normalized adjacency A_hat using cosine similarity top-k.
        Returns A_hat: (K, K)
        """
        K = proto_vec.size(0)
        v = F.normalize(proto_vec, dim=-1)          # (K, D)
        sim = v @ v.t()                             # (K, K)
        sim.fill_diagonal_(-1e9)                    # exclude self for topk selection

        k = min(self.topk, K - 1)
        vals, idx = torch.topk(sim, k=k, dim=1)     # (K, k)

        # weights: softmax over neighbors
        w = torch.softmax(vals / self.tau_edge, dim=1)  # (K, k)

        A = torch.zeros((K, K), device=proto_vec.device, dtype=proto_vec.dtype)
        A.scatter_(1, idx, w)

        # add self-loop
        A = A + torch.eye(K, device=A.device, dtype=A.dtype)

        # row-normalize
        A = A / (A.sum(dim=1, keepdims=True).clamp_min(1e-8))
        return A

    def forward(self, proto_vec: torch.Tensor, proto_logits: torch.Tensor, node_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Residual refinement: logits' = logits + GNN(logits, feat) aggregated by A_hat
        """
        if node_feat is None:
            X = proto_logits
        else:
            X = torch.cat([proto_logits, node_feat], dim=-1)  # (K, C+F)

        A_hat = self.build_adj(proto_vec)                     # (K, K)
        H = A_hat @ X                                         # (K, C+F)
        delta = self.mlp(H)                                   # (K, C)
        return proto_logits + delta
