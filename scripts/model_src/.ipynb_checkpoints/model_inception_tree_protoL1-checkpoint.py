import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_inceptiontime import InceptionTimeEncoder
from src.proto_graph import ProtoGraphRefiner 


def compose_probs(p1, p2, p3, eps=1e-8):
    """
    p1: (B,3) [N3, N2, O]
    p2: (B,2) [REM, O2]  within Others
    p3: (B,2) [W,   N1]  within Others->O2
    output: (B,5) [W,N1,N2,N3,REM]
    """
    p1 = p1.clamp_min(eps)
    p2 = p2.clamp_min(eps)
    p3 = p3.clamp_min(eps)

    P_N3 = p1[:, 0]
    P_N2 = p1[:, 1]
    P_O  = p1[:, 2]

    P_REM = P_O * p2[:, 0]
    P_O2  = P_O * p2[:, 1]

    P_W  = P_O2 * p3[:, 0]
    P_N1 = P_O2 * p3[:, 1]

    probs5 = torch.stack([P_W, P_N1, P_N2, P_N3, P_REM], dim=-1)
    probs5 = probs5 / probs5.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs5


def level_targets(y):
    """
    y: (B,) with mapping W=0,N1=1,N2=2,N3=3,REM=4

    Level-1 classes: [N3, N2, O]
      y1=0 if N3
      y1=1 if N2
      y1=2 otherwise (W/N1/REM)

    Level-2 classes: [REM, O2] within Others (W/N1/REM)
      y2=0 if REM else 1  (used with mask m2)

    Level-3 classes: [W, N1] within {W,N1}
      y3=0 if W else 1    (used with mask m3)
    """
    y1 = torch.full_like(y, 2)
    y1[y == 3] = 0
    y1[y == 2] = 1

    y2 = torch.ones_like(y)
    y2[y == 4] = 0

    y3 = torch.ones_like(y)
    y3[y == 0] = 0
    return y1, y2, y3


class TreeHeadMLP_L23(nn.Module):
    """Keep Level-2/3 as MLP heads."""
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.h2 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )
        self.h3 = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, z):
        l2 = self.h2(z)
        l3 = self.h3(z)
        p2 = F.softmax(l2, dim=-1)
        p3 = F.softmax(l3, dim=-1)
        return l2, l3, p2, p3


class ProtoHeadL1(nn.Module):
    """
    Level-1 prototype head:
      - prototypes: C (K, D)
      - per-prototype logits: W (K, 3)
      - soft assignment p(k|x) via distances -> softmax(-d^2/tau)
      - output logits: sum_k p(k|x) * W_k
      - output probs p1 = softmax(logits)
    """
    def __init__(self, z_dim: int, K1: int = 64, tau: float = 1.0,
                 use_pg: bool = True):
        super().__init__()
        self.K1 = K1
        self.tau = tau
        self.use_pg = use_pg

        self.proto = nn.Parameter(torch.randn(K1, z_dim) * 0.02)
        self.proto_logits = nn.Parameter(torch.zeros(K1, 3))

        # NEW: prototype graph refiner (logits-level)
        self.pg = ProtoGraphRefiner(
            D=z_dim, C=3,
            Fdim=0,           # 先不加 mech/usage 特征，控制复杂度
            hidden=64,
            topk=12,
            tau_edge=0.2
        )

    def forward(self, z):
        # z: (B, D)
        # squared euclidean distance to prototypes
        # dist2: (B, K)
        z2 = (z ** 2).sum(dim=1, keepdim=True)               # (B,1)
        p2 = (self.proto ** 2).sum(dim=1).unsqueeze(0)      # (1,K)
        dist2 = z2 + p2 - 2.0 * (z @ self.proto.t())        # (B,K)

        # assignment p(k|x)
        assign = F.softmax(-dist2 / max(self.tau, 1e-6), dim=-1)  # (B,K)

        # aggregated logits over 3 classes
        # logits1: (B,3)
        proto_logits = self.proto_logits
        if self.use_pg:
            proto_logits = self.pg(self.proto, proto_logits)   # (K,3)
        
        logits1 = assign @ proto_logits
        p1 = F.softmax(logits1, dim=-1)
        return logits1, p1, assign, dist2, proto_logits



class InceptionTreeProtoL1(nn.Module):
    """
    x30 -> InceptionTimeEncoder -> z
       -> Level-1 ProtoHead (3-class)
       -> Level-2/3 MLP heads
       -> compose -> 5-class probs
    """
    def __init__(self, z_dim=256, blocks=6, out_ch=32, head_hidden=256, K1=64, tau=1.0):
        super().__init__()
        self.enc = InceptionTimeEncoder(in_ch=1, num_blocks=blocks, out_ch=out_ch, z_dim=z_dim)
        self.l1 = ProtoHeadL1(z_dim=z_dim, K1=K1, tau=tau)
        self.l23 = TreeHeadMLP_L23(in_dim=z_dim, hidden=head_hidden)

    def forward(self, x30):
        z = self.enc(x30)                         # (B,D)
        l1, p1, assign1, dist2, proto_logits = self.l1(z)
        l2, l3, p2, p3 = self.l23(z)              # l2:(B,2), l3:(B,2)
        p5 = compose_probs(p1, p2, p3)
        extras = {
            "assign1": assign1,
            "dist2": dist2,
            "proto_logits_l1": proto_logits,   # NEW
        }
        return z, (l1, l2, l3), (p1, p2, p3), p5, extras
