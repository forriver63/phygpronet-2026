import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# Multi-scale Encoder (CNN + Transformer) -> embedding z(x) in R^D
# ------------------------

class ConvStem(nn.Module):
    def __init__(self, in_ch=1, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d, 7, stride=2, padding=3),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Conv1d(d, d, 3, stride=1, padding=1),
            nn.BatchNorm1d(d),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TinyTransformer(nn.Module):
    def __init__(self, d=64, nhead=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x):
        return self.enc(x)


class MultiScaleEncoder(nn.Module):
    """
    输入:
      x30: (B,1,T30)
      x90: (B,1,3*T30)
      x5 : (B,1,6,T5)
    输出:
      z: (B, D)
    """
    def __init__(self, d_model=64, out_dim=256):
        super().__init__()

        # 30s
        self.stem30 = ConvStem(1, d_model)
        self.proj30 = nn.Linear(d_model, d_model)
        self.tr30 = TinyTransformer(d=d_model)

        # 90s
        self.stem90 = ConvStem(1, d_model)
        self.pool90 = nn.AdaptiveAvgPool1d(64)
        self.proj90 = nn.Linear(d_model, d_model)
        self.tr90 = TinyTransformer(d=d_model)

        # 5s tokens
        self.stem5 = ConvStem(1, d_model)
        self.proj5 = nn.Linear(d_model, d_model)
        self.tr5 = TinyTransformer(d=d_model)

        self.fuse = nn.Sequential(
            nn.Linear(3 * d_model, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.out_dim = out_dim

    def encode30(self, x):
        z = self.stem30(x)          # (B,d,T')
        z = z.transpose(1, 2)       # (B,L,d)
        z = self.proj30(z)
        z = self.tr30(z).mean(1)    # (B,d)
        return z

    def encode90(self, x):
        z = self.stem90(x)          # (B,d,T')
        z = self.pool90(z)          # (B,d,64)
        z = z.transpose(1, 2)       # (B,64,d)
        z = self.proj90(z)
        z = self.tr90(z).mean(1)    # (B,d)
        return z

    def encode5(self, x):
        # x: (B,1,6,T5) -> (B*6,1,T5)
        B, C, K, T = x.shape
        x = x.reshape(B * K, C, T)
        z = self.stem5(x).mean(-1)     # (B*6,d) token pooling
        z = z.reshape(B, K, -1)        # (B,6,d)
        z = self.proj5(z)
        z = self.tr5(z).mean(1)        # (B,d)
        return z

    def forward(self, x30, x90, x5):
        a = self.encode30(x30)
        b = self.encode90(x90)
        c = self.encode5(x5)
        feat = torch.cat([a, b, c], dim=-1)      # (B,3d)
        z = self.fuse(feat)                      # (B,D)
        return z


# ------------------------
# Compose: (p1,p2,p3) -> 5-class probs
# y mapping: W=0,N1=1,N2=2,N3=3,REM=4
# Level-1: [N3, N2, O]
# Level-2: [REM, O2] (only meaningful when Level-1 chooses O)
# Level-3: [W, N1]   (only meaningful when Level-1 chooses O and Level-2 chooses O2)
# ------------------------

def compose_probs(p1, p2, p3, eps=1e-8):
    """
    p1: (B,3) -> [N3, N2, O]
    p2: (B,2) -> [REM, O2]
    p3: (B,2) -> [W, N1]
    return probs5: (B,5) order [W,N1,N2,N3,REM]
    """
    # ensure float
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


# ------------------------
# Baseline head: MLP(z) outputs each level logits
# ------------------------

class TreeHeadMLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.h1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 3))
        self.h2 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 2))
        self.h3 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 2))

    def forward(self, z):
        l1 = self.h1(z)
        l2 = self.h2(z)
        l3 = self.h3(z)
        p1 = F.softmax(l1, dim=-1)
        p2 = F.softmax(l2, dim=-1)
        p3 = F.softmax(l3, dim=-1)
        return l1, l2, l3, p1, p2, p3


# ------------------------
# PrototypeLayer: learnable prototypes + soft assignment + prototype logits
# ------------------------

class PrototypeLayer(nn.Module):
    """
    Given z (B,D), produce:
      logits_x: (B,C) = sum_k p(k|x) * logits_proto(k)
    where logits_proto(k) = MLP(c_k)

    - prototypes: (K,D) learnable
    - tau: fixed temperature (float)
    """
    def __init__(self, D, K, C, tau=1.0, hidden=256):
        super().__init__()
        self.D, self.K, self.C = D, K, C
        self.tau = float(tau)
        self.prototypes = nn.Parameter(torch.randn(K, D) * 0.02)

        self.proto_mlp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, C),
        )

    def forward(self, z):
        # z: (B,D), prototypes: (K,D)
        # dist^2 = ||z||^2 + ||c||^2 - 2 z c
        z2 = (z ** 2).sum(dim=1, keepdim=True)             # (B,1)
        c2 = (self.prototypes ** 2).sum(dim=1).unsqueeze(0) # (1,K)
        dist2 = z2 + c2 - 2.0 * (z @ self.prototypes.t())  # (B,K)

        p = torch.softmax(-dist2 / self.tau, dim=-1)       # (B,K)
        logits_proto = self.proto_mlp(self.prototypes)     # (K,C)
        logits_x = p @ logits_proto                        # (B,C)
        probs_x = F.softmax(logits_x, dim=-1)
        return logits_x, probs_x, p


class TreeHeadProto(nn.Module):
    def __init__(self, in_dim, K1=64, K2=48, K3=48, tau=1.0, hidden=256):
        super().__init__()
        self.l1 = PrototypeLayer(in_dim, K1, 3, tau=tau, hidden=hidden)
        self.l2 = PrototypeLayer(in_dim, K2, 2, tau=tau, hidden=hidden)
        self.l3 = PrototypeLayer(in_dim, K3, 2, tau=tau, hidden=hidden)

    def forward(self, z):
        log1, p1, a1 = self.l1(z)
        log2, p2, a2 = self.l2(z)
        log3, p3, a3 = self.l3(z)
        return log1, log2, log3, p1, p2, p3, (a1, a2, a3)


# ------------------------
# Full Models
# ------------------------

class P1TreeMLP(nn.Module):
    def __init__(self, d_model=64, z_dim=256):
        super().__init__()
        self.enc = MultiScaleEncoder(d_model=d_model, out_dim=z_dim)
        self.head = TreeHeadMLP(in_dim=z_dim)

    def forward(self, x30, x90, x5):
        z = self.enc(x30, x90, x5)
        return (z,) + self.head(z)


class P1TreeProto(nn.Module):
    def __init__(self, d_model=64, z_dim=256, K1=64, K2=48, K3=48, tau=1.0):
        super().__init__()
        self.enc = MultiScaleEncoder(d_model=d_model, out_dim=z_dim)
        self.head = TreeHeadProto(in_dim=z_dim, K1=K1, K2=K2, K3=K3, tau=tau)

    def forward(self, x30, x90, x5):
        z = self.enc(x30, x90, x5)
        log1, log2, log3, p1, p2, p3, assigns = self.head(z)
        return z, log1, log2, log3, p1, p2, p3, assigns
