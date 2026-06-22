import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_inceptiontime import InceptionTimeEncoder


def compose_probs(p1, p2, p3, eps=1e-8):
    # p1: (B,3) [N3,N2,O]
    # p2: (B,2) [REM, O2]
    # p3: (B,2) [W, N1]
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
    # y: W=0,N1=1,N2=2,N3=3,REM=4
    y1 = torch.full_like(y, 2)
    y1[y == 3] = 0  # N3
    y1[y == 2] = 1  # N2

    y2 = torch.ones_like(y)
    y2[y == 4] = 0  # REM

    y3 = torch.ones_like(y)
    y3[y == 0] = 0  # W
    return y1, y2, y3


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


class InceptionTreeMLP(nn.Module):
    """
    x30 -> InceptionTimeEncoder -> z -> TreeHeadMLP -> compose -> 5-class probs
    """
    def __init__(self, z_dim=256, blocks=6, out_ch=32, head_hidden=256):
        super().__init__()
        self.enc = InceptionTimeEncoder(in_ch=1, num_blocks=blocks, out_ch=out_ch, z_dim=z_dim)
        self.head = TreeHeadMLP(in_dim=z_dim, hidden=head_hidden)

    def forward(self, x30):
        z = self.enc(x30)
        l1, l2, l3, p1, p2, p3 = self.head(z)
        p5 = compose_probs(p1, p2, p3)
        return z, (l1, l2, l3), (p1, p2, p3), p5
