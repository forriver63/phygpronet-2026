import torch
import torch.nn as nn

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

    def forward(self, x):  # (B,1,T)
        return self.net(x) # (B,d,T')

class TinyTransformer(nn.Module):
    def __init__(self, d=64, nhead=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x):  # (B,L,d)
        return self.enc(x)

class SharedEncoder(nn.Module):
    """共享多尺度编码器，输出融合后的 feature (B, 3d)."""
    def __init__(self, d=64):
        super().__init__()
        # 30s branch
        self.stem30 = ConvStem(1, d)
        self.proj30 = nn.Linear(d, d)
        self.tr30 = TinyTransformer(d=d)

        # 90s branch
        self.stem90 = ConvStem(1, d)
        self.pool90 = nn.AdaptiveAvgPool1d(64)
        self.proj90 = nn.Linear(d, d)
        self.tr90 = TinyTransformer(d=d)

        # 5s token branch
        self.stem5 = ConvStem(1, d)
        self.proj5 = nn.Linear(d, d)
        self.tr5 = TinyTransformer(d=d)

        self.out_dim = 3 * d

    def encode30(self, x):
        z = self.stem30(x)          # (B,d,T')
        z = z.transpose(1,2)        # (B,L,d)
        z = self.proj30(z)
        z = self.tr30(z).mean(1)    # (B,d)
        return z

    def encode90(self, x):
        z = self.stem90(x)          # (B,d,T')
        z = self.pool90(z)          # (B,d,64)
        z = z.transpose(1,2)        # (B,64,d)
        z = self.proj90(z)
        z = self.tr90(z).mean(1)    # (B,d)
        return z

    def encode5(self, x):
        # x: (B,1,6,T5)
        B, C, K, T = x.shape
        x = x.reshape(B*K, C, T)
        z = self.stem5(x).mean(-1)  # token pooling -> (B*K,d)
        z = z.reshape(B, K, -1)     # (B,6,d)
        z = self.proj5(z)
        z = self.tr5(z).mean(1)     # (B,d)
        return z

    def forward(self, x30, x90, x5):
        a = self.encode30(x30)
        b = self.encode90(x90)
        c = self.encode5(x5)
        feat = torch.cat([a,b,c], dim=-1)  # (B,3d)
        return feat

class HierSleepNet(nn.Module):
    """
    输出：
      - logit_wake: (B,1)  预测 W(1) vs Sleep(0)
      - logits_sleep: (B,4)  预测 N1,N2,N3,REM（仅对 sleep 样本计算loss）
    """
    def __init__(self, d=64):
        super().__init__()
        self.enc = SharedEncoder(d=d)

        self.head_wake = nn.Sequential(
            nn.Linear(self.enc.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.head_sleep = nn.Sequential(
            nn.Linear(self.enc.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )

    def forward(self, x30, x90, x5):
        feat = self.enc(x30, x90, x5)
        logit_wake = self.head_wake(feat)       # (B,1)
        logits_sleep = self.head_sleep(feat)    # (B,4)
        return logit_wake, logits_sleep
