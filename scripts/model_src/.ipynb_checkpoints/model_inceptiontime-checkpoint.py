import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock1D(nn.Module):
    """
    InceptionTime-style block for 1D signals.
    """
    def __init__(self, in_ch, out_ch, kernel_sizes=(9, 19, 39), bottleneck=32, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.bottleneck = None
        if in_ch > 1:
            self.bottleneck = nn.Conv1d(in_ch, bottleneck, kernel_size=1, bias=False)
            in_ch2 = bottleneck
        else:
            in_ch2 = in_ch

        # parallel convs
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch2, out_ch, k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])

        # maxpool branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_ch * (len(kernel_sizes) + 1))
        self.act = nn.ReLU()

        if use_residual:
            self.res = nn.Sequential(
                nn.Conv1d(in_ch, out_ch * (len(kernel_sizes) + 1), kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch * (len(kernel_sizes) + 1)),
            )

    def forward(self, x):
        x_in = x
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        outs = [conv(x) for conv in self.convs]
        outs.append(self.conv_pool(self.maxpool(x_in)))
        y = torch.cat(outs, dim=1)
        y = self.act(self.bn(y))

        if self.use_residual:
            y = y + self.res(x_in)
            y = self.act(y)
        return y


class InceptionTimeEncoder(nn.Module):
    """
    Input: (B,1,T)
    Output: (B, D)
    """
    def __init__(self, in_ch=1, num_blocks=6, out_ch=32, z_dim=256, kernel_sizes=(9, 19, 39)):
        super().__init__()
        blocks = []
        ch = in_ch
        for i in range(num_blocks):
            blocks.append(InceptionBlock1D(
                in_ch=ch,
                out_ch=out_ch,
                kernel_sizes=kernel_sizes,
                bottleneck=32,
                use_residual=True
            ))
            ch = out_ch * (len(kernel_sizes) + 1)
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, z_dim)

    def forward(self, x):
        h = self.blocks(x)
        h = self.pool(h).squeeze(-1)
        z = self.proj(h)
        return z


class InceptionTimeClassifier(nn.Module):
    """
    Simple 5-class baseline: z -> logits(5)
    """
    def __init__(self, z_dim=256, num_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(z_dim, num_classes)
        )

    def forward(self, z):
        return self.fc(z)
