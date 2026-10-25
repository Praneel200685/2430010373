"""Model zoo: baselines (1D-CNN, LSTM, BiLSTM, ResNet-1D), configurable hybrid, Transformer variant."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=5, pool=True):
        super().__init__()
        layers = [nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SE1d(nn.Module):
    """Squeeze-and-Excitation channel attention (Eq. 3)."""
    def __init__(self, c, r=8):
        super().__init__()
        m = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, m), nn.ReLU(inplace=True), nn.Linear(m, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class MultiScaleBlock(nn.Module):
    """Parallel conv branches (k=3,5,7) + optional SE gating (Eq. 2)."""
    def __init__(self, cin, branch_ch, ks=(3, 5, 7), se=True):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(cin, branch_ch, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True))
            for k in ks
        ])
        self.out_ch = branch_ch * len(ks)
        self.se = SE1d(self.out_ch) if se else nn.Identity()

    def forward(self, x):
        return self.se(torch.cat([b(x) for b in self.branches], dim=1))


class CNN1D(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3, **_):
        super().__init__()
        self.features = nn.Sequential(ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(-1))


class LSTMNet(nn.Module):
    def __init__(self, num_classes=5, hidden=64, dropout=0.3, bidirectional=False, **_):
        super().__init__()
        self.bi = bidirectional
        self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x):
        _, (h, _) = self.lstm(x.transpose(1, 2))
        return self.head(torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1])


class ResBlock(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.c2 = nn.Conv1d(c, c, k, padding=k // 2)
        self.pool = nn.MaxPool1d(5, stride=2, padding=2)

    def forward(self, x):
        o = self.c2(F.relu(self.c1(x)))
        return self.pool(F.relu(o + x))


class ResNet1D(nn.Module):
    """Kachuee et al. (2018)-style residual network."""
    def __init__(self, num_classes=5, dropout=0.3, **_):
        super().__init__()
        self.stem = nn.Conv1d(1, 32, 5, padding=2)
        self.blocks = nn.Sequential(*[ResBlock(32) for _ in range(5)])
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(-1))


class HybridNet(nn.Module):
    """Configurable CNN-BiLSTM.

    multiscale/se -> Component A (multi-scale SE-CNN front end)
    attn          -> Component B (additive temporal attention pooling, Eq. 4)
    all False     -> plain CNN-BiLSTM baseline with mean pooling
    """
    def __init__(self, num_classes=5, hidden=64, dropout=0.3, multiscale=False, se=False, attn=False, **_):
        super().__init__()
        self.use_attn = attn
        if multiscale:
            b1 = MultiScaleBlock(1, 16, se=se)    # 48 channels
            b2 = MultiScaleBlock(48, 32, se=se)   # 96 channels
            self.cnn = nn.Sequential(b1, nn.MaxPool1d(2), b2, nn.MaxPool1d(2))
            c = b2.out_ch
        else:
            self.cnn = nn.Sequential(ConvBlock(1, 48), ConvBlock(48, 96))
            c = 96
        self.lstm = nn.LSTM(c, hidden, batch_first=True, bidirectional=True)
        d = 2 * hidden
        if attn:
            self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, num_classes))
        self.last_attn = None

    def forward(self, x):
        h, _ = self.lstm(self.cnn(x).transpose(1, 2))      # (B, T, 2H)
        if self.use_attn:
            a = torch.softmax(self.att(h).squeeze(-1), dim=1)
            self.last_attn = a.detach()
            z = (h * a.unsqueeze(-1)).sum(1)
        else:
            z = h.mean(1)
        return self.head(z)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MS_SE_CNN_Transformer(nn.Module):
    """Multi-scale SE-CNN + Pre-LN Transformer encoder with [CLS] token readout.

    NOTE: this implementation uses d_model=96, dim_feedforward=192, sinusoidal positional
    encoding and a [CLS] token. It is NOT the same as the paper's description
    (d_model=64, d_ff=256, learnable positional embedding, additive attention pooling).
    Keep the paper and code consistent before reporting results.
    """
    def __init__(self, num_classes=5, d_model=96, nhead=4, num_layers=2, dim_feedforward=192, dropout=0.3, **_):
        super().__init__()
        b1 = MultiScaleBlock(1, 16, se=True)
        b2 = MultiScaleBlock(48, 32, se=True)
        self.cnn = nn.Sequential(b1, nn.MaxPool1d(2), b2, nn.MaxPool1d(2))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_encoder = PositionalEncoding1D(d_model, max_len=60)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        feat = self.cnn(x).transpose(1, 2)                         # (B, 46, 96)
        cls = self.cls_token.expand(feat.shape[0], -1, -1)
        feat = self.pos_encoder(torch.cat([cls, feat], dim=1))     # (B, 47, 96)
        return self.head(self.transformer(feat)[:, 0])


def build_model(name, num_classes=5, **kw):
    registry = {
        "cnn": lambda: CNN1D(num_classes, **kw),
        "lstm": lambda: LSTMNet(num_classes, bidirectional=False, **kw),
        "bilstm": lambda: LSTMNet(num_classes, bidirectional=True, **kw),
        "resnet1d": lambda: ResNet1D(num_classes, **kw),
        "hybrid": lambda: HybridNet(num_classes, **kw),
        "ms_transformer": lambda: MS_SE_CNN_Transformer(num_classes, **kw),
    }
    if name not in registry:
        raise ValueError(f"Unknown architecture: {name}")
    return registry[name]()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: python -m src.models
    for label, kw in [("proposed (MS-SE-CNN-BiLSTM-Attn)", dict(multiscale=True, se=True, attn=True)),
                      ("plain CNN-BiLSTM", dict(multiscale=False, se=False, attn=False))]:
        print(f"{label:<36}: {count_params(build_model('hybrid', **kw)):,} params")
    for n in ("cnn", "lstm", "bilstm", "resnet1d", "ms_transformer"):
        print(f"{n:<36}: {count_params(build_model(n)):,} params")
