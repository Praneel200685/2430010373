"""Model zoo: baselines (CNN, LSTM, BiLSTM, ResNet1D, CNN-BiLSTM) and the proposed
Multi-scale SE-CNN + BiLSTM + Temporal-Attention network (HybridNet with all flags on).

Input: (B, 1, 187)  ->  Output logits: (B, 5)
"""
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
    """Squeeze-and-Excitation channel attention for 1-D feature maps."""
    def __init__(self, c, r=8):
        super().__init__()
        m = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, m), nn.ReLU(inplace=True), nn.Linear(m, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class MultiScaleBlock(nn.Module):
    """Parallel conv branches (k=3,5,7) capture narrow QRS and wider P/T-wave morphology; concat + optional SE."""
    def __init__(self, cin, branch_ch, ks=(3, 5, 7), se=True):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(cin, branch_ch, k, padding=k // 2, bias=False),
                          nn.BatchNorm1d(branch_ch), nn.ReLU(inplace=True)) for k in ks])
        self.out_ch = branch_ch * len(ks)
        self.se = SE1d(self.out_ch) if se else nn.Identity()

    def forward(self, x):
        return self.se(torch.cat([b(x) for b in self.branches], 1))


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
        h = torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1]
        return self.head(h)


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
    """Residual 1-D CNN in the style of Kachuee et al. (2018), ECG heartbeat classification."""
    def __init__(self, num_classes=5, dropout=0.3, **_):
        super().__init__()
        self.stem = nn.Conv1d(1, 32, 5, padding=2)
        self.blocks = nn.Sequential(*[ResBlock(32) for _ in range(5)])
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(-1))


class HybridNet(nn.Module):
    """CNN -> BiLSTM -> pooling -> classifier.

    multiscale / se : component A (multi-scale SE-CNN front end)
    attn            : component B (additive temporal attention pooling over BiLSTM states)
    all False       : plain CNN-BiLSTM baseline (mean pooling), same width as the proposed model
    """
    def __init__(self, num_classes=5, hidden=64, dropout=0.3, multiscale=False, se=False, attn=False, **_):
        super().__init__()
        if multiscale:
            b1, b2 = MultiScaleBlock(1, 16, se=se), MultiScaleBlock(48, 32, se=se)
            self.cnn = nn.Sequential(b1, nn.MaxPool1d(2), b2, nn.MaxPool1d(2))
            c = b2.out_ch                                   # 96 channels
        else:
            self.cnn = nn.Sequential(ConvBlock(1, 48), ConvBlock(48, 96))
            c = 96
        self.lstm = nn.LSTM(c, hidden, batch_first=True, bidirectional=True)
        d = 2 * hidden
        self.use_attn = attn
        if attn:
            self.att = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(64, num_classes))
        self.last_attn = None

    def forward(self, x):
        h, _ = self.lstm(self.cnn(x).transpose(1, 2))       # (B, T, 2H)
        if self.use_attn:
            a = torch.softmax(self.att(h).squeeze(-1), dim=1)   # (B, T)
            self.last_attn = a.detach()
            z = (h * a.unsqueeze(-1)).sum(1)
        else:
            z = h.mean(1)
        return self.head(z)


def build_model(name, num_classes=5, **kw):
    if name == "cnn": return CNN1D(num_classes, **kw)
    if name == "lstm": return LSTMNet(num_classes, bidirectional=False, **kw)
    if name == "bilstm": return LSTMNet(num_classes, bidirectional=True, **kw)
    if name == "resnet1d": return ResNet1D(num_classes, **kw)
    if name == "hybrid": return HybridNet(num_classes, **kw)
    raise ValueError(f"Unknown model: {name}")
