"""The two fixed, compact models (< 1M parameters each)."""
import torch
import torch.nn as nn


def _conv(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Spatial model. Input (N, T*B, H, W) with H, W multiples of 4. Output (N, K, H, W)."""

    def __init__(self, in_ch, n_classes, base=32, dropout=0.1):
        super().__init__()
        self.e1 = _conv(in_ch, base)
        self.e2 = _conv(base, base * 2)
        self.b = nn.Sequential(_conv(base * 2, base * 4), nn.Dropout2d(dropout))
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _conv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _conv(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


class PixelTransformer(nn.Module):
    """Temporal model. Input (N, T, B): one token per month. Output (N, K)."""

    def __init__(self, n_bands, months, n_classes, d_model=64, layers=3, heads=4, ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_bands, d_model)
        self.month_emb = nn.Embedding(13, d_model)            # calendar month 1..12
        self.register_buffer("month_ids", torch.tensor(list(months), dtype=torch.long))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.inp(x) + self.month_emb(self.month_ids).unsqueeze(0)
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1), h], dim=1)
        h = self.enc(h)
        return self.head(self.norm(h[:, 0]))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(kind, n_months, n_bands, months, n_classes, cfg):
    if kind == "cnn":
        return SmallUNet(n_months * n_bands, n_classes)
    if kind == "transformer":
        t = cfg["training"]["transformer"]
        return PixelTransformer(n_bands, months, n_classes, t["d_model"], t["layers"], t["heads"], t["ff"])
    raise ValueError(kind)
