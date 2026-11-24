"""ResNet-34 + CTC baseline (SoniSpeech paper, section 4.1; README section 7).

Input (B, 4, 80, T): 4 sound paths as channels, 80 range bins as height, T frames as width.
- torchvision ResNet-34 with every BatchNorm replaced by GroupNorm(32) and the stem max-pool removed
- stride 2 in time in the stem conv and the first block of stages 2, 3, 4 -> 16x temporal downsampling
- range-axis stride: 2 (standard 2D stride, default), 1 (time-only), or per stage; averaged away at the end
- mean over the range axis -> (B, T/16, 512) -> linear -> 1,001 classes (blank = 0) -> log_softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _set_stride(conv: nn.Conv2d, stride):
    conv.stride = stride


class ResNet34CTC(nn.Module):
    def __init__(self, n_classes: int = 1001, in_channels: int = 4, range_stride=2):
        """range_stride: one int for all four strided stages, or a list of 4 for
        (stem, layer2, layer3, layer4), e.g. [1, 2, 2, 2] keeps full range resolution in the stem."""
        super().__init__()
        rs = list(range_stride) if isinstance(range_stride, (list, tuple)) else [range_stride] * 4
        assert len(rs) == 4, "range_stride must be an int or a list of 4"
        r = torchvision.models.resnet34(weights=None, norm_layer=lambda c: nn.GroupNorm(32, c))
        r.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=(rs[0], 2), padding=3, bias=False)
        r.maxpool = nn.Identity()
        for layer, s in zip((r.layer2, r.layer3, r.layer4), rs[1:]):
            block = layer[0]
            _set_stride(block.conv1, (s, 2))
            _set_stride(block.downsample[0], (s, 2))
        self.encoder = nn.Sequential(r.conv1, r.bn1, r.relu, r.layer1, r.layer2, r.layer3, r.layer4)
        self.fc = nn.Linear(512, n_classes)

    @staticmethod
    def out_lengths(lengths: torch.Tensor) -> torch.Tensor:
        """Exact output length per input length: four stride-2 stages, each ceil(L / 2)
        (7x7 pad 3, 3x3 pad 1 and 1x1 pad 0 convolutions all give ceil(L / 2) at stride 2)."""
        out = lengths.clone()
        for _ in range(4):
            out = torch.div(out + 1, 2, rounding_mode="floor")
        return out

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:  # lengths unused (convolutions only)
        h = self.encoder(x)              # (B, 512, F', T')
        h = h.mean(dim=2).transpose(1, 2)  # (B, T', 512)
        return F.log_softmax(self.fc(h).float(), dim=-1)  # (B, T', n_classes), fp32 for CTC


# ---- step 5a: ResNet-18 front end + Transformer encoder ------------------------------------------
def _rope(x, cos, sin):
    """Rotary position embedding on the last dim (relative positions enter q.k through the rotation)."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class _RoPEBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder layer with rotary self-attention."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.h, self.p = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = _rope(q, cos.to(q.dtype), sin.to(q.dtype)), _rope(k, cos.to(k.dtype), sin.to(k.dtype))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class ResNetTransformerCTC(nn.Module):
    """ResNet-18 (GroupNorm, no max-pool, 16x time downsampling, range stride 2) -> mean over range
    -> linear 512 -> d_model -> N rotary Transformer layers (padding masked) -> LayerNorm -> 1,001 classes."""

    def __init__(self, n_classes=1001, in_channels=4, d_model=384, layers=6, heads=8, ff=1536, dropout=0.1):
        super().__init__()
        r = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(32, c))
        r.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        r.maxpool = nn.Identity()
        self.encoder = nn.Sequential(r.conv1, r.bn1, r.relu, r.layer1, r.layer2, r.layer3, r.layer4)
        self.inp = nn.Linear(512, d_model)
        self.blocks = nn.ModuleList(_RoPEBlock(d_model, heads, ff, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)
        self.head_dim = d_model // heads

    out_lengths = staticmethod(ResNet34CTC.out_lengths)  # same stride layout -> same lengths

    def forward(self, x, lengths=None):
        h = self.encoder(x).mean(dim=2).transpose(1, 2)  # (B, T', 512)
        h = self.inp(h)
        B, T, _ = h.shape
        pos = torch.arange(T, device=h.device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, self.head_dim // 2, device=h.device, dtype=torch.float32) * 2 / self.head_dim))
        ang = pos[:, None] * inv[None]
        cos, sin = ang.cos(), ang.sin()  # (T, head_dim/2), broadcast over (B, heads)
        mask = None
        if lengths is not None:
            olens = self.out_lengths(lengths.to(h.device))
            mask = (torch.arange(T, device=h.device)[None] < olens[:, None])[:, None, None, :]  # (B,1,1,T) keep
        for blk in self.blocks:
            h = blk(h, cos, sin, mask)
        return F.log_softmax(self.fc(self.ln(h)).float(), dim=-1)


def build_model(model_cfg, n_classes):
    """Model from the config's `model` section: arch resnet34 (default) or resnet18_transformer."""
    arch = model_cfg.get("arch", "resnet34")
    if arch == "resnet34":
        return ResNet34CTC(n_classes=n_classes, range_stride=model_cfg.get("range_stride", 2))
    if arch == "resnet18_transformer":
        return ResNetTransformerCTC(n_classes=n_classes, d_model=model_cfg.get("d_model", 384),
                                    layers=model_cfg.get("layers", 6), heads=model_cfg.get("heads", 8),
                                    ff=model_cfg.get("ff", 1536), dropout=model_cfg.get("dropout", 0.1))
    raise ValueError(f"unknown model.arch {arch}")
