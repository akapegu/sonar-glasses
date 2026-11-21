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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)              # (B, 512, F', T')
        h = h.mean(dim=2).transpose(1, 2)  # (B, T', 512)
        return F.log_softmax(self.fc(h).float(), dim=-1)  # (B, T', n_classes), fp32 for CTC
