"""TED — Tiny and Efficient Edge Detector (TEED architecture).

Ported from https://github.com/xavysp/TEED (ted.py). The only change vs
the original is that ``in_channels`` is a constructor argument so we can
feed either RGB (3) or single-channel gray (1) inputs depending on
``tied.toml``'s ``[dataset].source``.

Returns a list of 4 logits tensors (3 multi-scale + 1 fused).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tied.smish import Smish, smish as Fsmish


def weight_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DoubleFusion(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, 3, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24, 3, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return Fsmish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.add_module("conv1", nn.Conv2d(in_features, out_features, 3, padding=2))
        self.add_module("smish1", Smish())
        self.add_module("conv2", nn.Conv2d(out_features, out_features, 3))

    def forward(self, x):
        x1, x2 = x
        new_features = super().forward(Fsmish(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, in_features: int, out_features: int):
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i+1}",
                            _DenseLayer(in_features, out_features))
            in_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int):
        super().__init__()
        self.constant_features = 16
        all_pads = [0, 0, 1, 3, 7]
        layers = []
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = 1 if i == up_scale - 1 else self.constant_features
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride)

    def forward(self, x):
        return self.conv(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features: int, mid_features: int,
                 out_features: int | None = None,
                 stride: int = 1, use_act: bool = True):
        super().__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = Smish()

    def forward(self, x):
        x = self.smish(self.conv1(x))
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class TED(nn.Module):
    """Tiny and Efficient Edge Detector.

    ``in_channels`` selects the input variant: 3 for RGB sources, 1 for
    grayscale sources.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.block_1 = DoubleConvBlock(in_channels, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)

        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        self.block_cat = DoubleFusion(3, 3)
        self.apply(weight_init)

    @staticmethod
    def resize_input(tensor: torch.Tensor) -> torch.Tensor:
        h, w = tensor.shape[2], tensor.shape[3]
        if h % 8 or w % 8:
            new_h = ((h // 8) + 1) * 8
            new_w = ((w // 8) + 1) * 8
            tensor = F.interpolate(tensor, size=(new_h, new_w),
                                   mode="bicubic", align_corners=False)
        return tensor

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        assert x.ndim == 4, x.shape
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side

        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        results = [out_1, out_2, out_3]
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)
        results.append(block_cat)
        return results


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
