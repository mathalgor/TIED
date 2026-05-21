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
    """TEED fusion before the final edge map.

    Mid-channel count is ``in_ch * 8`` (TEED hard-coded it as 24, which
    only worked for in_ch=3). Parameterising it lets in_ch=4 — needed
    by TEDdeep, whose 4th head extends the fusion input. For in_ch=3
    the structure is byte-identical to TEED, so existing TED/TEDup
    checkpoints keep loading.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = in_ch * 8
        self.DWconv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, groups=mid_ch)
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


class TEDup(nn.Module):
    """Wider + deeper TED (MCED's TEEDup, ~180k params, ~3x TED).

    * Channel widths 24 / 48 / 72 (was 16 / 32 / 48).
    * Dense block with 2 layers (was 1).
    Forward and output list are identical to TED — 4 heads (3 side + 1
    fused). Use when TED saturates on richer data.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.block_1 = DoubleConvBlock(in_channels, 24, 24, stride=2)
        self.block_2 = DoubleConvBlock(24, 48, use_act=False)
        self.dblock_3 = _DenseBlock(2, 48, 72)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(24, 48, 2)
        self.pre_dense_3 = SingleConvBlock(48, 72, 1)

        self.up_block_1 = UpConvBlock(24, 1)
        self.up_block_2 = UpConvBlock(48, 1)
        self.up_block_3 = UpConvBlock(72, 2)

        self.block_cat = DoubleFusion(3, 3)
        self.apply(weight_init)

    # Forward and resize_input are state-independent — reuse TED's.
    resize_input = staticmethod(TED.resize_input)
    forward = TED.forward


class TEDdeep(nn.Module):
    """TED + a 4th encoder stage at stride 8 (MCED's TEEDdeep).

    Doubles the receptive field (~32 px -> ~64 px on input). Channel
    widths kept at TED defaults; only depth grows. Returns FIVE outputs
    (4 side + 1 fused) instead of TED's 4.

    Useful when the model needs more global context — e.g. to treat one
    long continuous edge as a single structure instead of evaluating it
    pixel by pixel.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        # Original 3-stage backbone, identical to TED.
        self.block_1 = DoubleConvBlock(in_channels, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        # New 4th stage at stride 8.
        self.block_3 = DoubleConvBlock(48, 64, use_act=False)
        self.dblock_4 = _DenseBlock(2, 64, 96)
        self.side_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)
        self.up_block_4 = UpConvBlock(96, 3)

        # Fusion now takes 4 side outputs.
        self.block_cat = DoubleFusion(4, 4)
        self.apply(weight_init)

    resize_input = staticmethod(TED.resize_input)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        assert x.ndim == 4, x.shape
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side

        block_3_pre_dense = self.pre_dense_3(block_2_down)
        dblock_3_out, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        block_3 = self.block_3(dblock_3_out)
        block_3_down = self.maxpool(block_3)
        block_2_side2 = self.side_2(block_2_down)
        block_3_add = block_3_down + block_2_side2

        block_4_pre_dense = self.pre_dense_4(block_3_down)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])

        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(dblock_3_out)
        out_4 = self.up_block_4(block_4)

        results = [out_1, out_2, out_3, out_4]
        fused = self.block_cat(torch.cat(results, dim=1))
        results.append(fused)
        return results


MODELS: dict[str, type[nn.Module]] = {
    "ted":     TED,
    "tedup":   TEDup,
    "teddeep": TEDdeep,
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
