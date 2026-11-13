"""
MORDEN: Multiscale ORiented DENdritic Semantic Segmentation Network.
Architecture based on Hu et al. (Research in Astronomy and Astrophysics 2026):
- 5x5 Convolutions for elongated dendritic filament structures
- Instance Normalization (IN) for robust local contrast sensitivity
- LeakyReLU(0.01) + Dropout(0.1)
- Adaptive Pyramid Pooling (APP) for multiscale feature fusion
- U-Net style skip connections with multi-scale branch blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvINLeakyReLU(nn.Module):
    """5x5 Convolution + InstanceNorm2d + LeakyReLU + Dropout"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class DoubleConvBlock(nn.Module):
    """Two consecutive 5x5 Conv + IN + LeakyReLU layers"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvINLeakyReLU(in_ch, out_ch, kernel_size=5, stride=1, dropout=dropout)
        self.conv2 = ConvINLeakyReLU(out_ch, out_ch, kernel_size=5, stride=1, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class AdaptivePyramidPooling(nn.Module):
    """
    Adaptive Pyramid Pooling (APP) module from Hu et al. (2026).
    Performs multiscale pooling across pyramid levels and fuses with 1x1 convolution.
    """

    def __init__(self, in_ch: int, out_ch: int, levels: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.levels = levels
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d((l, l)) for l in levels])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch // len(levels), kernel_size=1, bias=False),
                nn.GroupNorm(1, in_ch // len(levels)),
                nn.LeakyReLU(0.01, inplace=True)
            )
            for _ in levels
        ])
        total_ch = in_ch + (in_ch // len(levels)) * len(levels)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        features = [x]
        for pool, conv in zip(self.pools, self.convs):
            p = pool(x)
            c = conv(p)
            up = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
            features.append(up)
        fused = torch.cat(features, dim=1)
        return self.fuse(fused)


class MORDEN(nn.Module):
    """
    Full MORDEN Architecture for High-Precision Solar Filament Segmentation.
    Encoder-Decoder with 5x5 kernels, Instance Normalization, and multiscale APP fusion.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        c = base_channels

        # Encoder (Downsampling)
        self.inc = DoubleConvBlock(in_channels, c, dropout=dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConvBlock(c, c * 2, dropout=dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConvBlock(c * 2, c * 4, dropout=dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConvBlock(c * 4, c * 8, dropout=dropout))

        # Bottleneck with Adaptive Pyramid Pooling
        self.bottleneck = nn.Sequential(
            DoubleConvBlock(c * 8, c * 16, dropout=dropout),
            AdaptivePyramidPooling(c * 16, c * 8, levels=(1, 2, 4))
        )

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConvBlock(c * 8, c * 4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConvBlock(c * 4, c * 2, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConvBlock(c * 2, c, dropout=dropout)

        # Final prediction head
        self.out_app = AdaptivePyramidPooling(c, c, levels=(1, 2))
        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)         # [B, c, H, W]
        x2 = self.down1(x1)      # [B, 2c, H/2, W/2]
        x3 = self.down2(x2)      # [B, 4c, H/4, W/4]
        x4 = self.down3(x3)      # [B, 8c, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(x4)  # [B, 8c, H/8, W/8]

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)

        out = self.out_app(u3)
        logits = self.out_conv(out)
        return logits
