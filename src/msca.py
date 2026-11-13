"""
Multi-Scale Convolutional Attention (MSCA) Module from SegNeXt.
Reference: Guo et al. (NeurIPS 2022) "SegNeXt: Rethinking Convolutional Attention Design
for Semantic Segmentation" (arXiv:2209.08575).

Uses parallel depth-wise strip convolutions:
- (7x1, 1x7)
- (11x1, 1x11)
- (21x1, 1x21)
Provides long-range curvilinear receptive fields along magnetic field lines.
Can be plugged as an attention bridge into U-Net encoders/decoders.
"""

import torch
import torch.nn as nn


class MultiScaleConvolutionalAttention(nn.Module):
    """
    MSCA block from SegNeXt (NeurIPS 2022).
    """

    def __init__(self, channels: int):
        super().__init__()
        # Initial depth-wise 5x5 convolution for local aggregation
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)

        # Multi-branch depthwise strip convolutions
        # Branch 1: 7x1 and 1x7
        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.conv1_2 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)

        # Branch 2: 11x1 and 1x11
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)

        # Branch 3: 21x1 and 1x21
        self.conv3_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv3_2 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)

        # 1x1 channel-mixing convolution to produce spatial attention weights
        self.conv_squeeze = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)

        # Multi-scale strip convolutions
        branch1 = self.conv1_2(self.conv1_1(attn))
        branch2 = self.conv2_2(self.conv2_1(attn))
        branch3 = self.conv3_2(self.conv3_1(attn))

        # Sum branches and squeeze
        attn = attn + branch1 + branch2 + branch3
        attn = self.conv_squeeze(attn)

        # Modulate input feature map via spatial attention
        out = u * attn
        return out


class MSCABottleneck(nn.Module):
    """
    MSCA block with residual connection and LayerNorm/BatchNorm for network bottlenecks.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.msca = MultiScaleConvolutionalAttention(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.norm(x)
        out = self.msca(out)
        out = self.proj(out)
        return identity + out
