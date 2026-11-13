"""
PyTorch U-Net Model for Solar Filament Foreground Segmentation.
Lightweight, modular, and self-contained architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net with configurable base channels.
    Takes 1-channel grayscale input (1, H, W) and outputs 1-channel logits (1, H, W).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        c = base_channels  # e.g., 32
        self.inc = DoubleConv(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))

        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 8, c * 16))

        # Decoders with bilinear upsampling + skip concatenation
        self.up1 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(c * 16, c * 8)

        self.up2 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(c * 8, c * 4)

        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(c * 4, c * 2)

        self.up4 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(c * 2, c)

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits
