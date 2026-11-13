"""
EdgeAttNet: Towards Barb-Aware Filament Segmentation
Paper: arXiv:2509.02964 (Solomon, Martens, Liu, Angryk - Georgia State University)
Architecture: U-Net Backbone + Edge-Guided Multi-Head Self-Attention (EG-MHSA) Bottleneck
"""

import math
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two sequential 3x3 Conv-BatchNorm-ReLU blocks."""

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


class EGMHSA(nn.Module):
    """
    Edge-Guided Multi-Head Self-Attention (EG-MHSA).
    Injects a learned structural edge prior E' into Query and Key matrices:
        Q = (X + E') W_Q
        K = (X + E') W_K
        V = X W_V
        Attention = Softmax(Q K^T / sqrt(d_k)) V
    Omits positional encodings since edge priors encode explicit spatial geometry.
    """

    def __init__(self, dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, N, C) where N = H' * W', C = 512
            edge_prior: Transformed edge prior tensor of shape (B, N, C)
        Returns:
            Attention-enhanced feature tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Residual connection
        residual = x

        # Edge-guided Query and Key inputs
        guided_in = x + edge_prior

        # Linear projections
        q = self.q_proj(guided_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(guided_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention: (B, num_heads, N, N)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Context aggregation
        out = torch.matmul(attn_probs, v)  # (B, num_heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.dropout(out)

        # LayerNorm after residual
        return self.norm(residual + out)


class EdgeAttNet(nn.Module):
    """
    EdgeAttNet Architecture for Barb-Aware Solar Filament Segmentation.
    4-Stage U-Net Encoder-Decoder with stacked EG-MHSA modules at the bottleneck.
    Trainable parameters: ~15.4 Million.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 1):
        super().__init__()

        # --- 1. Edge Prior Branch ---
        # 3x3 Conv predicting edge map logits from 2-channel input
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # 1x1 Conv projecting downsampled edge map to bottleneck channels (512)
        self.edge_proj = nn.Conv2d(1, 512, kernel_size=1, bias=False)

        # --- 2. Encoder Path ---
        # Stage 1: (B, 64, H, W)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2: (B, 128, H/2, W/2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3: (B, 256, H/4, W/4)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Stage 4: (B, 512, H/8, W/8)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # --- 3. Bottleneck Path (H/16, W/16) ---
        self.bottleneck_conv = DoubleConv(512, 512)
        # Two stacked EG-MHSA modules
        self.eg_mhsa1 = EGMHSA(dim=512, num_heads=4, dropout=0.1)
        self.eg_mhsa2 = EGMHSA(dim=512, num_heads=4, dropout=0.1)

        # --- 4. Decoder Path ---
        # Stage 4 Up: 512 -> 256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512 + 256, 256)  # concat with enc4 (512)

        # Stage 3 Up: 256 -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256 + 128, 128)  # concat with enc3 (256)

        # Stage 2 Up: 128 -> 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128 + 64, 64)   # concat with enc2 (128)

        # Stage 1 Up: 64 -> 64
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)    # concat with enc1 (64)

        # Final 1x1 Conv
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def extract_edge_prior(self, x: torch.Tensor, target_h: int, target_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes learned edge prior and projects to bottleneck dimensions."""
        e_logits = self.edge_conv(x)  # (B, 1, H, W)
        e = torch.sigmoid(e_logits)   # Normalized edge activation
        e_down = F.interpolate(e, size=(target_h, target_w), mode="bilinear", align_corners=False)
        e_proj = self.edge_proj(e_down)  # (B, 512, H/16, W/16)
        B, C, H, W = e_proj.shape
        return e_proj.flatten(2).transpose(1, 2), e_logits

    def forward(self, x: torch.Tensor, return_edges: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, _, H, W = x.shape

        # 1. Encoder Forward
        e1 = self.enc1(x)       # (B, 64, H, W)
        p1 = self.pool1(e1)     # (B, 64, H/2, W/2)

        e2 = self.enc2(p1)      # (B, 128, H/2, W/2)
        p2 = self.pool2(e2)     # (B, 128, H/4, W/4)

        e3 = self.enc3(p2)      # (B, 256, H/4, W/4)
        p3 = self.pool3(e3)     # (B, 256, H/8, W/8)

        e4 = self.enc4(p3)      # (B, 512, H/8, W/8)
        p4 = self.pool4(e4)     # (B, 512, H/16, W/16)

        # 2. Bottleneck with EG-MHSA
        b = self.bottleneck_conv(p4)  # (B, 512, H/16, W/16)
        _, _, bH, bW = b.shape

        # Extract edge prior at bottleneck scale
        edge_prior, e_logits = self.extract_edge_prior(x, bH, bW)  # (B, bH*bW, 512), (B, 1, H, W)

        # Flatten bottleneck features for transformer
        b_flat = b.flatten(2).transpose(1, 2)  # (B, bH*bW, 512)

        # Apply two stacked EG-MHSA modules
        b_attn = self.eg_mhsa1(b_flat, edge_prior)
        b_attn = self.eg_mhsa2(b_attn, edge_prior)

        # Reshape back to 2D feature map
        b_out = b_attn.transpose(1, 2).reshape(B, 512, bH, bW)

        # 3. Decoder Forward with Skip Connections
        d4 = self.up4(b_out)                            # (B, 256, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)                 # (B, 256+512, H/8, W/8)
        d4 = self.dec4(d4)                              # (B, 256, H/8, W/8)

        d3 = self.up3(d4)                               # (B, 128, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)                 # (B, 128+256, H/4, W/4)
        d3 = self.dec3(d3)                              # (B, 128, H/4, W/4)

        d2 = self.up2(d3)                               # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)                 # (B, 64+128, H/2, W/2)
        d2 = self.dec2(d2)                              # (B, 64, H/2, W/2)

        d1 = self.up1(d2)                               # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)                 # (B, 64+64, H, W)
        d1 = self.dec1(d1)                              # (B, 64, H, W)

        # 4. Final Logits
        logits = self.final_conv(d1)                    # (B, num_classes, H, W)

        if return_edges:
            return logits, e_logits
        return logits


if __name__ == "__main__":
    model = EdgeAttNet(in_channels=2, num_classes=1)
    x = torch.randn(2, 2, 512, 512)
    logits, edges = model(x, return_edges=True)
    print("EdgeAttNet with edges:", logits.shape, edges.shape)
