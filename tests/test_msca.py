import sys
import os
sys.path.append(r"d:/Kaggle")
import torch
from src.msca import MultiScaleConvolutionalAttention, MSCABottleneck


def test_msca_forward_and_backward():
    C = 64
    B, H, W = 2, 32, 32
    block = MSCABottleneck(channels=C)
    x = torch.randn((B, C, H, W), requires_grad=True)

    out = block(x)
    assert out.shape == (B, C, H, W), f"Shape mismatch: {out.shape}"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "Gradient has NaN!"
    print("test_msca_forward_and_backward PASSED!")


if __name__ == "__main__":
    test_msca_forward_and_backward()
    print("\nALL MSCA TESTS PASSED 100%!")
