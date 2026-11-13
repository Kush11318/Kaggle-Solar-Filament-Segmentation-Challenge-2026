import sys
import os
sys.path.append(r"d:/Kaggle")
import torch
import numpy as np
from src.losses import SoftSkeletonize, CombinedDiceBCEclDiceLoss, soft_cldice_score


def test_soft_skeletonize_shape():
    skeletonizer = SoftSkeletonize(num_iterations=8)
    x = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
    # Draw a line
    x[:, :, 20:44, 30:34] = 1.0

    skel = skeletonizer(x)
    assert skel.shape == (2, 1, 64, 64)
    assert skel.min() >= 0.0
    assert skel.max() <= 1.0
    print("test_soft_skeletonize_shape PASSED!")


def test_cldice_gradient_and_loss():
    loss_fn = CombinedDiceBCEclDiceLoss(alpha=0.2, k_iterations=6)
    logits = torch.randn((2, 1, 64, 64), requires_grad=True)
    targets = torch.zeros((2, 1, 64, 64), dtype=torch.float32)
    targets[:, :, 20:44, 30:34] = 1.0

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any(), "Gradient contains NaN!"
    print("test_cldice_gradient_and_loss PASSED!")


def test_cldice_penalizes_breaks():
    """Verify that breaking a continuous tube reduces clDice score significantly."""
    skeletonizer = SoftSkeletonize(num_iterations=6)
    
    # Ground truth: continuous vertical stripe
    gt = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    gt[:, :, 10:54, 30:34] = 1.0

    # Prediction 1: Perfect match
    pred_perfect = gt.clone()
    score_perfect = soft_cldice_score(pred_perfect, gt, skeletonizer).item()

    # Prediction 2: Broken in the middle (gap of 6 pixels)
    pred_broken = gt.clone()
    pred_broken[:, :, 29:35, 30:34] = 0.0
    score_broken = soft_cldice_score(pred_broken, gt, skeletonizer).item()

    print(f"Perfect clDice score: {score_perfect:.4f}")
    print(f"Broken clDice score:  {score_broken:.4f}")
    assert score_perfect > score_broken + 0.05, "clDice failed to heavily penalize broken spine!"
    print("test_cldice_penalizes_breaks PASSED!")


if __name__ == "__main__":
    test_soft_skeletonize_shape()
    test_cldice_gradient_and_loss()
    test_cldice_penalizes_breaks()
    print("\nALL clDice UNIT TESTS PASSED 100%!")
