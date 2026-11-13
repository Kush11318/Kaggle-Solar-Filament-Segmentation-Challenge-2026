"""
clDice: Topology-Preserving Loss Function for Tubular & Curvilinear Structures.
Reference: Shit et al. (CVPR 2021) "clDice - a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation" (arXiv:2003.07311).

Implements:
- Soft-skeletonization via iterative morphological min/max pooling (Algorithm 1)
- Differentiable soft-clDice loss (Algorithm 2)
- Hybrid Loss: (1 - alpha) * DiceBCELoss + alpha * (1 - soft_clDice)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """
    Differentiable soft-skeletonization using iterative min- and max-pooling.
    Algorithm 1 from Shit et al. (CVPR 2021).
    """

    def __init__(self, num_iterations: int = 12):
        super().__init__()
        self.num_iterations = num_iterations
        # 3x3 maxpool with stride 1, padding 1 represents morphological dilation
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _min_pool(self, x: torch.Tensor) -> torch.Tensor:
        # Min-pooling via negative max-pooling represents morphological erosion
        return -self.max_pool(-x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Probabilistic prediction or binary mask tensor of shape (B, 1, H, W) in [0, 1].
        Returns soft skeleton tensor of shape (B, 1, H, W) in [0, 1].
        """
        # Step 1: Initial morphological open: dilation(erosion(x))
        x_open = self.max_pool(self._min_pool(x))
        skeleton = F.relu(x - x_open)

        curr = x
        for _ in range(self.num_iterations):
            curr = self._min_pool(curr)
            curr_open = self.max_pool(self._min_pool(curr))
            skeleton = skeleton + (1.0 - skeleton) * F.relu(curr - curr_open)

        return skeleton


def soft_cldice_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    skeleton_fn: SoftSkeletonize,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Computes soft-clDice similarity between predicted probability map and target mask.
    Algorithm 2 from Shit et al. (CVPR 2021).
    """
    # Extract soft skeletons
    s_pred = skeleton_fn(y_pred)
    s_true = skeleton_fn(y_true)

    # Topology precision: fraction of predicted skeleton intersecting ground truth mask
    t_prec = (torch.sum(s_pred * y_true, dim=(-2, -1)) + epsilon) / (
        torch.sum(s_pred, dim=(-2, -1)) + epsilon
    )

    # Topology sensitivity: fraction of ground truth skeleton intersecting predicted mask
    t_sens = (torch.sum(s_true * y_pred, dim=(-2, -1)) + epsilon) / (
        torch.sum(s_true, dim=(-2, -1)) + epsilon
    )

    # Harmonic mean
    cldice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + epsilon)
    return torch.mean(cldice)


class CombinedDiceBCEclDiceLoss(nn.Module):
    """
    Hybrid loss: (1 - alpha) * SoftDiceBCELoss + alpha * (1 - soft_clDice)
    Recommended alpha in [0.1, 0.3] (default: 0.20).
    Recommended iterations k = 12 matching solar filament radii.
    """

    def __init__(
        self,
        alpha: float = 0.20,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        k_iterations: int = 12,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.epsilon = epsilon
        self.skeletonizer = SoftSkeletonize(num_iterations=k_iterations)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) raw unactivated logits
        targets: (B, 1, H, W) float targets in [0, 1]
        """
        probs = torch.sigmoid(logits)

        # 1. BCE Loss
        bce_loss = self.bce(logits, targets)

        # 2. Soft Dice Loss
        intersection = torch.sum(probs * targets, dim=(-2, -1))
        cardinality = torch.sum(probs + targets, dim=(-2, -1))
        dice_score = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)
        dice_loss = 1.0 - torch.mean(dice_score)

        standard_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        # 3. Soft-clDice Loss
        cldice = soft_cldice_score(probs, targets, self.skeletonizer, epsilon=self.epsilon)
        cldice_loss = 1.0 - cldice

        # Combined Loss
        total_loss = (1.0 - self.alpha) * standard_loss + self.alpha * cldice_loss
        return total_loss
