"""
STAPLE: Simultaneous Truth and Performance Level Estimation.
Reference: Warfield et al. (IEEE TMI 2004) "Simultaneous Truth and Performance Level
Estimation (STAPLE): An Algorithm for the Validation of Image Segmentation".

Used to resolve multi-annotator drawing conflicts across physical solar observations
into optimal, continuous consensus ground-truth probability maps W in [0, 1].
"""

import numpy as np
from typing import List, Tuple, Dict, Any


def compute_staple_consensus(
    masks: List[np.ndarray],
    gamma: float = 0.002,
    p_init: float = 0.999,
    q_init: float = 0.999,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes STAPLE consensus truth across R multi-annotator binary masks.
    
    Args:
        masks: List of R binary boolean or uint8 2D masks of same shape (H, W).
        gamma: Foreground class prior probability (default: 0.002 matching ~0.2% solar filament ratio).
        p_init: Initial sensitivity for all annotators (default: 0.999).
        q_init: Initial specificity for all annotators (default: 0.999).
        max_iter: Maximum EM iterations (typically converges in < 10).
        tol: Convergence tolerance on parameter change.
        
    Returns:
        W: (H, W) float32 array in [0, 1] representing the posterior probability of genuine filament truth.
        p: (R,) sensitivity parameter for each rater.
        q: (R,) specificity parameter for each rater.
    """
    R = len(masks)
    if R == 0:
        raise ValueError("Must provide at least one segmentation mask!")
    if R == 1:
        # Trivial case: single annotator
        return masks[0].astype(np.float32), np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    H, W_dim = masks[0].shape
    N = H * W_dim

    # Flatten masks into (R, N) boolean matrix
    D = np.stack([m.ravel().astype(bool) for m in masks], axis=0) # (R, N)

    # Initialize sensitivity (p) and specificity (q)
    p = np.full(R, p_init, dtype=np.float64)
    q = np.full(R, q_init, dtype=np.float64)

    # Estimate prior dynamically if gamma is None
    if gamma is None:
        gamma = float(np.mean(D))
        gamma = max(1e-5, min(1.0 - 1e-5, gamma))

    W = np.zeros(N, dtype=np.float64)

    for iteration in range(max_iter):
        # 1. E-Step: Compute conditional probability W_i = P(T_i = 1 | D_i, p, q)
        # Using log-space for numerical stability against underflow
        # a_i = log(gamma) + sum_j [ D_ij * log(p_j) + (1 - D_ij) * log(1 - p_j) ]
        # b_i = log(1 - gamma) + sum_j [ (1 - D_ij) * log(q_j) + D_ij * log(1 - q_j) ]

        p_clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
        q_clipped = np.clip(q, 1e-7, 1.0 - 1e-7)

        log_p = np.log(p_clipped)
        log_1_minus_p = np.log(1.0 - p_clipped)
        log_q = np.log(q_clipped)
        log_1_minus_q = np.log(1.0 - q_clipped)

        # Vectorized over all N pixels:
        # D is (R, N)
        term_true = np.sum(np.where(D, log_p[:, None], log_1_minus_p[:, None]), axis=0)
        term_false = np.sum(np.where(~D, log_q[:, None], log_1_minus_q[:, None]), axis=0)

        log_a = np.log(gamma) + term_true
        log_b = np.log(1.0 - gamma) + term_false

        # W_i = 1 / (1 + exp(log_b - log_a))
        diff = np.clip(log_b - log_a, -50.0, 50.0)
        W_new = 1.0 / (1.0 + np.exp(diff))

        # 2. M-Step: Maximize parameters p and q
        sum_W = np.sum(W_new)
        sum_1_minus_W = np.sum(1.0 - W_new)

        if sum_W < 1e-8 or sum_1_minus_W < 1e-8:
            break

        p_new = np.sum(np.where(D, W_new[None, :], 0.0), axis=1) / sum_W
        q_new = np.sum(np.where(~D, (1.0 - W_new)[None, :], 0.0), axis=1) / sum_1_minus_W

        # Check convergence: normalized trace difference
        delta = np.max(np.abs(p_new - p)) + np.max(np.abs(q_new - q))
        p = p_new
        q = q_new
        W = W_new

        if delta < tol:
            break

    W_2d = W.reshape((H, W_dim)).astype(np.float32)
    return W_2d, p.astype(np.float32), q.astype(np.float32)
