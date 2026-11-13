import sys
import os
sys.path.append(r"d:/Kaggle")
import numpy as np
from src.staple import compute_staple_consensus


def test_staple_consensus_agreement():
    H, W = 100, 100
    m1 = np.zeros((H, W), dtype=bool)
    m2 = np.zeros((H, W), dtype=bool)
    m3 = np.zeros((H, W), dtype=bool)

    # All 3 agree on central core [40:60, 40:60]
    m1[40:60, 40:60] = True
    m2[40:60, 40:60] = True
    m3[40:60, 40:60] = True

    # Only rater 1 draws a noisy barb at [10:20, 10:20]
    m1[10:20, 10:20] = True

    W_map, p, q = compute_staple_consensus([m1, m2, m3], gamma=0.002)

    # Core consensus should be very high
    assert np.mean(W_map[40:60, 40:60]) > 0.95, "Core consensus failed!"
    
    # Isolated noisy barb from 1 rater should be suppressed
    assert np.mean(W_map[10:20, 10:20]) < 0.10, "Isolated noise was not suppressed!"

    print(f"Rater Sensitivities (p): {p}")
    print(f"Rater Specificities (q): {q}")
    # Rater 1 should have lower specificity because of the false barb
    assert q[0] < q[1], "Rater 1 specificity should be lower than Rater 2!"
    print("test_staple_consensus_agreement PASSED!")


def test_staple_single_rater():
    m = np.zeros((50, 50), dtype=bool)
    m[10:20, 10:20] = True
    W_map, p, q = compute_staple_consensus([m])
    assert np.array_equal(W_map, m.astype(np.float32))
    print("test_staple_single_rater PASSED!")


if __name__ == "__main__":
    test_staple_consensus_agreement()
    test_staple_single_rater()
    print("\nALL STAPLE TESTS PASSED 100%!")
