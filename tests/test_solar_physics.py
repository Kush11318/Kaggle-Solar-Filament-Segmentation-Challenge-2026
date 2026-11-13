import sys
import os
sys.path.append(r"d:/Kaggle")
import numpy as np
import time
from src.solar_physics import fit_and_subtract_polynomial_surface, filter_sunspots_with_safety_rule


def test_polynomial_surface_flattening():
    H, W = 512, 512
    Y, X = np.ogrid[:H, :W]
    center = (256, 256)
    radius = 200
    dist_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    disk_mask = dist_sq <= (radius ** 2)

    # Synthetic parabolic limb darkening background: brighter in center, dimmer near edges
    synthetic_background = 200.0 - 100.0 * (dist_sq / (radius ** 2))
    # Add a dark filament line
    img = synthetic_background.copy()
    img[200:210, 200:300] -= 50.0 # filament absorption

    t0 = time.time()
    flattened = fit_and_subtract_polynomial_surface(img, disk_mask, order=4, downsample_factor=2)
    dt = time.time() - t0
    print(f"Surface fitting runtime: {dt:.3f}s")

    # The background should now be flattened near 0.0, while filament stands out as negative
    bg_std_before = np.std(img[disk_mask])
    bg_std_after = np.std(flattened[disk_mask])
    print(f"Disk Std Before: {bg_std_before:.2f}, After Flattening: {bg_std_after:.2f}")
    assert bg_std_after < bg_std_before * 0.5, "Background was not flattened!"
    print("test_polynomial_surface_flattening PASSED!")


def test_sunspot_safety_rule():
    H, W = 200, 200
    disk_mask = np.ones((H, W), dtype=bool)
    img = np.full((H, W), 100.0, dtype=np.float32)

    # 1. Small sunspot: 50 pixels area (dark)
    img[20:25, 20:30] = 10.0 # area = 50

    # 2. Giant filament core: 2,500 pixels area (dark)
    img[100:150, 100:150] = 10.0 # area = 2,500

    spot_mask, cleaned = filter_sunspots_with_safety_rule(img, disk_mask, spot_seed_percentile=5.0, max_sunspot_area=2000)

    # Small spot should be masked
    assert spot_mask[22, 25] == True, "Small spot was not detected as sunspot!"

    # Giant filament core should be PRESERVED (safety rule)
    assert spot_mask[120, 120] == False, "Giant filament core was wrongly masked out!"
    print("test_sunspot_safety_rule PASSED!")


if __name__ == "__main__":
    test_polynomial_surface_flattening()
    test_sunspot_safety_rule()
    print("\nALL SOLAR PHYSICS TESTS PASSED 100%!")
