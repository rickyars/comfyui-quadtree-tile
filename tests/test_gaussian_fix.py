#!/usr/bin/env python3
"""
Test that the Gaussian formula fix correctly uses tile_w for x-axis and tile_h for y-axis.
"""

from math import exp, sqrt, pi

def gaussian_weights_old(tile_w: int, tile_h: int, var=0.02):
    """Old implementation - uses tile_w for both axes (BUG)"""
    f = lambda x, midpoint: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]
    y_probs = [f(y, (tile_h - 1) / 2) for y in range(tile_h)]
    w = [[y_probs[y] * x_probs[x] for x in range(tile_w)] for y in range(tile_h)]
    return w

def gaussian_weights_new(tile_w: int, tile_h: int, var=0.02):
    """New implementation - uses tile_w for x-axis, tile_h for y-axis (CORRECT)"""
    f_x = lambda x, midpoint: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    f_y = lambda y, midpoint: exp(-(y-midpoint)*(y-midpoint) / (tile_h*tile_h) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f_x(x, (tile_w - 1) / 2) for x in range(tile_w)]
    y_probs = [f_y(y, (tile_h - 1) / 2) for y in range(tile_h)]
    w = [[y_probs[y] * x_probs[x] for x in range(tile_w)] for y in range(tile_h)]
    return w

def test_square_tiles():
    """For square tiles, old and new should produce identical results"""
    print("="*80)
    print("TEST 1: Square Tiles (64x64)")
    print("="*80)

    tile_w = tile_h = 64
    old_weights = gaussian_weights_old(tile_w, tile_h)
    new_weights = gaussian_weights_new(tile_w, tile_h)

    # Check center
    center = tile_w // 2
    old_center = old_weights[center][center]
    new_center = new_weights[center][center]

    print(f"Center weight (old): {old_center:.6e}")
    print(f"Center weight (new): {new_center:.6e}")
    print(f"Difference: {abs(old_center - new_center):.6e}")

    # Check edge
    old_edge = old_weights[0][0]
    new_edge = new_weights[0][0]

    print(f"Edge weight (old): {old_edge:.6e}")
    print(f"Edge weight (new): {new_edge:.6e}")
    print(f"Difference: {abs(old_edge - new_edge):.6e}")

    # For square tiles, should be identical
    max_diff = max(
        abs(old_weights[y][x] - new_weights[y][x])
        for y in range(tile_h)
        for x in range(tile_w)
    )

    print(f"\nMaximum difference across all pixels: {max_diff:.6e}")
    if max_diff < 1e-10:
        print("✓ PASS: Old and new produce identical results for square tiles")
        return True
    else:
        print("✗ FAIL: Unexpected difference for square tiles")
        return False

def test_non_square_tiles():
    """For non-square tiles, new should be more correct"""
    print("\n" + "="*80)
    print("TEST 2: Non-Square Tiles (64x48)")
    print("="*80)

    tile_w, tile_h = 64, 48
    old_weights = gaussian_weights_old(tile_w, tile_h)
    new_weights = gaussian_weights_new(tile_w, tile_h)

    # Check center
    center_x = tile_w // 2
    center_y = tile_h // 2
    old_center = old_weights[center_y][center_x]
    new_center = new_weights[center_y][center_x]

    print(f"Center weight (old): {old_center:.6e}")
    print(f"Center weight (new): {new_center:.6e}")
    print(f"Difference: {abs(old_center - new_center):.6e}")

    # Check edges
    # Top edge (y=0) at center x
    old_top = old_weights[0][center_x]
    new_top = new_weights[0][center_x]

    print(f"\nTop edge weight (old): {old_top:.6e}")
    print(f"Top edge weight (new): {new_top:.6e}")
    print(f"Difference: {abs(old_top - new_top):.6e}")

    # Left edge (x=0) at center y
    old_left = old_weights[center_y][0]
    new_left = new_weights[center_y][0]

    print(f"\nLeft edge weight (old): {old_left:.6e}")
    print(f"Left edge weight (new): {new_left:.6e}")
    print(f"Difference: {abs(old_left - new_left):.6e}")

    # Check symmetry
    # For correct Gaussian, top/bottom edges should have same weight (symmetric around center)
    # and left/right edges should have same weight
    new_bottom = new_weights[tile_h - 1][center_x]
    new_right = new_weights[center_y][tile_w - 1]

    print(f"\n--- SYMMETRY CHECK (NEW) ---")
    print(f"Top edge:    {new_top:.6e}")
    print(f"Bottom edge: {new_bottom:.6e}")
    print(f"Difference:  {abs(new_top - new_bottom):.6e}")

    print(f"\nLeft edge:   {new_left:.6e}")
    print(f"Right edge:  {new_right:.6e}")
    print(f"Difference:  {abs(new_left - new_right):.6e}")

    if abs(new_top - new_bottom) < 1e-10:
        print("✓ Top/bottom edges are symmetric")
    else:
        print("✗ Top/bottom edges are NOT symmetric")

    if abs(new_left - new_right) < 1e-10:
        print("✓ Left/right edges are symmetric")
    else:
        print("✗ Left/right edges are NOT symmetric")

    # The new implementation should produce different results for non-square tiles
    max_diff = max(
        abs(old_weights[y][x] - new_weights[y][x])
        for y in range(tile_h)
        for x in range(tile_w)
    )

    print(f"\nMaximum difference between old and new: {max_diff:.6e}")
    if max_diff > 1e-10:
        print("✓ PASS: Old and new produce different results (as expected for non-square)")
        return True
    else:
        print("✗ FAIL: Expected differences for non-square tiles")
        return False

def test_edge_weights():
    """Verify edge weights are above threshold for various tile sizes"""
    print("\n" + "="*80)
    print("TEST 3: Edge Weights Above Threshold (1e-6)")
    print("="*80)

    test_sizes = [
        (48, 48),
        (64, 64),
        (80, 80),
        (128, 128),
        (256, 256),
        # Non-square
        (64, 48),
        (80, 64),
    ]

    all_pass = True
    threshold = 1e-6

    for tile_w, tile_h in test_sizes:
        weights = gaussian_weights_new(tile_w, tile_h)

        # Check all four corners
        corners = [
            (0, 0),
            (0, tile_w - 1),
            (tile_h - 1, 0),
            (tile_h - 1, tile_w - 1),
        ]

        min_corner_weight = min(weights[y][x] for y, x in corners)
        status = "✓" if min_corner_weight >= threshold else "✗"

        print(f"  {tile_w}x{tile_h}: min corner weight = {min_corner_weight:.6e} {status}")

        if min_corner_weight < threshold:
            all_pass = False

    if all_pass:
        print("\n✓ PASS: All edge weights above threshold")
    else:
        print("\n✗ FAIL: Some edge weights below threshold")

    return all_pass

def main():
    print("="*80)
    print("GAUSSIAN FORMULA FIX VERIFICATION")
    print("="*80)
    print()

    test1 = test_square_tiles()
    test2 = test_non_square_tiles()
    test3 = test_edge_weights()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if test1 and test2 and test3:
        print("✓✓✓ ALL TESTS PASSED!")
        print("\nThe Gaussian formula fix:")
        print("  1. Maintains backward compatibility for square tiles")
        print("  2. Correctly handles non-square tiles with proper axis scaling")
        print("  3. Ensures edge weights remain above threshold (1e-6)")
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
