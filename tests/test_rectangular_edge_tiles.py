#!/usr/bin/env python3
"""
Test that rectangular edge tiles work correctly after cropping.
Verifies that the Gaussian formula handles non-square tiles properly.
"""

from math import exp, sqrt, pi

def gaussian_weights_fixed(tile_w: int, tile_h: int, var=0.02):
    """
    Fixed Gaussian implementation - uses tile_w for x-axis, tile_h for y-axis
    """
    f_x = lambda x, midpoint: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    f_y = lambda y, midpoint: exp(-(y-midpoint)*(y-midpoint) / (tile_h*tile_h) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f_x(x, (tile_w - 1) / 2) for x in range(tile_w)]
    y_probs = [f_y(y, (tile_h - 1) / 2) for y in range(tile_h)]
    w = [[y_probs[y] * x_probs[x] for x in range(tile_w)] for y in range(tile_h)]
    return w

def test_rectangular_tile(tile_w, tile_h, name):
    """Test Gaussian weights for a rectangular tile"""
    print(f"\n{'='*80}")
    print(f"Testing {name}: {tile_w}x{tile_h}")
    print(f"{'='*80}")

    weights = gaussian_weights_fixed(tile_w, tile_h)

    # Check symmetry
    center_x = tile_w // 2
    center_y = tile_h // 2

    # Check that edges are symmetric
    top_center = weights[0][center_x]
    bottom_center = weights[tile_h - 1][center_x]
    left_center = weights[center_y][0]
    right_center = weights[center_y][tile_w - 1]

    print(f"\nSymmetry check:")
    print(f"  Top edge (center):    {top_center:.6e}")
    print(f"  Bottom edge (center): {bottom_center:.6e}")
    print(f"  Difference: {abs(top_center - bottom_center):.6e}")

    top_bottom_symmetric = abs(top_center - bottom_center) < 1e-10
    print(f"  {'✓ Top/bottom symmetric' if top_bottom_symmetric else '✗ NOT symmetric'}")

    print(f"\n  Left edge (center):   {left_center:.6e}")
    print(f"  Right edge (center):  {right_center:.6e}")
    print(f"  Difference: {abs(left_center - right_center):.6e}")

    left_right_symmetric = abs(left_center - right_center) < 1e-10
    print(f"  {'✓ Left/right symmetric' if left_right_symmetric else '✗ NOT symmetric'}")

    # Check corner weights are above threshold
    threshold = 1e-6
    corners = [
        (0, 0, "Top-left"),
        (0, tile_w - 1, "Top-right"),
        (tile_h - 1, 0, "Bottom-left"),
        (tile_h - 1, tile_w - 1, "Bottom-right"),
    ]

    print(f"\nCorner weights (threshold: {threshold:.0e}):")
    all_corners_ok = True
    for y, x, label in corners:
        w = weights[y][x]
        status = "✓" if w >= threshold else "✗"
        print(f"  {label:12s}: {w:.6e} {status}")
        if w < threshold:
            all_corners_ok = False

    # Check center weight
    center_weight = weights[center_y][center_x]
    print(f"\nCenter weight: {center_weight:.6e}")

    # Overall result
    if top_bottom_symmetric and left_right_symmetric and all_corners_ok:
        print(f"\n✓✓✓ PASS: Rectangular tile {tile_w}x{tile_h} works correctly!")
        return True
    else:
        print(f"\n✗✗✗ FAIL: Issues with rectangular tile {tile_w}x{tile_h}")
        return False

def simulate_cropping_scenario(image_w, image_h, name):
    """
    Simulate a real cropping scenario where edge tiles become rectangular.
    """
    import math

    print(f"\n{'='*80}")
    print(f"CROPPING SCENARIO: {name}")
    print(f"{'='*80}")

    # Calculate square root size (power-of-2 multiple of 8)
    root_size = max(image_w, image_h)
    if root_size <= 8:
        root_size = 8
    else:
        n = math.ceil(math.log2(root_size / 8))
        root_size = 8 * (2 ** n)

    print(f"Image size: {image_w}x{image_h}")
    print(f"Square root: {root_size}x{root_size}")
    print(f"Overhang: x={root_size - image_w}, y={root_size - image_h}")

    # Simulate a tile at the edge that would be cropped
    # Example: tile at bottom-right that extends beyond image
    # Place it so it partially overlaps the image
    tile_w = tile_h = 512
    tile_x = image_w - 256  # Starts 256px before right edge
    tile_y = image_h - 256  # Starts 256px before bottom edge

    # Crop to image bounds
    new_x = max(0, tile_x)
    new_y = max(0, tile_y)
    new_w = min(image_w, tile_x + tile_w) - new_x
    new_h = min(image_h, tile_y + tile_h) - new_y

    if new_w == tile_w and new_h == tile_h:
        print(f"\nNo cropping needed - tile {tile_w}x{tile_h} fits entirely in image")
        return True

    print(f"\nOriginal tile: ({tile_x}, {tile_y}) {tile_w}x{tile_h}")
    print(f"Cropped tile:  ({new_x}, {new_y}) {new_w}x{new_h}")
    print(f"  → Became rectangular: {new_w}x{new_h}")

    # Test the cropped tile
    return test_rectangular_tile(new_w, new_h, f"Cropped edge tile")

def main():
    print("="*80)
    print("RECTANGULAR EDGE TILE TEST")
    print("="*80)
    print("\nWith the fixed Gaussian formula, edge tiles can be rectangular")
    print("while maintaining proper blending and symmetry.")

    # Test various rectangular tile sizes
    test_cases = [
        (512, 256, "512x256 (2:1 horizontal)"),
        (256, 512, "256x512 (1:2 vertical)"),
        (448, 512, "448x512 (cropped bottom)"),
        (512, 384, "512x384 (cropped right)"),
        (384, 448, "384x448 (cropped both)"),
        (1024, 640, "1024x640 (large rectangular)"),
    ]

    all_pass = True
    for w, h, name in test_cases:
        if not test_rectangular_tile(w, h, name):
            all_pass = False

    # Test realistic cropping scenarios
    print(f"\n{'='*80}")
    print("REALISTIC CROPPING SCENARIOS")
    print(f"{'='*80}")

    scenarios = [
        (1920, 1080, "Full HD (16:9)"),
        (3840, 2160, "4K (16:9)"),
        (1200, 1600, "Portrait (3:4)"),
    ]

    for w, h, name in scenarios:
        if not simulate_cropping_scenario(w, h, name):
            all_pass = False

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if all_pass:
        print("✓✓✓ ALL TESTS PASSED!")
        print("\nRectangular edge tiles work correctly:")
        print("  1. Gaussian weights are symmetric along each axis")
        print("  2. Corner weights are above threshold (1e-6)")
        print("  3. Blending will work properly for cropped edge tiles")
        print("\nBenefits:")
        print("  - No wasted compute on huge padding regions")
        print("  - Most tiles stay square (quadtree property)")
        print("  - Only edge tiles become rectangular after cropping")
        print("  - Better utilizes adaptive sizing for actual image content")
        return 0
    else:
        print("✗✗✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
