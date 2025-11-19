#!/usr/bin/env python3
"""
Test that the variance fix resolves the uncovered pixel issue.
"""

from math import exp, sqrt, pi

def gaussian_edge_weight(tile_size, overlap, var):
    """Calculate edge weight with given variance"""
    tile_w = tile_h = tile_size
    x_midpoint = (tile_w - 1) / 2
    y_midpoint = (tile_h - 1) / 2  # FIXED: was tile_h / 2

    f = lambda x, midpoint: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)

    x_prob = f(overlap, x_midpoint)
    y_prob = f(overlap, y_midpoint)

    return x_prob * y_prob

print("="*80)
print("VARIANCE FIX VALIDATION")
print("="*80)

overlap = 6
threshold = 1e-6

# Test the problematic tile size (64x64 core → 76x76 with overlap)
tile_size = 76

print(f"\nTile size: {tile_size}×{tile_size} (64×64 core + overlap={overlap})")
print(f"Threshold: {threshold:.0e}")
print()

# Before fix
var_old = 0.01
weight_old = gaussian_edge_weight(tile_size, overlap, var_old)
print(f"BEFORE FIX (var={var_old}):")
print(f"  Edge weight: {weight_old:.2e}")
print(f"  Status: {'✗ FAIL - UNCOVERED' if weight_old < threshold else '✓ PASS'}")

# After fix
var_new = 0.02
weight_new = gaussian_edge_weight(tile_size, overlap, var_new)
print(f"\nAFTER FIX (var={var_new}):")
print(f"  Edge weight: {weight_new:.2e}")
print(f"  Status: {'✗ FAIL - UNCOVERED' if weight_new < threshold else '✓ PASS - COVERED'}")

print(f"\nImprovement: {weight_new / weight_old:.1f}x increase in edge weight")

# Test all common tile sizes
print("\n" + "="*80)
print("COMPREHENSIVE TEST: All Common Tile Sizes")
print("="*80)

core_sizes = [56, 64, 72, 80, 96, 128, 192, 256]
all_pass = True

for core_size in core_sizes:
    tile_size = core_size + 2 * overlap
    weight = gaussian_edge_weight(tile_size, overlap, var_new)
    status = "✓" if weight >= threshold else "✗"
    print(f"Core {core_size}×{core_size} → Tile {tile_size}×{tile_size}: {weight:.2e} {status}")

    if weight < threshold:
        all_pass = False

print("\n" + "="*80)
if all_pass:
    print("✓✓✓ ALL TESTS PASSED! Fix resolves the uncovered pixel issue.")
else:
    print("✗✗✗ SOME TESTS FAILED! May need higher variance.")
print("="*80)
