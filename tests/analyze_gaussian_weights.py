#!/usr/bin/env python3
"""
Analyze Gaussian weights for different tile sizes with overlap=6.
Determine which tile sizes result in edge weights below 1e-6.
"""

from math import exp, sqrt, pi

def calculate_edge_weight(tile_size, overlap, var=0.01):
    """
    Calculate the Gaussian weight at the edge of the overlap region.

    For a tile of size (tile_size, tile_size) with overlap on each side:
    - The overlap region starts at index `overlap`
    - The center is at index (tile_size-1)/2 for x, tile_size/2 for y
    - We want the weight at index `overlap`
    """
    tile_w = tile_h = tile_size

    # Midpoints (from tiled_diffusion.py)
    x_midpoint = (tile_w - 1) / 2
    y_midpoint = tile_h / 2

    # Gaussian function
    f = lambda x, midpoint: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)

    # Calculate probability at the edge of overlap region
    x_prob = f(overlap, x_midpoint)
    y_prob = f(overlap, y_midpoint)

    # Weight is the product (outer product)
    edge_weight = x_prob * y_prob

    return edge_weight, x_prob, y_prob

print("="*90)
print("GAUSSIAN WEIGHT ANALYSIS: Edge Weights for Different Tile Sizes")
print("="*90)
print(f"\nOverlap: 6 latent pixels")
print(f"Variance: 0.01 (from gaussian_weights function)")
print(f"Threshold: 1e-6 (from validation check)")
print("\n" + "-"*90)
print(f"{'Tile Size':<12} {'Core Size':<12} {'Edge Weight':<15} {'Status':<20} {'Distance to Center'}")
print("-"*90)

overlap = 6
threshold = 1e-6

# Test various tile sizes (core + 2*overlap)
core_sizes = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128, 192, 256]

failed_sizes = []
passed_sizes = []

for core_size in core_sizes:
    tile_size = core_size + 2 * overlap
    edge_weight, x_prob, y_prob = calculate_edge_weight(tile_size, overlap)

    status = "✓ PASS" if edge_weight >= threshold else "✗ FAIL (UNCOVERED!)"
    if edge_weight >= threshold:
        passed_sizes.append((core_size, tile_size, edge_weight))
    else:
        failed_sizes.append((core_size, tile_size, edge_weight))

    # Calculate distance from edge to center
    center_x = (tile_size - 1) / 2
    distance = center_x - overlap

    print(f"{tile_size:<12} {core_size:<12} {edge_weight:<15.2e} {status:<20} {distance:.1f} pixels")

print("-"*90)
print(f"\nRESULTS:")
print(f"  Tile sizes that PASS (edge weight >= 1e-6): {len(passed_sizes)}")
for core, tile, weight in passed_sizes[:5]:
    print(f"    Core {core}×{core} → Tile {tile}×{tile}: weight = {weight:.2e}")
if len(passed_sizes) > 5:
    print(f"    ... and {len(passed_sizes) - 5} more")

print(f"\n  Tile sizes that FAIL (edge weight < 1e-6): {len(failed_sizes)}")
for core, tile, weight in failed_sizes[:10]:
    print(f"    Core {core}×{core} → Tile {tile}×{tile}: weight = {weight:.2e} ❌")
if len(failed_sizes) > 10:
    print(f"    ... and {len(failed_sizes) - 10} more")

print("\n" + "="*90)
print("ROOT CAUSE ANALYSIS")
print("="*90)

if failed_sizes:
    min_failing = failed_sizes[0]
    print(f"""
The Gaussian weight function with var=0.01 produces weights BELOW 1e-6 at the
overlap edges for tiles of size {min_failing[1]}×{min_failing[1]} and larger.

This happens because:
1. Tiles are expanded by overlap={overlap} on each side
2. Core size {min_failing[0]}×{min_failing[0]} becomes tile size {min_failing[1]}×{min_failing[1]}
3. Gaussian is centered at ({(min_failing[1]-1)/2:.1f}, {min_failing[1]/2:.1f})
4. Edge of overlap region is at index {overlap}
5. Distance from center to overlap edge: ~{(min_failing[1]-1)/2 - overlap:.1f} pixels
6. With var=0.01, Gaussian weight at this distance ≈ {min_failing[2]:.2e}
7. This is BELOW the threshold of 1e-6!

IMPACT:
- Pixels covered ONLY by tiles of size ≥ {min_failing[1]}×{min_failing[1]} will be marked as "uncovered"
- The first 6 pixels (overlap region) of such tiles contribute effectively ZERO weight
- This causes coverage gaps at tile boundaries!
""")

print("\n" + "="*90)
print("SOLUTION OPTIONS")
print("="*90)
print("""
Option A: INCREASE VARIANCE in gaussian_weights()
  Change var=0.01 to var=0.02 or higher
  This makes the Gaussian wider, giving higher weights at edges

Option B: USE DIFFERENT WEIGHT FUNCTION
  Instead of Gaussian, use a function that guarantees non-zero weights
  Example: Cosine fade, linear fade, or clipped Gaussian

Option C: FIX THE ASYMMETRY
  Line 799 uses `tile_h / 2` but should use `(tile_h - 1) / 2`
  This creates a 0.5 pixel shift in the y-axis midpoint
  While this doesn't fix the main issue, it improves symmetry

Option D: ADJUST THE CORE SIZE CALCULATION
  Reduce overlap or adjust how tiles are expanded
  This is more invasive and may affect image quality

Option E: CHANGE THE THRESHOLD
  Lower the threshold from 1e-6 to 1e-7 or 1e-8
  This accepts smaller weights as "covered"
  Risk: May allow true gaps to pass validation

RECOMMENDED FIX: Option A + Option C
  1. Increase var from 0.01 to 0.02 or 0.03
  2. Fix the y-axis midpoint to use (tile_h - 1) / 2
  3. This will make weights symmetric and sufficiently large at edges
""")

print("\n" + "="*90)
print("VERIFICATION: What variance is needed?")
print("="*90)

# Find the minimum variance needed to pass threshold for largest failing tile
if failed_sizes:
    largest_fail_core, largest_fail_tile, _ = failed_sizes[-1]

    print(f"\nFor the largest failing tile ({largest_fail_tile}×{largest_fail_tile}):")
    print(f"Testing different variance values...\n")

    for var in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1]:
        edge_weight, _, _ = calculate_edge_weight(largest_fail_tile, overlap, var)
        status = "✓" if edge_weight >= threshold else "✗"
        print(f"  var={var:.3f}: edge_weight = {edge_weight:.2e} {status}")

    # Find minimum variance
    for var in [0.01 + i*0.001 for i in range(100)]:
        edge_weight, _, _ = calculate_edge_weight(largest_fail_tile, overlap, var)
        if edge_weight >= threshold:
            print(f"\n  Minimum variance needed: {var:.3f}")
            print(f"  This ensures edge weights >= 1e-6 for all tile sizes up to {largest_fail_tile}×{largest_fail_tile}")
            break

print("\n" + "="*90)
