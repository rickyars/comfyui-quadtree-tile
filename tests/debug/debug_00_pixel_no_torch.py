#!/usr/bin/env python3
"""
Debug script to trace why pixel (0,0) has zero weights.
Simulates the exact weight accumulation for a single tile covering (0,0).
No torch dependency - pure Python simulation.
"""

import numpy as np
from math import exp, sqrt, pi

def gaussian_weights(tile_w, tile_h):
    '''Gaussian weight generation from tiled_diffusion.py'''
    f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]
    y_probs = [f(y,  tile_h      / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return w

print("="*80)
print("DEBUG: Why is pixel (0,0) uncovered?")
print("="*80)

# Simulate the scenario
image_w_latent = 512
image_h_latent = 232
overlap = 6

# Initialize weights (simulated as numpy array)
weights = np.zeros((image_h_latent, image_w_latent), dtype=np.float32)
print(f"\nInitialized weights: shape {weights.shape}")
print(f"Initial value at (0,0): {weights[0, 0]}")

# Simulate a tile at core (0, 0, 64, 64) - this should cover pixel (0,0)
# After overlap: tile at (-6, -6, 76, 76)
core_x, core_y, core_w, core_h = 0, 0, 64, 64
x = core_x - overlap  # -6
y = core_y - overlap  # -6
w = core_w + 2 * overlap  # 76
h = core_h + 2 * overlap  # 76

print(f"\nTile 1: Core at ({core_x}, {core_y}, {core_w}, {core_h})")
print(f"        With overlap: ({x}, {y}, {w}, {h})")

# Get Gaussian weights
tile_weights = gaussian_weights(w, h)
print(f"        Gaussian weights shape: {tile_weights.shape}")
print(f"        Weight at center [38, 38]: {tile_weights[38, 38]:.6f}")
print(f"        Weight at [6, 6] (maps to image 0,0): {tile_weights[6, 6]:.6f}")
print(f"        Weight range: [{tile_weights.min():.8f}, {tile_weights.max():.6f}]")

# Calculate intersection with image
x_start = max(0, x)
y_start = max(0, y)
x_end = min(image_w_latent, x + w)
y_end = min(image_h_latent, y + h)

print(f"\nIntersection with image:")
print(f"        x: [{x_start}, {x_end}) - covers {x_end - x_start} pixels")
print(f"        y: [{y_start}, {y_end}) - covers {y_end - y_start} pixels")

# Calculate tile offsets
tile_x_offset = x_start - x
tile_y_offset = y_start - y
tile_x_end_offset = tile_x_offset + (x_end - x_start)
tile_y_end_offset = tile_y_offset + (y_end - y_start)

print(f"\nTile weights slice:")
print(f"        y: [{tile_y_offset}, {tile_y_end_offset}) - {tile_y_end_offset - tile_y_offset} elements")
print(f"        x: [{tile_x_offset}, {tile_x_end_offset}) - {tile_x_end_offset - tile_x_offset} elements")

# Perform accumulation
if x_end > x_start and y_end > y_start:
    print(f"\n✓ Condition met: x_end ({x_end}) > x_start ({x_start}) and y_end ({y_end}) > y_start ({y_start})")

    # Get the slices
    tile_slice = tile_weights[tile_y_offset:tile_y_end_offset, tile_x_offset:tile_x_end_offset]

    print(f"\nBefore accumulation:")
    print(f"        tile slice shape: {tile_slice.shape}")
    print(f"        weights[0,0] = {weights[0, 0]}")
    print(f"        tile_weights[{tile_y_offset}, {tile_x_offset}] = {tile_weights[tile_y_offset, tile_x_offset]:.6f}")

    # Accumulate (simulate the exact operation from line 493-494)
    weights[y_start:y_end, x_start:x_end] += tile_slice

    print(f"\nAfter accumulation:")
    print(f"        weights[0,0] = {weights[0, 0]:.6f}")
    print(f"        weights[0,5] = {weights[0, 5]:.6f}")
    print(f"        weights[5,0] = {weights[5, 0]:.6f}")

    if weights[0, 0] > 1e-6:
        print(f"\n✓ SUCCESS: Pixel (0,0) has weight {weights[0, 0]:.6f}")
    else:
        print(f"\n✗ FAILURE: Pixel (0,0) still has weight {weights[0, 0]}")
        print(f"   This means the tile did NOT contribute weight to (0,0)!")
else:
    print(f"\n✗ Condition NOT met - no accumulation would happen!")

print(f"\n" + "="*80)
print("CHECKING THE GAUSSIAN WEIGHTS ASYMMETRY")
print("="*80)

# Check the asymmetry bug in gaussian_weights
tile_size = 20
print(f"\nFor a {tile_size}×{tile_size} tile:")

f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_size*tile_size) / (2*var)) / sqrt(2*pi*var)

x_midpoint = (tile_size - 1) / 2  # 9.5 for size=20
y_midpoint = tile_size / 2  # 10.0 for size=20

print(f"  x_probs midpoint: {x_midpoint}")
print(f"  y_probs midpoint: {y_midpoint}")
print(f"  ASYMMETRY: {y_midpoint - x_midpoint:.1f} unit difference")

# Calculate weights at index 0 (edge of tile)
x_prob_0 = f(0, x_midpoint)
y_prob_0 = f(0, y_midpoint)

print(f"\n  x_probs[0] = {x_prob_0:.8f}")
print(f"  y_probs[0] = {y_prob_0:.8f}")
print(f"  Ratio: {y_prob_0 / x_prob_0:.4f}")

# Calculate weights at center
x_prob_center = f(9, x_midpoint)
y_prob_center = f(10, y_midpoint)

print(f"\n  x_probs[9] (near center) = {x_prob_center:.6f}")
print(f"  y_probs[10] (at center) = {y_prob_center:.6f}")

print(f"\n" + "="*80)
