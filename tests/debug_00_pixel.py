#!/usr/bin/env python3
"""
Debug script to trace why pixel (0,0) has zero weights.
Simulates the exact weight accumulation for a single tile covering (0,0).
"""

import torch
import numpy as np
from math import exp, sqrt, pi

def gaussian_weights(tile_w:int, tile_h:int) -> torch.Tensor:
    '''Gaussian weight generation from tiled_diffusion.py'''
    f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]
    y_probs = [f(y,  tile_h      / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to('cpu', dtype=torch.float32)

print("="*80)
print("DEBUG: Why is pixel (0,0) uncovered?")
print("="*80)

# Simulate the scenario
image_w_latent = 512
image_h_latent = 232
overlap = 6

# Initialize weights tensor
weights = torch.zeros((1, 1, image_h_latent, image_w_latent), dtype=torch.float32)
print(f"\nInitialized weights: shape {weights.shape}")
print(f"Initial value at (0,0): {weights[0, 0, 0, 0].item()}")

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
print(f"        Weight at center [38, 38]: {tile_weights[38, 38].item():.6f}")
print(f"        Weight at [6, 6] (maps to image 0,0): {tile_weights[6, 6].item():.6f}")
print(f"        Weight range: [{tile_weights.min().item():.8f}, {tile_weights.max().item():.6f}]")

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
    weights_slice = weights[:, :, y_start:y_end, x_start:x_end]
    tile_slice = tile_weights[tile_y_offset:tile_y_end_offset, tile_x_offset:tile_x_end_offset]

    print(f"\nBefore accumulation:")
    print(f"        weights slice shape: {weights_slice.shape}")
    print(f"        tile slice shape: {tile_slice.shape}")
    print(f"        weights[0,0,0,0] = {weights[0, 0, 0, 0].item()}")

    # Accumulate
    weights[:, :, y_start:y_end, x_start:x_end] += tile_slice

    print(f"\nAfter accumulation:")
    print(f"        weights[0,0,0,0] = {weights[0, 0, 0, 0].item():.6f}")
    print(f"        weights[0,0,0,5] = {weights[0, 0, 0, 5].item():.6f}")
    print(f"        weights[0,0,5,0] = {weights[0, 0, 5, 0].item():.6f}")

    if weights[0, 0, 0, 0].item() > 1e-6:
        print(f"\n✓ SUCCESS: Pixel (0,0) has weight {weights[0, 0, 0, 0].item():.6f}")
    else:
        print(f"\n✗ FAILURE: Pixel (0,0) still has weight {weights[0, 0, 0, 0].item()}")
        print(f"   Expected: tile_weights[{tile_y_offset}, {tile_x_offset}] = {tile_weights[tile_y_offset, tile_x_offset].item():.6f}")
else:
    print(f"\n✗ Condition NOT met - no accumulation would happen!")

# Check for potential broadcasting issues
print(f"\n" + "="*80)
print("BROADCASTING TEST")
print("="*80)

# Test if broadcasting works correctly
test_4d = torch.zeros((1, 1, 10, 10), dtype=torch.float32)
test_2d = torch.ones((10, 10), dtype=torch.float32)

print(f"4D tensor shape: {test_4d.shape}")
print(f"2D tensor shape: {test_2d.shape}")

test_4d[:, :, 0:5, 0:5] += test_2d[0:5, 0:5]
print(f"After adding 2D[0:5,0:5] to 4D[:,:,0:5,0:5]:")
print(f"  test_4d[0,0,0,0] = {test_4d[0,0,0,0].item()}")
print(f"  test_4d[0,0,4,4] = {test_4d[0,0,4,4].item()}")

if test_4d[0,0,0,0].item() == 1.0:
    print("✓ Broadcasting works correctly")
else:
    print("✗ Broadcasting FAILED - this is the bug!")

print("\n" + "="*80)
