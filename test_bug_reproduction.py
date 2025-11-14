#!/usr/bin/env python3
"""
Reproduce the exact bug scenario from the bug report and verify it's fixed.

BUG REPORT:
- Image: 512×1856 pixels = 64×232 latent
- Quadtree tiles: min=512, max=512 (all 512×512)
- Overlap mode: 64 pixels = 8 latent
- Problem: Tiles at y=1856 (latent 232) were filtered out
- Result: Gap at pixels [1792, 1856] with zero weights

EXPECTED FIX:
- Tiles at y=232 should be KEPT because after overlap they cover [224, 232]
- This provides 8 latent pixels (64 pixels) of overlap with the image
"""

def simulate_old_filter(leaf_x_pixels, leaf_y_pixels, leaf_w_pixels, leaf_h_pixels,
                        img_w_pixels, img_h_pixels):
    """
    Simulate the OLD filtering logic (before the fix) from tiled_vae.py.
    Filtered in IMAGE SPACE without accounting for overlap or latent conversion.
    This was done at build-time with potentially different dimensions.
    """
    # Old logic from tiled_vae.py: filter if position is >= image dimensions
    # This was done in IMAGE SPACE (pixels) at build time
    remove = (leaf_x_pixels >= img_w_pixels or leaf_y_pixels >= img_h_pixels or
              leaf_x_pixels + leaf_w_pixels <= 0 or leaf_y_pixels + leaf_h_pixels <= 0)
    return not remove

def simulate_new_filter(leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h):
    """
    Simulate the NEW filtering logic (after the fix).
    Accounts for overlap that will be added later.
    """
    # New logic: filter based on tile position after overlap
    remove = (
        (leaf_x - overlap) >= img_w or
        (leaf_y - overlap) >= img_h or
        (leaf_x + leaf_w + overlap) <= 0 or
        (leaf_y + leaf_h + overlap) <= 0
    )
    return not remove

print("="*70)
print("BUG REPRODUCTION TEST")
print("="*70)
print("\nScenario: 512×1856 pixel image with 64-pixel overlap")
print("Latent space: 64×232 with overlap=8")
print()

# Simulate the quadtree leaves from the bug scenario
# Quadtree produces 512×512 pixel tiles = 64×64 latent tiles
img_w_latent = 64
img_h_latent = 232
overlap_latent = 8
tile_size_latent = 64

print("Simulating quadtree tiles covering the square root area...")
print("Square root dimension: 1856 pixels = 232 latent")
print()

# Generate tiles that would be produced by the quadtree
# The quadtree covers a square of 1920×1920 (240×240 latent) - rounded up
# With 512×512 tiles (64×64 latent)
# This matches the actual bug where tiles were generated at y=232 (1856 pixels)
tiles = []
for y in range(0, 240, 64):  # 0, 64, 128, 192, 256 would be beyond but quadtree generates it
    for x in range(0, 64, 64):  # 0 (only one tile wide for 512px)
        tiles.append((x, y, tile_size_latent, tile_size_latent))

# Add the critical boundary tile at y=232 that was causing the bug
# This tile starts at y=1856 pixels (232 latent), which is exactly at the image height
tiles.append((0, 232, tile_size_latent, tile_size_latent))

print(f"Generated {len(tiles)} tiles:")
for i, (x, y, w, h) in enumerate(tiles):
    print(f"  Tile {i}: ({x}, {y}, {w}, {h}) latent")

print("\n" + "-"*70)
print("TESTING OLD FILTER (Before Fix - from tiled_vae.py)")
print("  Note: Old filter used IMAGE SPACE (pixels) at build time")
print("-"*70)

img_w_pixels = img_w_latent * 8
img_h_pixels = img_h_latent * 8

old_kept = []
old_filtered = []
for x, y, w, h in tiles:
    # Old filter worked in IMAGE SPACE
    x_pixels, y_pixels = x * 8, y * 8
    w_pixels, h_pixels = w * 8, h * 8
    keep = simulate_old_filter(x_pixels, y_pixels, w_pixels, h_pixels,
                               img_w_pixels, img_h_pixels)
    if keep:
        old_kept.append((x, y, w, h))
    else:
        old_filtered.append((x, y, w, h))

print(f"Kept: {len(old_kept)} tiles")
for x, y, w, h in old_kept:
    print(f"  ✓ ({x}, {y}, {w}, {h}) latent = ({x*8}, {y*8}, {w*8}, {h*8}) pixels")

print(f"Filtered: {len(old_filtered)} tiles")
for x, y, w, h in old_filtered:
    print(f"  ✗ ({x}, {y}, {w}, {h}) latent = ({x*8}, {y*8}, {w*8}, {h*8}) pixels")
    # Check if this tile should have been kept
    tile_y_after_overlap = y - overlap_latent
    tile_y_end_after_overlap = y + h + overlap_latent
    if tile_y_end_after_overlap > 0 and tile_y_after_overlap < img_h_latent:
        print(f"    ⚠️  BUG: After overlap, tile covers latent y ∈ [{tile_y_after_overlap}, {min(tile_y_end_after_overlap, img_h_latent)})")
        print(f"            This is {min(tile_y_end_after_overlap, img_h_latent) - tile_y_after_overlap}×{w} latent pixels")
        print(f"            = {(min(tile_y_end_after_overlap, img_h_latent) - tile_y_after_overlap) * 8}×{w*8} PIXELS UNCOVERED!")

# Check coverage with old filter
print("\nCoverage analysis (old filter):")
if old_kept:
    # Calculate actual coverage considering overlap and clipping
    coverage = set()
    for x, y, w, h in old_kept:
        tile_y_start = max(0, y - overlap_latent)
        tile_y_end = min(img_h_latent, y + h + overlap_latent)
        for py in range(tile_y_start, tile_y_end):
            coverage.add(py)

    min_covered = min(coverage) if coverage else 0
    max_covered = max(coverage) if coverage else 0
    gaps = []
    for y in range(img_h_latent):
        if y not in coverage:
            gaps.append(y)

    print(f"  Coverage: y ∈ [{min_covered}, {max_covered + 1}] latent")
    print(f"  Covered: {len(coverage)} / {img_h_latent} latent pixels")
    if gaps:
        gap_start = min(gaps)
        gap_end = max(gaps) + 1
        gap_count = len(gaps)
        print(f"  ✗ GAP DETECTED: {gap_count} latent pixels ({gap_count * 8} pixels) NOT COVERED!")
        print(f"    Gap range: y ∈ [{gap_start}, {gap_end}] latent = [{gap_start*8}, {gap_end*8}] pixels")
    else:
        print(f"  ✓ Full coverage")

print("\n" + "-"*70)
print("TESTING NEW FILTER (After Fix)")
print("-"*70)

new_kept = []
new_filtered = []
for x, y, w, h in tiles:
    # New filter accounts for overlap
    keep = simulate_new_filter(x, y, w, h, overlap_latent, img_w_latent, img_h_latent)
    if keep:
        new_kept.append((x, y, w, h))
    else:
        new_filtered.append((x, y, w, h))

print(f"Kept: {len(new_kept)} tiles")
for x, y, w, h in new_kept:
    tile_y_start = y - overlap_latent
    tile_y_end = y + h + overlap_latent
    overlap_start = max(0, tile_y_start)
    overlap_end = min(img_h_latent, tile_y_end)
    print(f"  ✓ ({x}, {y}, {w}, {h}) -> covers y ∈ [{overlap_start}, {overlap_end}]")

print(f"Filtered: {len(new_filtered)} tiles")
for x, y, w, h in new_filtered:
    print(f"  ✗ ({x}, {y}, {w}, {h})")

# Check coverage with new filter
print("\nCoverage analysis (new filter):")
if new_kept:
    # Calculate actual coverage considering overlap and clipping
    coverage = set()
    for x, y, w, h in new_kept:
        tile_y_start = max(0, y - overlap_latent)
        tile_y_end = min(img_h_latent, y + h + overlap_latent)
        for py in range(tile_y_start, tile_y_end):
            coverage.add(py)

    min_covered = min(coverage)
    max_covered = max(coverage)
    gaps = []
    for y in range(img_h_latent):
        if y not in coverage:
            gaps.append(y)

    print(f"  Coverage: y ∈ [{min_covered}, {max_covered + 1}]")
    print(f"  Covered pixels: {len(coverage)} / {img_h_latent}")
    if gaps:
        print(f"  ✗ GAPS FOUND at: {gaps[:10]}{'...' if len(gaps) > 10 else ''}")
    else:
        print(f"  ✓ FULL COVERAGE - NO GAPS!")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Old filter: {len(old_kept)} tiles kept, {len(old_filtered)} filtered")
print(f"New filter: {len(new_kept)} tiles kept, {len(new_filtered)} filtered")
print(f"Difference: {len(new_kept) - len(old_kept)} additional tiles kept")

if len(new_kept) > len(old_kept):
    print("\n✓ FIX WORKING: More tiles kept to provide full coverage")
else:
    print("\n⚠️  WARNING: Fix may not be working as expected")

print("="*70)
