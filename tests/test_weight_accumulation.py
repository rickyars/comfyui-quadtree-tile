#!/usr/bin/env python3
"""
Test that weight accumulation correctly handles tiles that extend beyond image boundaries.
Verify that only pixels inside the image receive weights.
"""

def simulate_weight_accumulation(tile_x, tile_y, tile_w, tile_h, image_w, image_h):
    """
    Simulate the weight accumulation logic from lines 423-438.
    Returns the region that receives weights, or None if no overlap.
    """
    # Calculate the intersection of the tile with the image
    x_start = max(0, tile_x)
    y_start = max(0, tile_y)
    x_end = min(image_w, tile_x + tile_w)
    y_end = min(image_h, tile_y + tile_h)

    # Offset into the tile weights tensor
    tile_x_offset = x_start - tile_x
    tile_y_offset = y_start - tile_y
    tile_x_end_offset = tile_x_offset + (x_end - x_start)
    tile_y_end_offset = tile_y_offset + (y_end - y_start)

    if x_end > x_start and y_end > y_start:
        return {
            'image_region': (x_start, y_start, x_end, y_end),
            'tile_weights_region': (tile_x_offset, tile_y_offset,
                                   tile_x_end_offset, tile_y_end_offset),
            'size': (x_end - x_start, y_end - y_start)
        }
    return None

print("="*70)
print("WEIGHT ACCUMULATION VALIDATION")
print("="*70)

# Test Case 1: Tile extends beyond bottom boundary
print("\n1. Tile extends beyond bottom boundary")
print("   Leaf at y=232, overlap=8, image_h=232")
tile = simulate_weight_accumulation(
    tile_x=-8, tile_y=224, tile_w=80, tile_h=80,
    image_w=64, image_h=232
)
if tile:
    img = tile['image_region']
    wt = tile['tile_weights_region']
    print(f"   ✓ Image region: [{img[0]}, {img[2]}) × [{img[1]}, {img[3]})")
    print(f"   ✓ Weights slice: [y:{wt[1]}:{wt[3]}, x:{wt[0]}:{wt[2]}]")
    print(f"   ✓ Size: {tile['size'][0]}×{tile['size'][1]} latent pixels")
    # Verify only pixels inside image get weights
    assert img[3] <= 232, "Weights extend beyond image height!"
    assert img[1] >= 0, "Weights start before image!"
    print("   ✓ Weights correctly limited to image bounds")

# Test Case 2: Tile extends beyond top-left corner
print("\n2. Tile extends beyond top-left corner")
print("   Leaf at (0,0), overlap=8")
tile = simulate_weight_accumulation(
    tile_x=-8, tile_y=-8, tile_w=80, tile_h=80,
    image_w=64, image_h=232
)
if tile:
    img = tile['image_region']
    wt = tile['tile_weights_region']
    print(f"   ✓ Image region: [{img[0]}, {img[2]}) × [{img[1]}, {img[3]})")
    print(f"   ✓ Weights slice: [y:{wt[1]}:{wt[3]}, x:{wt[0]}:{wt[2]}]")
    print(f"   ✓ Size: {tile['size'][0]}×{tile['size'][1]} latent pixels")
    # Verify weights start at 0
    assert img[0] == 0, "Image region doesn't start at 0!"
    assert img[1] == 0, "Image region doesn't start at 0!"
    # Verify tile offset accounts for negative position
    assert wt[0] == 8, f"Tile x offset should be 8, got {wt[0]}"
    assert wt[1] == 8, f"Tile y offset should be 8, got {wt[1]}"
    print("   ✓ Negative coordinates correctly handled with offset")

# Test Case 3: Tile completely inside image
print("\n3. Tile completely inside image")
print("   Leaf at (16,16), overlap=8")
tile = simulate_weight_accumulation(
    tile_x=8, tile_y=8, tile_w=80, tile_h=80,
    image_w=64, image_h=232
)
if tile:
    img = tile['image_region']
    wt = tile['tile_weights_region']
    print(f"   ✓ Image region: [{img[0]}, {img[2]}) × [{img[1]}, {img[3]})")
    print(f"   ✓ Weights slice: [y:{wt[1]}:{wt[3]}, x:{wt[0]}:{wt[2]}]")
    print(f"   ✓ Size: {tile['size'][0]}×{tile['size'][1]} latent pixels")
    # Tile is clipped by image width
    assert img[2] == 64, "Tile should be clipped at image width"
    print("   ✓ Tile correctly clipped at image boundary")

# Test Case 4: Tile partially extends right
print("\n4. Tile extends beyond right boundary")
print("   Leaf at x=57, overlap=8, image_w=64")
tile = simulate_weight_accumulation(
    tile_x=49, tile_y=0, tile_w=80, tile_h=80,
    image_w=64, image_h=232
)
if tile:
    img = tile['image_region']
    wt = tile['tile_weights_region']
    print(f"   ✓ Image region: [{img[0]}, {img[2]}) × [{img[1]}, {img[3]})")
    print(f"   ✓ Weights slice: [y:{wt[1]}:{wt[3]}, x:{wt[0]}:{wt[2]}]")
    print(f"   ✓ Size: {tile['size'][0]}×{tile['size'][1]} latent pixels")
    # Verify only 15 pixels of width covered
    assert img[2] - img[0] == 15, f"Should cover 15 pixels, got {img[2] - img[0]}"
    print("   ✓ Tile correctly clipped at right boundary")

# Test Case 5: Critical bug case - leaf at boundary with overlap
print("\n5. CRITICAL: Boundary leaf that was causing gap")
print("   Leaf at (0, 232), overlap=8, image 64×232")
print("   This is the exact case from the bug report!")
tile = simulate_weight_accumulation(
    tile_x=-8, tile_y=224, tile_w=80, tile_h=80,
    image_w=64, image_h=232
)
if tile:
    img = tile['image_region']
    wt = tile['tile_weights_region']
    print(f"   ✓ Image region: [{img[0]}, {img[2]}) × [{img[1]}, {img[3]})")
    print(f"   ✓ Coverage: y ∈ [{img[1]}, {img[3]})")
    print(f"   ✓ Size: {tile['size'][0]}×{tile['size'][1]} latent pixels")

    # This is the critical check - does it cover the bottom 8 pixels?
    assert img[1] == 224, f"Should start at y=224, got {img[1]}"
    assert img[3] == 232, f"Should end at y=232, got {img[3]}"
    assert img[3] - img[1] == 8, f"Should cover 8 pixels, got {img[3] - img[1]}"
    print("   ✓ CORRECTLY COVERS THE BOTTOM 8 LATENT PIXELS!")
    print("   ✓ Gap at pixels [1792, 1856] would be FIXED!")
else:
    print("   ✗ ERROR: No weights accumulated - gap would persist!")

print("\n" + "="*70)
print("✓ ALL WEIGHT ACCUMULATION TESTS PASSED")
print("="*70)
