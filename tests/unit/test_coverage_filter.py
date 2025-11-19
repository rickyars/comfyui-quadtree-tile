#!/usr/bin/env python3
"""
Test the coverage gap fix filtering logic.
Verify that the filtering correctly identifies tiles that overlap with the image.
"""

def should_keep_leaf(leaf_x_latent, leaf_y_latent, leaf_w_latent, leaf_h_latent,
                     overlap, image_w_latent, image_h_latent):
    """
    Simulate the filtering logic from lines 385-389.
    Returns True if the leaf should be KEPT, False if it should be FILTERED OUT.
    """
    # The filtering condition - remove if ANY condition is true
    remove = (
        (leaf_x_latent - overlap) >= image_w_latent or
        (leaf_y_latent - overlap) >= image_h_latent or
        (leaf_x_latent + leaf_w_latent + overlap) <= 0 or
        (leaf_y_latent + leaf_h_latent + overlap) <= 0
    )
    return not remove

def calculate_intersection(leaf_x_latent, leaf_y_latent, leaf_w_latent, leaf_h_latent,
                          overlap, image_w_latent, image_h_latent):
    """
    Calculate the actual intersection between tile (after overlap) and image.
    Returns (x_start, y_start, x_end, y_end) or None if no intersection.
    """
    # Tile position after overlap
    tile_x = leaf_x_latent - overlap
    tile_y = leaf_y_latent - overlap
    tile_w = leaf_w_latent + 2 * overlap
    tile_h = leaf_h_latent + 2 * overlap

    # Calculate intersection
    x_start = max(0, tile_x)
    y_start = max(0, tile_y)
    x_end = min(image_w_latent, tile_x + tile_w)
    y_end = min(image_h_latent, tile_y + tile_h)

    if x_end > x_start and y_end > y_start:
        return (x_start, y_start, x_end, y_end)
    return None

def run_test_case(name, leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h, expected_keep):
    """Run a single test case."""
    keep = should_keep_leaf(leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h)
    intersection = calculate_intersection(leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h)

    status = "✓" if keep == expected_keep else "✗"
    print(f"\n{status} Test: {name}")
    print(f"  Leaf: ({leaf_x}, {leaf_y}, {leaf_w}, {leaf_h}) latent")
    print(f"  Image: {img_w}×{img_h} latent, Overlap: {overlap}")
    print(f"  Tile after overlap: ({leaf_x - overlap}, {leaf_y - overlap}, {leaf_w + 2*overlap}, {leaf_h + 2*overlap})")
    print(f"  Filter decision: {'KEEP' if keep else 'REMOVE'} (expected: {'KEEP' if expected_keep else 'REMOVE'})")

    if intersection:
        x_s, y_s, x_e, y_e = intersection
        print(f"  Intersection: [{x_s}, {x_e}) × [{y_s}, {y_e}) = {x_e-x_s}×{y_e-y_s} latent pixels")
        if not keep:
            print(f"  ⚠️ WARNING: Tile has intersection but was filtered out!")
            return False
    else:
        print(f"  Intersection: None")
        if keep:
            print(f"  ⚠️ WARNING: Tile has no intersection but was kept!")
            return False

    return keep == expected_keep

# Run test suite
print("="*70)
print("COVERAGE GAP FIX - FILTERING LOGIC VALIDATION")
print("="*70)

all_passed = True

# Test Case 1: Leaf exactly at bottom boundary (from bug report)
# Image: 512×1856 pixels = 64×232 latent
all_passed &= run_test_case(
    "Leaf at bottom boundary (y=232)",
    leaf_x=0, leaf_y=232, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=True  # Should keep - has 8 pixels of overlap
)

# Test Case 2: Leaf one tile beyond boundary
all_passed &= run_test_case(
    "Leaf beyond boundary (y=240)",
    leaf_x=0, leaf_y=240, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=False  # Should remove - no overlap
)

# Test Case 3: Leaf just before boundary
all_passed &= run_test_case(
    "Leaf before boundary (y=224)",
    leaf_x=0, leaf_y=224, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=True  # Should keep - has overlap
)

# Test Case 4: Leaf at top-left corner (negative after overlap)
all_passed &= run_test_case(
    "Leaf at top-left corner",
    leaf_x=0, leaf_y=0, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=True  # Should keep - covers top-left area
)

# Test Case 5: Leaf partially to the left of image
all_passed &= run_test_case(
    "Leaf partially left of image (x=-60)",
    leaf_x=-60, leaf_y=0, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=True  # Should keep - tile end at -60+64+8=12 > 0
)

# Test Case 6: Leaf completely to the left of image
all_passed &= run_test_case(
    "Leaf completely left of image (x=-80)",
    leaf_x=-80, leaf_y=0, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=False  # Should remove - tile end at -80+64+8=-8 <= 0
)

# Test Case 7: Leaf at right boundary (overlaps slightly)
all_passed &= run_test_case(
    "Leaf at right boundary (x=57)",
    leaf_x=57, leaf_y=0, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=True  # Should keep - tile start at 57-8=49 < 64
)

# Test Case 8: Leaf beyond right boundary
all_passed &= run_test_case(
    "Leaf beyond right boundary (x=72)",
    leaf_x=72, leaf_y=0, leaf_w=64, leaf_h=64,
    overlap=8, img_w=64, img_h=232,
    expected_keep=False  # Should remove - tile start at 72-8=64 >= 64
)

# Test Case 9: Zero overlap mode
all_passed &= run_test_case(
    "Zero overlap at boundary",
    leaf_x=0, leaf_y=232, leaf_w=64, leaf_h=64,
    overlap=0, img_w=64, img_h=232,
    expected_keep=False  # Should remove - tile start at 232 >= 232
)

# Test Case 10: Large overlap
all_passed &= run_test_case(
    "Large overlap (overlap=16)",
    leaf_x=0, leaf_y=232, leaf_w=64, leaf_h=64,
    overlap=16, img_w=64, img_h=232,
    expected_keep=True  # Should keep - tile start at 232-16=216 < 232
)

print("\n" + "="*70)
if all_passed:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED")
print("="*70)
