#!/usr/bin/env python3
"""
Mathematical verification of the filtering conditions.
Proves that the filtering logic correctly identifies tiles that overlap with the image.
"""

def verify_filter_math(leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h):
    """
    Verify the mathematical correctness of the filtering logic.
    Returns True if math is correct, False otherwise.
    """
    # Filtering conditions (tile is REMOVED if ANY is true)
    cond1 = (leaf_x - overlap) >= img_w
    cond2 = (leaf_y - overlap) >= img_h
    cond3 = (leaf_x + leaf_w + overlap) <= 0
    cond4 = (leaf_y + leaf_h + overlap) <= 0

    # Tile should be removed if ANY condition is true
    should_remove = cond1 or cond2 or cond3 or cond4

    # Calculate actual tile boundaries after overlap
    tile_x_start = leaf_x - overlap
    tile_y_start = leaf_y - overlap
    tile_x_end = leaf_x + leaf_w + overlap  # This is correct: (x - overlap) + (w + 2*overlap) = x + w + overlap
    tile_y_end = leaf_y + leaf_h + overlap

    # Calculate intersection with image [0, img_w) × [0, img_h)
    intersect_x_start = max(0, tile_x_start)
    intersect_y_start = max(0, tile_y_start)
    intersect_x_end = min(img_w, tile_x_end)
    intersect_y_end = min(img_h, tile_y_end)

    # Has intersection if ranges overlap
    has_intersection = (intersect_x_end > intersect_x_start and
                       intersect_y_end > intersect_y_start)

    # VERIFICATION: Tile should be removed IFF it has no intersection
    math_correct = (should_remove == (not has_intersection))

    return math_correct, {
        'leaf': (leaf_x, leaf_y, leaf_w, leaf_h),
        'tile_after_overlap': (tile_x_start, tile_y_start, tile_x_end, tile_y_end),
        'conditions': {
            'start_x >= img_w': cond1,
            'start_y >= img_h': cond2,
            'end_x <= 0': cond3,
            'end_y <= 0': cond4,
        },
        'should_remove': should_remove,
        'has_intersection': has_intersection,
        'math_correct': math_correct,
    }

print("="*70)
print("MATHEMATICAL VERIFICATION OF FILTERING LOGIC")
print("="*70)
print("\nVerifying that filtering conditions correctly identify overlapping tiles")
print()

test_cases = [
    # (name, leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h, expected_keep)
    ("Boundary tile (y=232, img_h=232)", 0, 232, 64, 64, 8, 64, 232, True),
    ("Beyond boundary (y=240, img_h=232)", 0, 240, 64, 64, 8, 64, 232, False),
    ("Just before boundary (y=224)", 0, 224, 64, 64, 8, 64, 232, True),
    ("Top-left corner (negative after overlap)", 0, 0, 64, 64, 8, 64, 232, True),
    ("Far left (tile_end_x = 12)", -60, 0, 64, 64, 8, 64, 232, True),
    ("Completely left (tile_end_x = -8)", -80, 0, 64, 64, 8, 64, 232, False),
    ("Right boundary (x=57)", 57, 0, 64, 64, 8, 64, 232, True),
    ("Beyond right (x=72)", 72, 0, 64, 64, 8, 64, 232, False),
    ("Zero overlap at boundary", 0, 232, 64, 64, 0, 64, 232, False),
]

all_correct = True
for name, *params in test_cases:
    leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h, expected_keep = params
    correct, details = verify_filter_math(leaf_x, leaf_y, leaf_w, leaf_h, overlap, img_w, img_h)

    should_keep = not details['should_remove']
    status = "✓" if (correct and should_keep == expected_keep) else "✗"

    print(f"{status} {name}")
    print(f"  Leaf: ({leaf_x}, {leaf_y}, {leaf_w}, {leaf_h})")
    tile = details['tile_after_overlap']
    print(f"  Tile: x∈[{tile[0]}, {tile[2]}), y∈[{tile[1]}, {tile[3]})")
    print(f"  Image: x∈[0, {img_w}), y∈[0, {img_h})")
    print(f"  Decision: {'KEEP' if should_keep else 'REMOVE'} (expected: {'KEEP' if expected_keep else 'REMOVE'})")
    print(f"  Has intersection: {details['has_intersection']}")
    print(f"  Math correct: {correct}")

    if not correct:
        print(f"  ✗ ERROR: Filtering logic does NOT match intersection!")
        print(f"     Conditions: {details['conditions']}")
        all_correct = False
    elif should_keep != expected_keep:
        print(f"  ✗ ERROR: Unexpected result!")
        all_correct = False

    print()

print("="*70)
if all_correct:
    print("✓ ALL MATH VERIFICATION TESTS PASSED")
    print("  The filtering logic CORRECTLY identifies tiles that overlap with the image")
else:
    print("✗ MATH VERIFICATION FAILED")
    print("  The filtering logic has errors")
print("="*70)

# Additional edge case verification
print("\nEDGE CASE: Tile end calculation")
print("  tile_end = tile_start + tile_size")
print("  tile_start = leaf.x - overlap")
print("  tile_size = leaf.w + 2*overlap")
print("  tile_end = (leaf.x - overlap) + (leaf.w + 2*overlap)")
print("           = leaf.x + leaf.w + overlap ✓")
print("\nThis matches the filtering condition: (leaf.x + leaf.w + overlap) <= 0")
print("="*70)
