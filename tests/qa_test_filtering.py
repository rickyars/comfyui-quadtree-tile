#!/usr/bin/env python3
"""
QA Test: Verify Coverage Gap Fix Filtering Logic
Tests the leaf filtering logic for various edge cases
"""

def test_filtering_logic(core_start_x, core_start_y, core_end_x, core_end_y,
                         overlap, w, h, test_name):
    """
    Simulates the filtering logic from tiled_diffusion.py lines 390-422

    Args:
        core_start_x, core_start_y: Core start position in latent space
        core_end_x, core_end_y: Core end position in latent space
        overlap: Overlap size in latent space
        w, h: Image dimensions in latent space
        test_name: Name of the test case

    Returns:
        bool: True if should filter, False if should keep
    """
    # Calculate tile bounds with overlap
    tile_start_x = core_start_x - overlap
    tile_start_y = core_start_y - overlap
    tile_end_x = core_end_x + overlap
    tile_end_y = core_end_y + overlap

    # Filter conditions
    core_outside_x = core_start_x >= w or core_end_x <= 0
    core_outside_y = core_start_y >= h or core_end_y <= 0

    # Also filter if tile doesn't overlap at all (safety check)
    tile_no_overlap_x = tile_start_x >= w or tile_end_x <= 0
    tile_no_overlap_y = tile_start_y >= h or tile_end_y <= 0

    should_filter = core_outside_x or core_outside_y or tile_no_overlap_x or tile_no_overlap_y

    # Build debug info
    reasons = []
    if core_outside_x:
        reasons.append("core X outside")
    if core_outside_y:
        reasons.append("core Y outside")
    if tile_no_overlap_x and not core_outside_x:
        reasons.append("tile X no overlap")
    if tile_no_overlap_y and not core_outside_y:
        reasons.append("tile Y no overlap")

    result = "FILTER" if should_filter else "KEEP"
    print(f"\n{test_name}:")
    print(f"  Core latent: ({core_start_x}, {core_start_y}) to ({core_end_x}, {core_end_y})")
    print(f"  Tile latent: ({tile_start_x}, {tile_start_y}) to ({tile_end_x}, {tile_end_y})")
    print(f"  Image latent: {w}x{h}")
    print(f"  Result: {result}")
    if reasons:
        print(f"  Reasons: {', '.join(reasons)}")

    return should_filter


if __name__ == "__main__":
    # Image: 512x232 pixels → 64x29 latent, overlap=8 latent
    W, H = 64, 29
    OVERLAP = 8

    print("=" * 70)
    print("QA TEST: Coverage Gap Fix - Filtering Logic Verification")
    print("=" * 70)
    print(f"Test image: 512x232 pixels → {W}x{H} latent, overlap={OVERLAP}")

    # Test A: Leaf at (0, 224, 8, 8) in pixel space
    # → (0, 28, 1, 1) in latent space (core)
    test_a_filter = test_filtering_logic(
        core_start_x=0, core_start_y=28,
        core_end_x=1, core_end_y=29,
        overlap=OVERLAP, w=W, h=H,
        test_name="Test A: Leaf at pixel (0, 224, 8, 8)"
    )
    assert not test_a_filter, "Test A should KEEP (core inside)"
    print("  ✓ PASS: Core inside, correctly kept")

    # Test B: Leaf at (0, 232, 8, 8) in pixel space
    # → (0, 29, 1, 1) in latent space (core)
    test_b_filter = test_filtering_logic(
        core_start_x=0, core_start_y=29,
        core_end_x=1, core_end_y=30,
        overlap=OVERLAP, w=W, h=H,
        test_name="Test B: Leaf at pixel (0, 232, 8, 8)"
    )
    assert test_b_filter, "Test B should FILTER (core outside)"
    print("  ✓ PASS: Core outside (start_y >= h), correctly filtered")

    # Test C: Leaf at (504, 224, 8, 8) in pixel space
    # → (63, 28, 1, 1) in latent space (core ends exactly at boundary)
    test_c_filter = test_filtering_logic(
        core_start_x=63, core_start_y=28,
        core_end_x=64, core_end_y=29,
        overlap=OVERLAP, w=W, h=H,
        test_name="Test C: Leaf at pixel (504, 224, 8, 8)"
    )
    assert not test_c_filter, "Test C should KEEP (core ends at boundary)"
    print("  ✓ PASS: Core end at exact boundary, correctly kept")

    # Test D: Leaf at (512, 0, 8, 8) in pixel space
    # → (64, 0, 1, 1) in latent space
    test_d_filter = test_filtering_logic(
        core_start_x=64, core_start_y=0,
        core_end_x=65, core_end_y=1,
        overlap=OVERLAP, w=W, h=H,
        test_name="Test D: Leaf at pixel (512, 0, 8, 8)"
    )
    assert test_d_filter, "Test D should FILTER (core outside)"
    print("  ✓ PASS: Core outside (start_x >= w), correctly filtered")

    # Edge Case 1: Zero overlap
    print("\n" + "=" * 70)
    print("EDGE CASE 1: Zero Overlap")
    print("=" * 70)
    test_e1_filter = test_filtering_logic(
        core_start_x=63, core_start_y=28,
        core_end_x=64, core_end_y=29,
        overlap=0, w=W, h=H,
        test_name="Edge Case 1a: Core at boundary, overlap=0"
    )
    assert not test_e1_filter, "Edge case 1a should KEEP"
    print("  ✓ PASS: Works with zero overlap")

    test_e1b_filter = test_filtering_logic(
        core_start_x=64, core_start_y=0,
        core_end_x=65, core_end_y=1,
        overlap=0, w=W, h=H,
        test_name="Edge Case 1b: Core outside, overlap=0"
    )
    assert test_e1b_filter, "Edge case 1b should FILTER"
    print("  ✓ PASS: Correctly filters with zero overlap")

    # Edge Case 2: Large overlap
    print("\n" + "=" * 70)
    print("EDGE CASE 2: Large Overlap (overlap >= tile size)")
    print("=" * 70)
    test_e2_filter = test_filtering_logic(
        core_start_x=0, core_start_y=0,
        core_end_x=1, core_end_y=1,
        overlap=16, w=W, h=H,
        test_name="Edge Case 2: Core inside with overlap=16"
    )
    assert not test_e2_filter, "Edge case 2 should KEEP"
    print("  ✓ PASS: Large overlap doesn't affect core-based filtering")

    # Edge Case 3: Negative tile coordinates (but positive core)
    print("\n" + "=" * 70)
    print("EDGE CASE 3: Negative Tile Coordinates")
    print("=" * 70)
    test_e3_filter = test_filtering_logic(
        core_start_x=0, core_start_y=0,
        core_end_x=1, core_end_y=1,
        overlap=8, w=W, h=H,
        test_name="Edge Case 3: Core at (0,0) - tile extends negative"
    )
    assert not test_e3_filter, "Edge case 3 should KEEP"
    print("  ✓ PASS: Negative tile coordinates handled (core positive)")

    # Edge Case 4: Tile extends far beyond image
    print("\n" + "=" * 70)
    print("EDGE CASE 4: Tile extends beyond image")
    print("=" * 70)
    test_e4_filter = test_filtering_logic(
        core_start_x=63, core_start_y=28,
        core_end_x=64, core_end_y=29,
        overlap=20, w=W, h=H,
        test_name="Edge Case 4: Core at boundary, large overlap=20"
    )
    assert not test_e4_filter, "Edge case 4 should KEEP (core inside)"
    print("  ✓ PASS: Core inside is kept even if tile extends far beyond")

    # Edge Case 5: Core partially outside (one dimension)
    print("\n" + "=" * 70)
    print("EDGE CASE 5: Core Partially Outside")
    print("=" * 70)
    test_e5_filter = test_filtering_logic(
        core_start_x=64, core_start_y=28,
        core_end_x=65, core_end_y=29,
        overlap=8, w=W, h=H,
        test_name="Edge Case 5: Core outside in X, inside in Y"
    )
    assert test_e5_filter, "Edge case 5 should FILTER (core outside in X)"
    print("  ✓ PASS: Filters if core outside in ANY dimension")

    # Edge Case 6: Core exactly at negative boundary
    print("\n" + "=" * 70)
    print("EDGE CASE 6: Core at Negative Boundary")
    print("=" * 70)
    test_e6_filter = test_filtering_logic(
        core_start_x=-1, core_start_y=0,
        core_end_x=0, core_end_y=1,
        overlap=8, w=W, h=H,
        test_name="Edge Case 6: Core ends at 0 (outside)"
    )
    assert test_e6_filter, "Edge case 6 should FILTER (core_end_x <= 0)"
    print("  ✓ PASS: Filters if core ends at or before 0")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nConclusion: The filtering logic correctly handles:")
    print("  • Cores inside the image → KEEP")
    print("  • Cores outside the image → FILTER")
    print("  • Cores at exact boundaries → KEEP (if end <= dimension)")
    print("  • Zero overlap → Works correctly")
    print("  • Large overlap → Works correctly")
    print("  • Negative tile coordinates → Handled properly")
    print("  • Partial outside → Filters correctly")
