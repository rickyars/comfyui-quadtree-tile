#!/usr/bin/env python3
"""
Edge case tests for square quadtree implementation
Tests scenarios identified by QA Agent:
- Sizes not divisible by 16
- Extreme aspect ratios
- Out-of-bounds tile handling
"""

import torch
import sys
import os

# Add the module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiled_vae import QuadtreeBuilder


def test_non_16_multiples():
    """Test sizes not divisible by 16 (Bug #1 validation)"""
    print("=" * 80)
    print("TEST 1: Sizes Not Divisible by 16")
    print("=" * 80)
    print()

    test_cases = [
        (1000, 800, "Non-16-aligned"),
        (1023, 767, "Odd dimensions"),
        (1500, 1000, "Larger non-16"),
        (999, 666, "Strange sizes"),
    ]

    all_passed = True

    for width, height, name in test_cases:
        print(f"Testing {name}: {width}x{height}")
        fake_image = torch.zeros(1, 3, height, width)
        builder = QuadtreeBuilder(
            content_threshold=0.05,
            max_depth=4,
            min_tile_size=128
        )

        try:
            root, leaves = builder.build(fake_image)

            # Check root is square
            if root.w != root.h:
                print(f"  ✗ FAILED: Root not square: {root.w}x{root.h}")
                all_passed = False
                continue

            # Check root is divisible by 16
            if root.w % 16 != 0:
                print(f"  ✗ FAILED: Root not divisible by 16: {root.w}")
                all_passed = False
                continue

            # Check all leaves are square
            non_square = [leaf for leaf in leaves if leaf.w != leaf.h]
            if non_square:
                print(f"  ✗ FAILED: {len(non_square)} non-square leaves")
                for leaf in non_square[:3]:
                    print(f"    - {leaf.w}x{leaf.h} at ({leaf.x},{leaf.y})")
                all_passed = False
            else:
                print(f"  ✓ PASSED: Root={root.w}x{root.h}, {len(leaves)} square leaves")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False

        print()

    return all_passed


def test_extreme_aspect_ratios():
    """Test extreme aspect ratios (QA Warning #1)"""
    print("=" * 80)
    print("TEST 2: Extreme Aspect Ratios")
    print("=" * 80)
    print()

    test_cases = [
        (3840, 1080, "Ultra-wide 21:9"),
        (1080, 3840, "Ultra-tall 9:21"),
        (4096, 1024, "Extreme 4:1"),
        (1024, 4096, "Extreme 1:4"),
    ]

    all_passed = True

    for width, height, name in test_cases:
        aspect_ratio = max(width, height) / min(width, height)
        print(f"Testing {name}: {width}x{height} (aspect ratio {aspect_ratio:.1f}:1)")

        fake_image = torch.zeros(1, 3, height, width)
        builder = QuadtreeBuilder(
            content_threshold=0.05,
            max_depth=4,
            min_tile_size=128
        )

        try:
            root, leaves = builder.build(fake_image)

            # Check all leaves are square
            non_square = [leaf for leaf in leaves if leaf.w != leaf.h]
            if non_square:
                print(f"  ✗ FAILED: {len(non_square)} non-square leaves")
                all_passed = False
            else:
                # Calculate padding overhead
                root_area = root.w * root.h
                image_area = width * height
                overhead = (root_area - image_area) / image_area * 100

                print(f"  ✓ PASSED: Root={root.w}x{root.h}, {len(leaves)} square leaves")
                print(f"  Padding overhead: {overhead:.1f}% (root extends beyond image)")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False

        print()

    return all_passed


def test_out_of_bounds_filtering():
    """Test that out-of-bounds leaves are filtered (Bug #2 validation)"""
    print("=" * 80)
    print("TEST 3: Out-of-Bounds Leaf Filtering")
    print("=" * 80)
    print()

    # Use non-square image to create extended root
    width, height = 1024, 768
    print(f"Testing {width}x{height} (root will extend to 1024x1024)")

    fake_image = torch.zeros(1, 3, height, width)
    builder = QuadtreeBuilder(
        content_threshold=0.05,
        max_depth=4,
        min_tile_size=128
    )

    try:
        root, leaves = builder.build(fake_image)

        print(f"  Root: {root.w}x{root.h}")
        print(f"  Leaf count: {len(leaves)}")

        # Check that no leaves are completely out of bounds
        out_of_bounds = []
        for leaf in leaves:
            # Completely outside if:
            # - starts after image ends: leaf.x >= width or leaf.y >= height
            # - ends before image starts: leaf.x + leaf.w <= 0 or leaf.y + leaf.h <= 0
            if (leaf.x >= width or leaf.y >= height or
                leaf.x + leaf.w <= 0 or leaf.y + leaf.h <= 0):
                out_of_bounds.append(leaf)

        if out_of_bounds:
            print(f"  ✗ FAILED: Found {len(out_of_bounds)} completely out-of-bounds leaves:")
            for leaf in out_of_bounds[:3]:
                print(f"    - {leaf.w}x{leaf.h} at ({leaf.x},{leaf.y})")
            return False
        else:
            print(f"  ✓ PASSED: All leaves overlap with image")

            # Show partial out-of-bounds leaves (these are OK, will be padded)
            partial_oob = []
            for leaf in leaves:
                if (leaf.x + leaf.w > width or leaf.y + leaf.h > height or
                    leaf.x < 0 or leaf.y < 0):
                    partial_oob.append(leaf)

            if partial_oob:
                print(f"  Info: {len(partial_oob)} leaves extend beyond image (will be padded)")
                for leaf in partial_oob[:3]:
                    print(f"    - {leaf.w}x{leaf.h} at ({leaf.x},{leaf.y})")

            return True

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def test_minimum_tile_sizes():
    """Test that minimum tile sizes are respected"""
    print("=" * 80)
    print("TEST 4: Minimum Tile Size Enforcement")
    print("=" * 80)
    print()

    width, height = 1024, 768
    min_tile_size = 256

    print(f"Testing {width}x{height} with min_tile_size={min_tile_size}")

    fake_image = torch.zeros(1, 3, height, width)
    builder = QuadtreeBuilder(
        content_threshold=0.05,
        max_depth=10,  # Very high, would create tiny tiles without min_tile_size
        min_tile_size=min_tile_size
    )

    try:
        root, leaves = builder.build(fake_image)

        # Check no tiles are smaller than min_tile_size
        too_small = [leaf for leaf in leaves if leaf.w < min_tile_size or leaf.h < min_tile_size]

        if too_small:
            print(f"  ✗ FAILED: {len(too_small)} tiles smaller than {min_tile_size}px:")
            for leaf in too_small[:3]:
                print(f"    - {leaf.w}x{leaf.h} at ({leaf.x},{leaf.y})")
            return False
        else:
            tile_sizes = sorted(set(leaf.w for leaf in leaves))
            print(f"  ✓ PASSED: All {len(leaves)} tiles >= {min_tile_size}px")
            print(f"  Tile sizes: {tile_sizes}")
            return True

    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


if __name__ == "__main__":
    print("\n")

    results = []
    results.append(("Non-16 Multiples", test_non_16_multiples()))
    results.append(("Extreme Aspect Ratios", test_extreme_aspect_ratios()))
    results.append(("Out-of-Bounds Filtering", test_out_of_bounds_filtering()))
    results.append(("Minimum Tile Sizes", test_minimum_tile_sizes()))

    print("\n")
    print("=" * 80)
    print("EDGE CASE TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL EDGE CASE TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME EDGE CASE TESTS FAILED!")
        sys.exit(1)
