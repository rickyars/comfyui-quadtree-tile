#!/usr/bin/env python3
"""
Test script to verify that quadtree tiles are 100% square
Tests various image sizes with Approach A (Square Root + Reflection Padding)
"""

import torch
import sys
import os

# Add the module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiled_vae import QuadtreeBuilder, QuadtreeNode


def test_square_quadtree():
    """Test that all quadtree tiles are square across various image sizes"""

    # Test various image sizes (W, H) including non-square images
    test_cases = [
        # (width, height, name)
        (1920, 1080, "Full HD 16:9"),
        (512, 768, "Portrait 2:3"),
        (1024, 1024, "Square 1024x1024"),
        (800, 600, "Landscape 4:3"),
        (2560, 1440, "2K 16:9"),
        (640, 480, "VGA 4:3"),
        (1280, 720, "HD 720p"),
        (768, 512, "Landscape 3:2"),
    ]

    print("=" * 80)
    print("SQUARE QUADTREE TEST - Approach A (Square Root + Reflection Padding)")
    print("=" * 80)
    print()

    all_passed = True
    results = []

    for width, height, name in test_cases:
        print(f"Testing {name}: {width}x{height}")
        print("-" * 80)

        # Create a fake image tensor (just zeros, we only care about structure)
        fake_image = torch.zeros(1, 3, height, width)

        # Build quadtree with various parameters
        builder = QuadtreeBuilder(
            content_threshold=0.05,  # Moderate threshold
            max_depth=4,             # Allow 4 levels of subdivision
            min_tile_size=128,       # Minimum 128px tiles
            min_denoise=0.0,
            max_denoise=1.0
        )

        try:
            root, leaves = builder.build(fake_image)

            # Check root is square
            root_square = root.w == root.h
            print(f"  Root node: {root.w}x{root.h} {'✓ SQUARE' if root_square else '✗ NOT SQUARE'}")

            if not root_square:
                all_passed = False
                results.append((name, width, height, False, f"Root not square: {root.w}x{root.h}"))
                continue

            # Check all leaves are square
            non_square_leaves = []
            for leaf in leaves:
                if leaf.w != leaf.h:
                    non_square_leaves.append((leaf.x, leaf.y, leaf.w, leaf.h))

            if non_square_leaves:
                all_passed = False
                print(f"  ✗ FAILED: Found {len(non_square_leaves)} non-square leaves:")
                for x, y, w, h in non_square_leaves[:5]:  # Show first 5
                    print(f"    - Position ({x}, {y}): {w}x{h}")
                results.append((name, width, height, False, f"{len(non_square_leaves)} non-square leaves"))
            else:
                print(f"  ✓ PASSED: All {len(leaves)} leaf tiles are square")

                # Show tile size distribution
                tile_sizes = set((leaf.w, leaf.h) for leaf in leaves)
                print(f"  Tile sizes: {sorted(tile_sizes)}")

                # Show some example tiles
                print(f"  Example tiles:")
                for i, leaf in enumerate(leaves[:5]):
                    print(f"    Tile {i}: {leaf.w}x{leaf.h} at ({leaf.x},{leaf.y}) depth={leaf.depth}")

                results.append((name, width, height, True, f"{len(leaves)} square tiles"))

        except Exception as e:
            all_passed = False
            print(f"  ✗ ERROR: {e}")
            results.append((name, width, height, False, str(e)))

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, width, height, passed, details in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name} ({width}x{height}) - {details}")

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED! All quadtree tiles are square.")
        return 0
    else:
        print("❌ SOME TESTS FAILED! Not all tiles are square.")
        return 1


def test_coverage():
    """Test that square quadtree tiles provide full coverage"""
    print("=" * 80)
    print("COVERAGE TEST - Verify no gaps in image coverage")
    print("=" * 80)
    print()

    test_sizes = [(1920, 1080), (1024, 768)]

    for width, height in test_sizes:
        print(f"Testing coverage for {width}x{height}")

        # Create coverage map
        coverage = torch.zeros(height, width, dtype=torch.int32)

        fake_image = torch.zeros(1, 3, height, width)
        builder = QuadtreeBuilder(
            content_threshold=0.05,
            max_depth=4,
            min_tile_size=128
        )

        root, leaves = builder.build(fake_image)

        # Mark each pixel covered by each tile
        for leaf in leaves:
            x1, x2, y1, y2 = leaf.x, leaf.x + leaf.w, leaf.y, leaf.y + leaf.h

            # Clamp to image boundaries
            x1 = max(0, x1)
            x2 = min(width, x2)
            y1 = max(0, y1)
            y2 = min(height, y2)

            if x2 > x1 and y2 > y1:
                coverage[y1:y2, x1:x2] += 1

        # Check for uncovered pixels
        uncovered = (coverage == 0).sum().item()
        total_pixels = height * width

        if uncovered > 0:
            print(f"  ✗ FAILED: {uncovered}/{total_pixels} pixels not covered ({100*uncovered/total_pixels:.2f}%)")
        else:
            print(f"  ✓ PASSED: All pixels covered")

            # Show coverage statistics
            min_coverage = coverage.min().item()
            max_coverage = coverage.max().item()
            avg_coverage = coverage.float().mean().item()
            print(f"  Coverage: min={min_coverage}, max={max_coverage}, avg={avg_coverage:.2f}")
        print()


if __name__ == "__main__":
    print("\n")
    exit_code = test_square_quadtree()
    print("\n")
    test_coverage()
    print("\n")
    sys.exit(exit_code)
