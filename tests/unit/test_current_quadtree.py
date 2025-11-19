#!/usr/bin/env python3
"""
Test script to demonstrate current quadtree behavior with rectangular images
Shows that current implementation creates rectangular tiles
"""

import sys
import torch

# Add current directory to path to import tiled_vae
sys.path.insert(0, '/home/user/comfyui-quadtree-tile')

from tiled_vae import QuadtreeBuilder, QuadtreeNode


def test_current_implementation():
    """Test current quadtree implementation with rectangular image"""

    print("=" * 80)
    print("CURRENT QUADTREE IMPLEMENTATION TEST")
    print("=" * 80)
    print()

    # Test Case 1: 1920×1080 image (16:9 landscape)
    print("TEST CASE 1: 1920×1080 Image (16:9 Landscape)")
    print("-" * 80)

    # Create test tensor
    image_1920x1080 = torch.randn(1, 3, 1080, 1920)

    # Build quadtree
    builder = QuadtreeBuilder(
        content_threshold=0.05,
        max_depth=3,
        min_tile_size=128
    )

    root, leaves = builder.build(image_1920x1080)

    # Analyze root
    print(f"\nROOT NODE:")
    print(f"  Dimensions: {root.w}×{root.h}")
    print(f"  Is Square: {root.w == root.h}")
    if root.w != root.h:
        print(f"  ❌ ROOT IS RECTANGULAR")
        aspect_ratio = root.w / root.h
        print(f"  Aspect ratio: {aspect_ratio:.3f}:1")

    # Analyze children at depth 1
    if root.children:
        print(f"\nDEPTH 1 CHILDREN (after first subdivision):")
        for i, child in enumerate(root.children):
            position = ["Top-left", "Top-right", "Bottom-left", "Bottom-right"][i]
            is_square = child.w == child.h
            print(f"  Child {i} ({position}): {child.w}×{child.h} - {'✅ Square' if is_square else '❌ Rectangular'}")

    # Analyze all leaves
    print(f"\nLEAF NODES (tiles):")
    print(f"  Total leaves: {len(leaves)}")

    # Check squareness
    square_leaves = [leaf for leaf in leaves if leaf.w == leaf.h]
    rectangular_leaves = [leaf for leaf in leaves if leaf.w != leaf.h]

    print(f"  Square leaves: {len(square_leaves)} ({100*len(square_leaves)/len(leaves):.1f}%)")
    print(f"  Rectangular leaves: {len(rectangular_leaves)} ({100*len(rectangular_leaves)/len(leaves):.1f}%)")

    if rectangular_leaves:
        print(f"\n  ❌ PROBLEM: {len(rectangular_leaves)} tiles are RECTANGULAR!")
        print(f"\n  Sample rectangular tiles:")
        for i, leaf in enumerate(rectangular_leaves[:5]):
            aspect = leaf.w / leaf.h
            print(f"    Tile at ({leaf.x}, {leaf.y}): {leaf.w}×{leaf.h} (aspect {aspect:.3f}:1)")
    else:
        print(f"  ✅ All tiles are square!")

    # Check dimensions distribution
    unique_dimensions = set((leaf.w, leaf.h) for leaf in leaves)
    print(f"\n  Unique tile dimensions: {len(unique_dimensions)}")
    for w, h in sorted(unique_dimensions):
        count = sum(1 for leaf in leaves if leaf.w == w and leaf.h == h)
        shape_type = "square" if w == h else "rectangular"
        print(f"    {w}×{h} ({shape_type}): {count} tiles")

    print()

    # Test Case 2: 512×768 image (2:3 portrait)
    print("\n" + "=" * 80)
    print("TEST CASE 2: 512×768 Image (2:3 Portrait)")
    print("-" * 80)

    image_512x768 = torch.randn(1, 3, 768, 512)
    root2, leaves2 = builder.build(image_512x768)

    print(f"\nROOT NODE:")
    print(f"  Dimensions: {root2.w}×{root2.h}")
    print(f"  Is Square: {root2.w == root2.h}")
    if root2.w != root2.h:
        print(f"  ❌ ROOT IS RECTANGULAR")

    if root2.children:
        print(f"\nDEPTH 1 CHILDREN:")
        for i, child in enumerate(root2.children):
            position = ["Top-left", "Top-right", "Bottom-left", "Bottom-right"][i]
            is_square = child.w == child.h
            print(f"  Child {i} ({position}): {child.w}×{child.h} - {'✅ Square' if is_square else '❌ Rectangular'}")

    print(f"\nLEAF NODES:")
    print(f"  Total leaves: {len(leaves2)}")
    square_leaves2 = [leaf for leaf in leaves2 if leaf.w == leaf.h]
    rectangular_leaves2 = [leaf for leaf in leaves2 if leaf.w != leaf.h]
    print(f"  Square: {len(square_leaves2)}, Rectangular: {len(rectangular_leaves2)}")

    if rectangular_leaves2:
        print(f"  ❌ PROBLEM: {len(rectangular_leaves2)} tiles are RECTANGULAR!")

    print()

    # Test Case 3: 1024×1024 image (square)
    print("\n" + "=" * 80)
    print("TEST CASE 3: 1024×1024 Image (Square)")
    print("-" * 80)

    image_1024x1024 = torch.randn(1, 3, 1024, 1024)
    root3, leaves3 = builder.build(image_1024x1024)

    print(f"\nROOT NODE:")
    print(f"  Dimensions: {root3.w}×{root3.h}")
    print(f"  Is Square: {root3.w == root3.h} {'✅' if root3.w == root3.h else '❌'}")

    print(f"\nLEAF NODES:")
    print(f"  Total leaves: {len(leaves3)}")
    square_leaves3 = [leaf for leaf in leaves3 if leaf.w == leaf.h]
    rectangular_leaves3 = [leaf for leaf in leaves3 if leaf.w != leaf.h]
    print(f"  Square: {len(square_leaves3)}, Rectangular: {len(rectangular_leaves3)}")

    if len(square_leaves3) == len(leaves3):
        print(f"  ✅ All tiles are square! (Square image works correctly)")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Current implementation behavior:")
    print("  • Rectangular images → Rectangular root → Rectangular tiles ❌")
    print("  • Square images → Square root → Square tiles ✅")
    print()
    print("Conclusion: Current implementation creates RECTANGULAR tiles")
    print("            for rectangular images, violating user requirement.")
    print()
    print("See RESEARCH_SUMMARY.md for proposed solution.")
    print()


if __name__ == "__main__":
    test_current_implementation()
