#!/usr/bin/env python3
"""
Test that the new root size calculation ensures all leaves are square.
"""

import math

def old_root_size(w, h):
    """Old implementation - just round to 16"""
    root_size = max(w, h)
    root_size = ((root_size + 15) // 16) * 16
    return root_size

def new_root_size(w, h):
    """New implementation - power-of-2 multiple of 8"""
    root_size = max(w, h)
    if root_size <= 8:
        root_size = 8
    else:
        n = math.ceil(math.log2(root_size / 8))
        root_size = 8 * (2 ** n)
    return root_size

def subdivide(size):
    """Subdivide a square with 8-pixel alignment"""
    half = (size // 2) // 8 * 8
    half = max(half, 8)
    remainder = size - half
    return half, remainder

def test_recursive_subdivision(root_size, max_depth=5):
    """Test if a root size can subdivide into squares recursively"""
    sizes_tested = set()
    queue = [(root_size, 0)]
    non_square_found = []

    while queue:
        size, depth = queue.pop(0)

        if depth >= max_depth or size < 16:
            continue

        if size in sizes_tested:
            continue
        sizes_tested.add(size)

        half, remainder = subdivide(size)

        if half != remainder:
            non_square_found.append((size, half, remainder, depth))
            # Don't recurse further from non-square subdivision
        else:
            # Both children are same size, recurse into just one
            queue.append((half, depth + 1))

    return non_square_found

def test_size(w, h, name):
    """Test a specific image size"""
    print(f"\n{'='*80}")
    print(f"Testing {name}: {w}x{h}")
    print(f"{'='*80}")

    old_root = old_root_size(w, h)
    new_root = new_root_size(w, h)

    print(f"Old root size: {old_root}x{old_root}")
    print(f"New root size: {new_root}x{new_root}")

    if old_root != new_root:
        print(f"  → Changed from {old_root} to {new_root}")

    # Test old implementation
    print(f"\nOld implementation subdivision test:")
    old_problems = test_recursive_subdivision(old_root)
    if old_problems:
        print(f"  ✗ Found {len(old_problems)} non-square subdivisions:")
        for size, half, remainder, depth in old_problems[:3]:
            print(f"    Level {depth}: {size} → {half},{remainder} (NON-SQUARE!)")
    else:
        print(f"  ✓ All subdivisions produce squares")

    # Test new implementation
    print(f"\nNew implementation subdivision test:")
    new_problems = test_recursive_subdivision(new_root)
    if new_problems:
        print(f"  ✗ Found {len(new_problems)} non-square subdivisions:")
        for size, half, remainder, depth in new_problems[:3]:
            print(f"    Level {depth}: {size} → {half},{remainder} (NON-SQUARE!)")
        return False
    else:
        print(f"  ✓ All subdivisions produce squares")

        # Show subdivision chain
        print(f"\n  Subdivision chain:")
        size = new_root
        for level in range(5):
            half, remainder = subdivide(size)
            if half != remainder or size < 16:
                break
            print(f"    Level {level}: {size} → {half} (4 equal squares)")
            size = half

        return True

def main():
    print("="*80)
    print("SQUARE ROOT SIZE FIX VERIFICATION")
    print("="*80)

    # Test cases from user's error + common sizes
    test_cases = [
        (296, 296, "296x296 (from error)"),
        (304, 304, "304x304 (from error)"),
        (600, 600, "600x600 (from error)"),
        (896, 896, "896x896 (from error)"),
        (1200, 1200, "1200x1200 (large)"),
        (1920, 1080, "1920x1080 (Full HD)"),
        (2048, 2048, "2048x2048 (2K square)"),
        (3840, 2160, "3840x2160 (4K)"),
        (512, 512, "512x512 (already power-of-2)"),
        (1024, 768, "1024x768 (XGA)"),
    ]

    all_pass = True
    for w, h, name in test_cases:
        if not test_size(w, h, name):
            all_pass = False

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if all_pass:
        print("✓✓✓ ALL TESTS PASSED!")
        print("\nThe fix successfully ensures:")
        print("  1. Root size is a power-of-2 multiple of 8")
        print("  2. All recursive subdivisions produce equal squares")
        print("  3. No non-square leaves will be created")
        print("\nValid root sizes: 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192...")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
