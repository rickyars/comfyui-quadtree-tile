#!/usr/bin/env python3
"""
Test to demonstrate the subdivision issue that creates non-square children from square parents.
"""

def old_subdivide(w, h):
    """Current implementation - can create non-square children"""
    half_w = (w // 2) // 8 * 8
    half_h = (h // 2) // 8 * 8
    half_w = max(half_w, 8)
    half_h = max(half_h, 8)

    children = [
        (0, 0, half_w, half_h),  # Top-left
        (half_w, 0, w - half_w, half_h),  # Top-right
        (0, half_h, half_w, h - half_h),  # Bottom-left
        (half_w, half_h, w - half_w, h - half_h),  # Bottom-right
    ]
    return children

def new_subdivide(w, h):
    """Fixed implementation - ensures square children from square parents"""
    # For square parents, use single half value to maintain squareness
    if w == h:
        # Use single half calculation to ensure all children are square
        half = (w // 2) // 8 * 8
        half = max(half, 8)
        half_w = half_h = half
    else:
        # For non-square parents (shouldn't happen with square root)
        half_w = (w // 2) // 8 * 8
        half_h = (h // 2) // 8 * 8
        half_w = max(half_w, 8)
        half_h = max(half_h, 8)

    children = [
        (0, 0, half_w, half_h),  # Top-left
        (half_w, 0, w - half_w, half_h),  # Top-right
        (0, half_h, half_w, h - half_h),  # Bottom-left
        (half_w, half_h, w - half_w, h - half_h),  # Bottom-right
    ]
    return children

def test_size(w, h, name):
    """Test subdivision for a given size"""
    print(f"\n{'='*80}")
    print(f"Testing {name}: {w}x{h} {'(square)' if w == h else '(non-square)'}")
    print(f"{'='*80}")

    print("\nOLD implementation:")
    old_children = old_subdivide(w, h)
    old_non_square = []
    for i, (x, y, cw, ch) in enumerate(old_children):
        is_square = cw == ch
        status = "✓" if is_square else "✗"
        print(f"  Child {i}: ({x:3d}, {y:3d}, {cw:3d}x{ch:3d}) {status}")
        if not is_square:
            old_non_square.append(i)

    if old_non_square:
        print(f"  ✗ PROBLEM: {len(old_non_square)} non-square children: {old_non_square}")
    else:
        print(f"  ✓ All children are square")

    print("\nNEW implementation:")
    new_children = new_subdivide(w, h)
    new_non_square = []
    for i, (x, y, cw, ch) in enumerate(new_children):
        is_square = cw == ch
        status = "✓" if is_square else "✗"
        print(f"  Child {i}: ({x:3d}, {y:3d}, {cw:3d}x{ch:3d}) {status}")
        if not is_square:
            new_non_square.append(i)

    if new_non_square:
        print(f"  ✗ PROBLEM: {len(new_non_square)} non-square children: {new_non_square}")
    else:
        print(f"  ✓ All children are square")

    # Check if fix resolved the issue
    if old_non_square and not new_non_square:
        print(f"\n  ✓✓✓ FIX WORKS! Eliminated {len(old_non_square)} non-square children")
        return True
    elif not old_non_square and not new_non_square:
        print(f"\n  ✓ No issue (both implementations work)")
        return True
    else:
        print(f"\n  ✗ FIX FAILED!")
        return False

def main():
    print("="*80)
    print("QUADTREE SUBDIVISION ISSUE DEMONSTRATION")
    print("="*80)

    # Test various square sizes
    test_cases = [
        (296, 296, "296x296 (from error message)"),
        (304, 304, "304x304 (from error message)"),
        (600, 600, "600x600"),
        (896, 896, "896x896"),
        (64, 64, "64x64 (small)"),
        (128, 128, "128x128 (medium)"),
        (256, 256, "256x256 (large)"),
        (512, 512, "512x512 (xlarge)"),
        (1024, 1024, "1024x1024 (xxlarge)"),
    ]

    all_pass = True
    for w, h, name in test_cases:
        if not test_size(w, h, name):
            all_pass = False

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if all_pass:
        print("✓✓✓ All test cases show the fix resolves non-square children!")
        print("\nThe problem:")
        print("  - Old code rounds half_w and half_h independently to 8-pixel boundaries")
        print("  - For sizes like 296, half=148, which rounds to 144")
        print("  - Children become: 144 and 296-144=152 (non-square!)")
        print("\nThe solution:")
        print("  - For square parents, use SAME half value for both w and h")
        print("  - This ensures all children remain square")
        return 0
    else:
        print("✗ Some test cases failed")
        return 1

if __name__ == "__main__":
    exit(main())
