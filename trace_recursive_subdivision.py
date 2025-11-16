#!/usr/bin/env python3
"""
Trace recursive subdivision to find where non-square leaves appear.
"""

def subdivide(size, depth=0, max_depth=3):
    """Recursively subdivide and track square-ness"""
    indent = "  " * depth
    half = (size // 2) // 8 * 8
    half = max(half, 8)
    remainder = size - half

    print(f"{indent}{size}x{size} subdivides into:")
    print(f"{indent}  - half = {half}")
    print(f"{indent}  - remainder = {remainder}")

    children = [
        (0, 0, half, half),
        (half, 0, remainder, half),
        (0, half, half, remainder),
        (half, half, remainder, remainder),
    ]

    non_square = []
    for i, (x, y, w, h) in enumerate(children):
        is_square = w == h
        status = "✓" if is_square else "✗"
        print(f"{indent}  Child {i}: ({x:3d}, {y:3d}, {w:3d}x{h:3d}) {status}")
        if not is_square:
            non_square.append((x, y, w, h))

    if non_square:
        print(f"{indent}  ✗✗✗ {len(non_square)} NON-SQUARE children!")
        return non_square

    if depth < max_depth:
        print(f"{indent}  Recursing into children...")
        all_non_square = []
        # Recurse into unique child sizes
        for w, h in set((c[2], c[3]) for c in children):
            if w == h:  # Only recurse into square children
                result = subdivide(w, depth + 1, max_depth)
                if result:
                    all_non_square.extend(result)
        return all_non_square

    return []

def find_good_size(target):
    """Find nearest 'good' size that can recursively subdivide into squares"""
    print(f"\nFinding good size for target {target}:")

    # Try powers of 2 times 8
    for exp in range(3, 12):  # 8*2^3=64 to 8*2^11=16384
        size = 8 * (2 ** exp)
        if size >= target:
            print(f"  Nearest power-of-2 multiple of 8: {size}")
            print(f"  This is 8 * 2^{exp}")

            # Verify it subdivides cleanly
            test_size = size
            levels = []
            for level in range(5):
                half = (test_size // 2) // 8 * 8
                remainder = test_size - half
                levels.append((test_size, half, remainder, half == remainder))
                if half == remainder:
                    test_size = half
                else:
                    break

            print(f"  Subdivision chain:")
            for i, (s, h, r, ok) in enumerate(levels):
                status = "✓" if ok else "✗"
                print(f"    Level {i}: {s} → {h},{r} {status}")

            return size

def main():
    print("="*80)
    print("RECURSIVE SUBDIVISION TRACE")
    print("="*80)

    # Test the problematic 600x600 case
    print("\nCase 1: 600x600 (from 1200x1200 root)")
    print("="*80)
    non_square = subdivide(600, depth=0, max_depth=2)
    if non_square:
        print(f"\n✗✗✗ Found {len(non_square)} non-square leaves in tree!")
        print("First few:")
        for x, y, w, h in non_square[:5]:
            print(f"  ({x}, {y}): {w}x{h}")

    print("\n" + "="*80)
    print("Case 2: 304x304 (should work)")
    print("="*80)
    non_square = subdivide(304, depth=0, max_depth=2)
    if not non_square:
        print("\n✓✓✓ All leaves are square!")

    print("\n" + "="*80)
    print("SOLUTION: Use power-of-2 multiples of 8")
    print("="*80)

    test_targets = [296, 600, 896, 1200]
    for target in test_targets:
        find_good_size(target)

if __name__ == "__main__":
    main()
