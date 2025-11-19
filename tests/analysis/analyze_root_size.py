#!/usr/bin/env python3
"""
Analyze the root size calculation to understand why non-square leaves are created.
"""

def current_root_calc(w, h):
    """Current root size calculation"""
    root_size = max(w, h)
    root_size = ((root_size + 15) // 16) * 16
    return root_size

def subdivide_once(size):
    """Simulate one level of subdivision"""
    half = (size // 2) // 8 * 8
    half = max(half, 8)
    remainder = size - half
    return half, remainder

def analyze_size(w, h):
    """Analyze if a size can be subdivided into squares"""
    print(f"\n{'='*80}")
    print(f"Analyzing {w}x{h}")
    print(f"{'='*80}")

    root = current_root_calc(w, h)
    print(f"Root size: {root}x{root}")

    # Simulate subdivision
    half, remainder = subdivide_once(root)
    print(f"\nFirst subdivision:")
    print(f"  half = {half}")
    print(f"  remainder = {remainder}")

    if half == remainder:
        print(f"  ✓ Can create 4 equal {half}x{half} squares")
    else:
        print(f"  ✗ Creates mixed sizes: 2x{half}x{half} and 2x{remainder}x{half} (NON-SQUARE!)")
        print(f"     Also: 2x{half}x{remainder} (NON-SQUARE!)")

    # Check what root size WOULD work
    print(f"\n  Finding root size that allows square subdivision...")
    for test_size in range(root - 32, root + 64, 16):
        h, r = subdivide_once(test_size)
        if h == r:
            print(f"    {test_size}x{test_size} → 4 equal {h}x{h} squares ✓")
            if test_size != root:
                print(f"    SUGGESTION: Use {test_size} instead of {root}")
            break

def main():
    print("="*80)
    print("ROOT SIZE ANALYSIS")
    print("="*80)

    # Test cases from the error
    test_cases = [
        (296, 296),
        (304, 304),
        (600, 600),
        (896, 896),
        (1200, 1200),
    ]

    for w, h in test_cases:
        analyze_size(w, h)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
The issue is that ((size + 15) // 16) * 16 doesn't guarantee that the root
can be subdivided into equal squares with 8-pixel alignment.

For a size to subdivide into 4 equal squares:
  - half = (size // 2) // 8 * 8
  - remainder = size - half
  - Need: half == remainder
  - Therefore: size = 2 * half = 2 * ((size // 2) // 8 * 8)

This means size must be divisible by 16 AND the half-point must land on a
multiple of 8 without remainder.

Better formula:
  - Round size up to nearest multiple of 16
  - Then check if (size // 2) is divisible by 8
  - If not, round up to next valid size
    """)

if __name__ == "__main__":
    main()
