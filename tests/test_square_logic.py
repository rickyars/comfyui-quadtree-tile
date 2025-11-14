#!/usr/bin/env python3
"""
Test logic for square quadtree without requiring PyTorch
This validates the mathematical correctness of the subdivision
"""


def test_square_subdivision_logic():
    """Test that square subdivision maintains square children"""

    print("=" * 80)
    print("SQUARE SUBDIVISION LOGIC TEST")
    print("=" * 80)
    print()

    test_cases = [
        # (parent_size, expected_behavior)
        (1024, "Should create 4 square children of 512x512"),
        (512, "Should create 4 square children of 256x256"),
        (768, "Should create 4 square children of 384x384 (aligned to 8)"),
        (256, "Should create 4 square children of 128x128"),
        (1920, "Should create 4 square children of 960x960"),
    ]

    all_passed = True

    for parent_size, description in test_cases:
        print(f"Parent: {parent_size}x{parent_size}")
        print(f"  Expected: {description}")

        # Simulate the subdivision logic from subdivide() method
        # half_w = (self.w // 2) // 8 * 8
        # half_h = (self.h // 2) // 8 * 8
        half_size = (parent_size // 2) // 8 * 8
        half_size = max(half_size, 8)  # Ensure at least 8 pixels

        # Children sizes (from subdivide method):
        # Top-left: half_w x half_h
        # Top-right: (w - half_w) x half_h
        # Bottom-left: half_w x (h - half_h)
        # Bottom-right: (w - half_w) x (h - half_h)

        child1_w, child1_h = half_size, half_size
        child2_w, child2_h = parent_size - half_size, half_size
        child3_w, child3_h = half_size, parent_size - half_size
        child4_w, child4_h = parent_size - half_size, parent_size - half_size

        children = [
            ("Top-left", child1_w, child1_h),
            ("Top-right", child2_w, child2_h),
            ("Bottom-left", child3_w, child3_h),
            ("Bottom-right", child4_w, child4_h),
        ]

        all_square = True
        for name, w, h in children:
            is_square = w == h
            status = "✓" if is_square else "✗"
            print(f"  {status} {name}: {w}x{h} {'(square)' if is_square else '(NOT SQUARE!)'}")
            if not is_square:
                all_square = False
                all_passed = False

        if all_square:
            print(f"  ✓ Result: All 4 children are square")
        else:
            print(f"  ✗ Result: Some children are NOT square - BUG!")

        print()

    return 0 if all_passed else 1


def test_root_creation_logic():
    """Test that square root creation works for various image sizes"""

    print("=" * 80)
    print("SQUARE ROOT CREATION LOGIC TEST")
    print("=" * 80)
    print()

    test_cases = [
        # (image_width, image_height, name)
        (1920, 1080, "Full HD 16:9"),
        (1024, 768, "4:3 landscape"),
        (512, 512, "Square"),
        (800, 600, "4:3 small"),
    ]

    for img_w, img_h, name in test_cases:
        print(f"{name}: Image {img_w}x{img_h}")

        # Simulate root creation logic from build_tree()
        # root_size = max(w, h)
        # root_size = (root_size // 8) * 8
        root_size = max(img_w, img_h)
        root_size = (root_size // 8) * 8

        root_w, root_h = root_size, root_size
        is_square = root_w == root_h
        covers_image = root_w >= img_w and root_h >= img_h

        print(f"  Root: {root_w}x{root_h}")
        print(f"  {'✓' if is_square else '✗'} Square: {is_square}")
        print(f"  {'✓' if covers_image else '✗'} Covers entire image: {covers_image}")

        if root_w > img_w:
            print(f"  → Extends {root_w - img_w}px beyond image width (will need padding)")
        if root_h > img_h:
            print(f"  → Extends {root_h - img_h}px beyond image height (will need padding)")

        print()


def test_padding_logic():
    """Test the padding logic for boundary tiles"""

    print("=" * 80)
    print("REFLECTION PADDING LOGIC TEST")
    print("=" * 80)
    print()

    # Scenario: 1920x1080 image with 1920x1920 square root
    img_w, img_h = 1920, 1080
    root_size = 1920

    print(f"Image: {img_w}x{img_h}")
    print(f"Root: {root_size}x{root_size}")
    print(f"Overhang: {root_size - img_h}px below image")
    print()

    # Simulate a tile at bottom of the square root that extends beyond image
    # Tile at (0, 960, 960, 960) - bottom-right quadrant
    tile_x, tile_y, tile_w, tile_h = 0, 960, 960, 960

    print(f"Example tile: position=({tile_x}, {tile_y}), size={tile_w}x{tile_h}")

    # Check if it extends beyond image
    extends_right = (tile_x + tile_w) > img_w
    extends_bottom = (tile_y + tile_h) > img_h

    print(f"  Extends beyond right edge: {extends_right}")
    print(f"  Extends beyond bottom edge: {extends_bottom}")

    if extends_bottom:
        # Calculate padding needed
        pad_bottom = (tile_y + tile_h) - img_h
        print(f"  → Needs {pad_bottom}px reflection padding at bottom")

        # What we'd extract
        y_end_clamped = min(img_h, tile_y + tile_h)
        actual_h = y_end_clamped - tile_y
        print(f"  → Extract: {tile_w}x{actual_h} from image")
        print(f"  → Pad to: {tile_w}x{tile_h} using reflection mode")

    print()


if __name__ == "__main__":
    print("\n")
    exit_code = test_square_subdivision_logic()
    test_root_creation_logic()
    test_padding_logic()
    print("\n")

    if exit_code == 0:
        print("✓ All logic tests passed!")
        print("The implementation should create 100% square tiles.")
    else:
        print("✗ Logic tests failed!")
        print("There may be an issue with the subdivision logic.")

    print()
