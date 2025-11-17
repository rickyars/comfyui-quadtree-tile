"""
Test suite for gradient-based variance metric implementation
Validates that the combined metric correctly distinguishes between different types of patterns
"""

import torch
from tiled_vae import QuadtreeBuilder

def test_smooth_gradient():
    """Smooth gradient should have high color variance, low spatial variance"""
    print("=" * 70)
    print("TEST 1: Smooth Gradient")
    print("=" * 70)

    # Create smooth gradient
    gradient = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    gradient = gradient.repeat(1, 3, 100, 1)

    builder_color = QuadtreeBuilder(variance_mode='color')
    builder_gradient = QuadtreeBuilder(variance_mode='gradient')
    builder_combined = QuadtreeBuilder(variance_mode='combined')

    var_color = builder_color.calculate_variance(gradient, 0, 0, 100, 100)
    var_gradient = builder_gradient.calculate_variance(gradient, 0, 0, 100, 100)
    var_combined = builder_combined.calculate_variance(gradient, 0, 0, 100, 100)

    print(f"Smooth gradient (0 -> 1 linear):")
    print(f"  Color variance:    {var_color:.6f}")
    print(f"  Gradient variance: {var_gradient:.6f}")
    print(f"  Combined variance: {var_combined:.6f}")
    print(f"  Ratio (color/gradient): {var_color/var_gradient if var_gradient > 0 else 'inf':.2f}")
    print()

    # Color variance should dominate for smooth gradients
    assert var_color > var_gradient, "Color variance should dominate for smooth gradients"
    print("✓ PASSED: Color variance dominates for smooth gradients")
    print()

def test_sharp_edge():
    """Sharp edge should have high spatial variance"""
    print("=" * 70)
    print("TEST 2: Sharp Edge")
    print("=" * 70)

    # Create sharp edge
    edge = torch.zeros(1, 3, 100, 100)
    edge[:, :, :, 50:] = 1.0

    builder_color = QuadtreeBuilder(variance_mode='color')
    builder_gradient = QuadtreeBuilder(variance_mode='gradient')
    builder_combined = QuadtreeBuilder(variance_mode='combined')

    var_color = builder_color.calculate_variance(edge, 0, 0, 100, 100)
    var_gradient = builder_gradient.calculate_variance(edge, 0, 0, 100, 100)
    var_combined = builder_combined.calculate_variance(edge, 0, 0, 100, 100)

    print(f"Sharp edge (0 | 1 at x=50):")
    print(f"  Color variance:    {var_color:.6f}")
    print(f"  Gradient variance: {var_gradient:.6f}")
    print(f"  Combined variance: {var_combined:.6f}")
    print(f"  Ratio (gradient/color): {var_gradient/var_color if var_color > 0 else 'inf':.2f}")
    print()

    # Gradient variance should be significant for sharp edges
    assert var_gradient > 0.1, "Gradient variance should be high for sharp edges"
    print("✓ PASSED: Gradient variance is high for sharp edges")
    print()

def test_texture():
    """Texture (high freq, low amplitude) should have higher gradient variance"""
    print("=" * 70)
    print("TEST 3: Fine Texture (Checkerboard)")
    print("=" * 70)

    # Create checkerboard texture
    texture = torch.zeros(1, 3, 100, 100)
    for i in range(0, 100, 4):
        for j in range(0, 100, 4):
            if (i // 4 + j // 4) % 2 == 0:
                texture[:, :, i:i+4, j:j+4] = 0.55
            else:
                texture[:, :, i:i+4, j:j+4] = 0.45

    builder_color = QuadtreeBuilder(variance_mode='color')
    builder_gradient = QuadtreeBuilder(variance_mode='gradient')
    builder_combined = QuadtreeBuilder(variance_mode='combined')

    var_color = builder_color.calculate_variance(texture, 0, 0, 100, 100)
    var_gradient = builder_gradient.calculate_variance(texture, 0, 0, 100, 100)
    var_combined = builder_combined.calculate_variance(texture, 0, 0, 100, 100)

    print(f"Texture (4x4 checkerboard, amplitude 0.1):")
    print(f"  Color variance:    {var_color:.6f}")
    print(f"  Gradient variance: {var_gradient:.6f}")
    print(f"  Combined variance: {var_combined:.6f}")
    print(f"  Enhancement: {var_combined/var_color if var_color > 0 else 'inf':.2f}x")
    print()

    # Combined should capture texture better than color alone
    assert var_combined > var_color, "Combined should capture texture better than color alone"
    print("✓ PASSED: Combined metric captures texture better than color alone")
    print()

def test_uniform_area():
    """Uniform area should have low variance in all modes"""
    print("=" * 70)
    print("TEST 4: Uniform Area")
    print("=" * 70)

    # Create uniform area
    uniform = torch.ones(1, 3, 100, 100) * 0.5

    builder_color = QuadtreeBuilder(variance_mode='color')
    builder_gradient = QuadtreeBuilder(variance_mode='gradient')
    builder_combined = QuadtreeBuilder(variance_mode='combined')

    var_color = builder_color.calculate_variance(uniform, 0, 0, 100, 100)
    var_gradient = builder_gradient.calculate_variance(uniform, 0, 0, 100, 100)
    var_combined = builder_combined.calculate_variance(uniform, 0, 0, 100, 100)

    print(f"Uniform area (all pixels = 0.5):")
    print(f"  Color variance:    {var_color:.6f}")
    print(f"  Gradient variance: {var_gradient:.6f}")
    print(f"  Combined variance: {var_combined:.6f}")
    print()

    # All variances should be near zero
    assert var_color < 0.01, "Color variance should be near zero for uniform area"
    assert var_gradient < 0.01, "Gradient variance should be near zero for uniform area"
    assert var_combined < 0.01, "Combined variance should be near zero for uniform area"
    print("✓ PASSED: All variance metrics are near zero for uniform area")
    print()

def test_weight_normalization():
    """Test that weights are properly normalized in combined mode"""
    print("=" * 70)
    print("TEST 5: Weight Normalization")
    print("=" * 70)

    # Create test pattern
    pattern = torch.rand(1, 3, 100, 100)

    # Test with equal weights
    builder_equal = QuadtreeBuilder(variance_mode='combined', color_weight=0.5, gradient_weight=0.5)
    assert abs(builder_equal.color_weight - 0.5) < 1e-6, "Equal weights should be 0.5 each"
    assert abs(builder_equal.gradient_weight - 0.5) < 1e-6, "Equal weights should be 0.5 each"
    print("✓ Equal weights (0.5, 0.5) are normalized correctly")

    # Test with unequal weights
    builder_unequal = QuadtreeBuilder(variance_mode='combined', color_weight=0.7, gradient_weight=0.3)
    assert abs(builder_unequal.color_weight - 0.7) < 1e-6, "Weights should normalize to 0.7"
    assert abs(builder_unequal.gradient_weight - 0.3) < 1e-6, "Weights should normalize to 0.3"
    print("✓ Unequal weights (0.7, 0.3) are normalized correctly")

    # Test with non-normalized input
    builder_unnorm = QuadtreeBuilder(variance_mode='combined', color_weight=2.0, gradient_weight=2.0)
    assert abs(builder_unnorm.color_weight - 0.5) < 1e-6, "Non-normalized weights should be normalized to 0.5"
    assert abs(builder_unnorm.gradient_weight - 0.5) < 1e-6, "Non-normalized weights should be normalized to 0.5"
    print("✓ Non-normalized weights (2.0, 2.0) are normalized to (0.5, 0.5)")

    print()

def test_variance_modes():
    """Test that variance modes correctly set weights"""
    print("=" * 70)
    print("TEST 6: Variance Mode Configuration")
    print("=" * 70)

    # Test color-only mode
    builder_color = QuadtreeBuilder(variance_mode='color')
    assert builder_color.color_weight == 1.0, "Color mode should set color_weight to 1.0"
    assert builder_color.gradient_weight == 0.0, "Color mode should set gradient_weight to 0.0"
    print("✓ Color mode: color_weight=1.0, gradient_weight=0.0")

    # Test gradient-only mode
    builder_gradient = QuadtreeBuilder(variance_mode='gradient')
    assert builder_gradient.color_weight == 0.0, "Gradient mode should set color_weight to 0.0"
    assert builder_gradient.gradient_weight == 1.0, "Gradient mode should set gradient_weight to 1.0"
    print("✓ Gradient mode: color_weight=0.0, gradient_weight=1.0")

    # Test combined mode
    builder_combined = QuadtreeBuilder(variance_mode='combined', color_weight=0.6, gradient_weight=0.4)
    assert abs(builder_combined.color_weight - 0.6) < 1e-6, "Combined mode should use specified weights"
    assert abs(builder_combined.gradient_weight - 0.4) < 1e-6, "Combined mode should use specified weights"
    print("✓ Combined mode: uses specified weights (normalized)")

    print()

def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "GRADIENT METRIC TEST SUITE" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_smooth_gradient()
        test_sharp_edge()
        test_texture()
        test_uniform_area()
        test_weight_normalization()
        test_variance_modes()

        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        return True

    except AssertionError as e:
        print()
        print("=" * 70)
        print("✗ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        return False
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ TEST ERROR")
        print("=" * 70)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
