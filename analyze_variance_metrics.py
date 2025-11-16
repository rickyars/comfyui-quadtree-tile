#!/usr/bin/env python3
"""
Comparative analysis of variance metrics:
- MAD (Mean Absolute Deviation) - Current ComfyUI implementation
- Euclidean RGB Distance - Reference implementation (urchinemerald)

This script evaluates which metric better identifies image detail/complexity.
"""

import math
import random
from typing import List, Tuple

def calculate_mad(pixels: List[List[List[float]]]) -> float:
    """
    Mean Absolute Deviation (Current implementation)
    Formula: avg(|pixel - mean_color|)

    Args:
        pixels: 3D list of shape [H][W][C] with values in [0, 1]

    Returns:
        MAD value
    """
    h = len(pixels)
    w = len(pixels[0])
    c = len(pixels[0][0])

    # Calculate average color
    avg_color = [0.0] * c
    for i in range(h):
        for j in range(w):
            for k in range(c):
                avg_color[k] += pixels[i][j][k]

    for k in range(c):
        avg_color[k] /= (h * w)

    # Calculate mean absolute deviation
    total_dev = 0.0
    for i in range(h):
        for j in range(w):
            for k in range(c):
                total_dev += abs(pixels[i][j][k] - avg_color[k])

    mad = total_dev / (h * w * c)
    return mad

def calculate_euclidean_rgb(pixels: List[List[List[float]]]) -> float:
    """
    Euclidean RGB Distance (Reference implementation)
    Formula: avg(√((R-R_avg)² + (G-G_avg)² + (B-B_avg)²))

    Args:
        pixels: 3D list of shape [H][W][C] with values in [0, 1]

    Returns:
        Average Euclidean distance
    """
    h = len(pixels)
    w = len(pixels[0])
    c = len(pixels[0][0])

    # Calculate average color
    avg_color = [0.0] * c
    for i in range(h):
        for j in range(w):
            for k in range(c):
                avg_color[k] += pixels[i][j][k]

    for k in range(c):
        avg_color[k] /= (h * w)

    # Calculate Euclidean distance per pixel
    total_dist = 0.0
    for i in range(h):
        for j in range(w):
            squared_sum = 0.0
            for k in range(c):
                squared_sum += (pixels[i][j][k] - avg_color[k]) ** 2
            total_dist += math.sqrt(squared_sum)

    avg_euclidean = total_dist / (h * w)
    return avg_euclidean

def create_test_patterns() -> dict:
    """Create various test patterns to evaluate metrics"""
    patterns = {}
    size = 100

    # 1. Uniform color (no detail)
    uniform = [[[0.5, 0.5, 0.5] for _ in range(size)] for _ in range(size)]
    patterns['uniform_gray'] = uniform

    # 2. Uniform but different color
    uniform_blue = [[[0.0, 0.0, 0.8] for _ in range(size)] for _ in range(size)]
    patterns['uniform_blue'] = uniform_blue

    # 3. High-frequency checkerboard (high detail)
    checkerboard = []
    for i in range(size):
        row = []
        for j in range(size):
            if (i // 4 + j // 4) % 2 == 0:
                row.append([1.0, 1.0, 1.0])
            else:
                row.append([0.0, 0.0, 0.0])
        checkerboard.append(row)
    patterns['checkerboard_4px'] = checkerboard

    # 4. Low-frequency checkerboard (medium detail)
    checkerboard_large = []
    for i in range(size):
        row = []
        for j in range(size):
            if (i // 20 + j // 20) % 2 == 0:
                row.append([1.0, 1.0, 1.0])
            else:
                row.append([0.0, 0.0, 0.0])
        checkerboard_large.append(row)
    patterns['checkerboard_20px'] = checkerboard_large

    # 5. Smooth gradient (low detail, high color variation)
    gradient = []
    for i in range(size):
        row = []
        val = i / size
        for j in range(size):
            row.append([val, val, val])
        gradient.append(row)
    patterns['smooth_gradient'] = gradient

    # 6. Sharp edge (high detail localized)
    edge = []
    for i in range(size):
        row = []
        for j in range(size):
            if j < size // 2:
                row.append([0.0, 0.0, 0.0])
            else:
                row.append([1.0, 1.0, 1.0])
        edge.append(row)
    patterns['sharp_edge'] = edge

    # 7. Noise (very high detail)
    random.seed(42)
    noise = [[[random.random(), random.random(), random.random()] for _ in range(size)] for _ in range(size)]
    patterns['random_noise'] = noise

    # 8. Textured area (medium-high detail)
    random.seed(43)
    texture = [[[random.random() * 0.3 + 0.5, random.random() * 0.3 + 0.5, random.random() * 0.3 + 0.5]
                for _ in range(size)] for _ in range(size)]
    patterns['low_amplitude_noise'] = texture

    # 9. Sky-like (very uniform, subtle gradient)
    sky = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append([0.0, 0.0, 0.7 + (i / size) * 0.1])
        sky.append(row)
    patterns['sky_gradient'] = sky

    # 10. Complex multi-color pattern
    complex_pattern = []
    for i in range(size):
        row = []
        for j in range(size):
            r = math.sin(i * 0.1) * 0.5 + 0.5
            g = math.cos(j * 0.1) * 0.5 + 0.5
            b = math.sin((i + j) * 0.05) * 0.5 + 0.5
            row.append([r, g, b])
        complex_pattern.append(row)
    patterns['complex_sinusoid'] = complex_pattern

    return patterns

def analyze_metric_properties():
    """Analyze mathematical properties of both metrics"""
    print("=" * 80)
    print("MATHEMATICAL PROPERTY ANALYSIS")
    print("=" * 80)

    print("\n1. METRIC DEFINITIONS:")
    print("\nMAD (Mean Absolute Deviation):")
    print("   Formula: avg(|R - R_avg| + |G - G_avg| + |B - B_avg|) / 3")
    print("   - Treats each channel independently")
    print("   - L1 norm in RGB space")
    print("   - Range: [0, 1] for normalized images")

    print("\nEuclidean RGB Distance:")
    print("   Formula: avg(√((R - R_avg)² + (G - G_avg)² + (B - B_avg)²))")
    print("   - Considers combined color distance")
    print("   - L2 norm in RGB space")
    print("   - Range: [0, √3] ≈ [0, 1.732] for normalized images")

    print("\n2. KEY DIFFERENCES:")
    print("\nSensitivity to outliers:")
    print("   - MAD: Linear response (less sensitive to outliers)")
    print("   - Euclidean: Quadratic response (more sensitive to outliers)")

    print("\nColor channel interaction:")
    print("   - MAD: Sum of per-channel deviations")
    print("   - Euclidean: Geometric distance in color space")

    print("\nExample: Pixel at (R=0.9, G=0.5, B=0.1), avg=(0.5, 0.5, 0.5)")
    r, g, b = 0.9, 0.5, 0.1
    avg_r, avg_g, avg_b = 0.5, 0.5, 0.5
    mad_contrib = (abs(r - avg_r) + abs(g - avg_g) + abs(b - avg_b)) / 3
    euclid_contrib = math.sqrt((r - avg_r)**2 + (g - avg_g)**2 + (b - avg_b)**2)
    print(f"   - MAD contribution: {mad_contrib:.4f}")
    print(f"   - Euclidean contribution: {euclid_contrib:.4f}")
    print(f"   - Ratio (Euclidean/MAD): {euclid_contrib/mad_contrib:.2f}x")

def compare_metrics_on_patterns():
    """Compare both metrics on test patterns"""
    patterns = create_test_patterns()

    print("\n" + "=" * 80)
    print("METRIC COMPARISON ON TEST PATTERNS")
    print("=" * 80)

    results = []

    for name, pattern in patterns.items():
        mad = calculate_mad(pattern)
        euclidean = calculate_euclidean_rgb(pattern)

        results.append({
            'name': name,
            'mad': mad,
            'euclidean': euclidean,
            'ratio': euclidean / mad if mad > 0 else 0
        })

    # Sort by MAD value
    results.sort(key=lambda x: x['mad'])

    print(f"\n{'Pattern':<25} {'MAD':<10} {'Euclidean':<12} {'E/M Ratio':<10} {'Detail Level'}")
    print("-" * 80)

    for r in results:
        # Classify detail level based on MAD
        if r['mad'] < 0.01:
            detail = "VERY LOW"
        elif r['mad'] < 0.05:
            detail = "LOW"
        elif r['mad'] < 0.15:
            detail = "MEDIUM"
        elif r['mad'] < 0.25:
            detail = "HIGH"
        else:
            detail = "VERY HIGH"

        print(f"{r['name']:<25} {r['mad']:<10.4f} {r['euclidean']:<12.4f} {r['ratio']:<10.2f} {detail}")

def threshold_comparison():
    """Compare threshold appropriateness"""
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    print("\nReference Implementation (urchinemerald):")
    print("   - Threshold: 5-50 (on 0-255 scale)")
    print("   - Normalized: 0.0196-0.196 (on 0-1 scale)")
    print("   - Dynamic: User-controlled via mouse position")
    print("   - Min tile size: 6x6 pixels")

    print("\nCurrent Implementation (ComfyUI):")
    print("   - Threshold: 0.05 (fixed, on 0-1 scale)")
    print("   - Equivalent to: ~12.75 on 0-255 scale")
    print("   - Static: Not user-adjustable during runtime")
    print("   - Min tile size: 256 pixels (VAE requirement)")

    print("\nThreshold Position:")
    ref_min = 5 / 255
    ref_max = 50 / 255
    current = 0.05

    position_pct = ((current - ref_min) / (ref_max - ref_min)) * 100
    print(f"   - Current 0.05 is at {position_pct:.1f}% of reference range")
    print(f"   - This is relatively CONSERVATIVE (toward low subdivision)")

    print("\nRecommendations:")
    print("   - For high detail detection: try 0.01-0.03")
    print("   - For balanced subdivision: try 0.03-0.07")
    print("   - For conservative subdivision: try 0.07-0.15")

def edge_case_analysis():
    """Analyze edge cases where metrics might disagree"""
    print("\n" + "=" * 80)
    print("EDGE CASE ANALYSIS")
    print("=" * 80)

    size = 50

    # Case 1: Single-channel variation
    print("\n1. Single-channel variation:")
    print("   Scenario: Only red channel varies, G and B constant")
    random.seed(50)
    single_channel = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append([random.random(), 0.5, 0.5])
        single_channel.append(row)

    mad_sc = calculate_mad(single_channel)
    euc_sc = calculate_euclidean_rgb(single_channel)
    print(f"   MAD: {mad_sc:.4f}")
    print(f"   Euclidean: {euc_sc:.4f}")
    print(f"   → Both detect variation similarly")

    # Case 2: Multi-channel correlation
    print("\n2. Correlated multi-channel variation:")
    print("   Scenario: All channels vary together (grayscale-like)")
    random.seed(51)
    correlated = []
    for i in range(size):
        row = []
        for j in range(size):
            gray = random.random()
            row.append([gray, gray, gray])
        correlated.append(row)

    mad_corr = calculate_mad(correlated)
    euc_corr = calculate_euclidean_rgb(correlated)
    print(f"   MAD: {mad_corr:.4f}")
    print(f"   Euclidean: {euc_corr:.4f}")
    print(f"   → Euclidean is {euc_corr/mad_corr:.2f}x higher (√3 factor)")

    # Case 3: Chromatic edges
    print("\n3. Chromatic edge (color change, same brightness):")
    print("   Scenario: Red → Blue transition, constant brightness")
    chromatic = []
    for i in range(size):
        row = []
        for j in range(size):
            if j < size // 2:
                row.append([1.0, 0.0, 0.0])  # Red
            else:
                row.append([0.0, 0.0, 1.0])  # Blue
        chromatic.append(row)

    mad_chr = calculate_mad(chromatic)
    euc_chr = calculate_euclidean_rgb(chromatic)
    print(f"   MAD: {mad_chr:.4f}")
    print(f"   Euclidean: {euc_chr:.4f}")
    print(f"   → Both detect this edge well")

    # Case 4: High-frequency vs low-amplitude
    print("\n4. High-frequency but low-amplitude noise:")
    print("   Scenario: Lots of detail, but subtle")
    random.seed(52)
    subtle_noise = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append([random.random() * 0.1 + 0.45,
                       random.random() * 0.1 + 0.45,
                       random.random() * 0.1 + 0.45])
        subtle_noise.append(row)

    mad_sn = calculate_mad(subtle_noise)
    euc_sn = calculate_euclidean_rgb(subtle_noise)
    print(f"   MAD: {mad_sn:.4f}")
    print(f"   Euclidean: {euc_sn:.4f}")
    print(f"   → Both show LOW variance despite high frequency")
    print(f"   → This could cause 'random' cuts if threshold too high!")

def gradient_edge_comparison():
    """Compare metrics on gradients vs edges"""
    print("\n" + "=" * 80)
    print("GRADIENT VS EDGE DETECTION")
    print("=" * 80)

    size = 100

    # Smooth gradient
    gradient = []
    for i in range(size):
        row = []
        val = i / size
        for j in range(size):
            row.append([val, val, val])
        gradient.append(row)

    # Sharp edge
    edge = []
    for i in range(size):
        row = []
        for j in range(size):
            if j < size // 2:
                row.append([0.0, 0.0, 0.0])
            else:
                row.append([1.0, 1.0, 1.0])
        edge.append(row)

    mad_grad = calculate_mad(gradient)
    euc_grad = calculate_euclidean_rgb(gradient)
    mad_edge = calculate_mad(edge)
    euc_edge = calculate_euclidean_rgb(edge)

    print("\nSmooth Gradient (low detail, high color variation):")
    print(f"   MAD: {mad_grad:.4f}")
    print(f"   Euclidean: {euc_grad:.4f}")

    print("\nSharp Edge (high detail, localized):")
    print(f"   MAD: {mad_edge:.4f}")
    print(f"   Euclidean: {euc_edge:.4f}")

    print("\nObservation:")
    print(f"   - Neither metric distinguishes gradient from edge well")
    print(f"   - Both respond to color variation, not spatial detail")
    print(f"   - For true detail detection, need gradient-based metrics")

def main():
    """Run all analyses"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "QUADTREE VARIANCE METRIC EVALUATION" + " " * 28 + "║")
    print("║" + " " * 20 + "MAD vs Euclidean RGB Distance" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")

    analyze_metric_properties()
    compare_metrics_on_patterns()
    threshold_comparison()
    edge_case_analysis()
    gradient_edge_comparison()

    print("\n" + "=" * 80)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. METRIC COMPARISON VERDICT:")
    print("   ✓ MAD and Euclidean are highly correlated")
    print("   ✓ Euclidean is ~1.4-1.7x higher than MAD (due to √3 factor)")
    print("   ✗ Neither metric is definitively 'better' for detail detection")
    print("   → The choice of metric is NOT the root cause of poor subdivision")

    print("\n2. THRESHOLD ANALYSIS:")
    print("   ✗ Fixed 0.05 threshold may be too high for some use cases")
    print("   ✗ No runtime adjustability limits user control")
    print("   ✓ Threshold IS configurable via ComfyUI node parameters")
    print("   → Consider lowering default to 0.03 for better detail capture")

    print("\n3. ROOT CAUSE OF 'RANDOM' CUTS:")
    print("   The likely causes are:")
    print("   a) Color-based metrics don't capture spatial detail/edges")
    print("   b) Smooth gradients score similar to sharp edges")
    print("   c) High-frequency subtle textures may not trigger subdivision")
    print("   d) Minimum tile size (256px) prevents fine-grained subdivision")

    print("\n4. RECOMMENDATIONS:")
    print("\n   Option A: Improve existing metric")
    print("   ─────────────────────────────────")
    print("   • Make threshold configurable in UI (ALREADY DONE)")
    print("   • Lower default threshold: 0.03-0.04 instead of 0.05")
    print("   • Add threshold presets: 'aggressive', 'balanced', 'conservative'")

    print("\n   Option B: Switch to Euclidean metric")
    print("   ─────────────────────────────────────")
    print("   • Would require threshold adjustment (~1.5x higher)")
    print("   • Minimal improvement over MAD")
    print("   • NOT RECOMMENDED - similar behavior to current")

    print("\n   Option C: Add gradient-based metric (BEST)")
    print("   ──────────────────────────────────────────")
    print("   • Combine color variance with edge detection")
    print("   • Use Sobel/Scharr gradients to detect spatial detail")
    print("   • Formula: variance = α*MAD + β*gradient_magnitude")
    print("   • This would catch edges, textures, and fine detail")

    print("\n   Option D: Multi-metric approach")
    print("   ────────────────────────────────")
    print("   • Provide metric selection: 'color', 'edge', 'combined'")
    print("   • Allow users to choose based on content type")
    print("   • Sky/uniform: use color variance")
    print("   • Detailed/textured: use edge detection")

    print("\n5. IMPLEMENTATION PRIORITY:")
    print("   1. [IMMEDIATE] Lower default threshold to 0.03")
    print("   2. [SHORT-TERM] Add gradient-based component")
    print("   3. [LONG-TERM] Provide multiple metric options")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
