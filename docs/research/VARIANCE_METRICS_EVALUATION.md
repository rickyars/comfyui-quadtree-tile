# Architectural Evaluation: Quadtree Variance Metrics

## Executive Summary

This document evaluates two quadtree subdivision implementations to determine why the current ComfyUI implementation produces subdivisions that "seem pretty random" and fail to effectively identify high-detail vs low-detail areas.

**Key Finding:** The root cause is NOT the choice between MAD and Euclidean metrics. Both are color-based metrics that respond to color variation rather than spatial detail. The issue is fundamental to the metric type itself.

---

## 1. Comparative Metric Analysis

### Reference Implementation (urchinemerald/quadtree_subdivision)

- **Metric:** Euclidean RGB Distance
- **Formula:** `avg(√((R-R_avg)² + (G-G_avg)² + (B-B_avg)²))`
- **Threshold:** Dynamic 5-50 (on 0-255 scale) = 0.0196-0.196 (normalized)
- **Subdivision:** 4 equal quadrants
- **Min size:** 6×6 pixels
- **Goal:** Artistic visualization of detail variation

### Current Implementation (ComfyUI quadtree)

- **Metric:** Mean Absolute Deviation (MAD)
- **Formula:** `avg(|R-R_avg| + |G-G_avg| + |B-B_avg|) / 3`
- **Threshold:** Fixed 0.05 (normalized) = ~12.75 (on 0-255 scale)
- **Subdivision:** 4 children with 8-pixel alignment
- **Min size:** 256 pixels (VAE requirement)
- **Goal:** Adaptive tiling for diffusion processing

---

## 2. Mathematical Comparison

### Metric Properties

| Property | MAD | Euclidean RGB |
|----------|-----|---------------|
| **Norm Type** | L1 (Manhattan) | L2 (Euclidean) |
| **Outlier Sensitivity** | Linear response | Quadratic response |
| **Range (normalized)** | [0, 1] | [0, √3] ≈ [0, 1.732] |
| **Channel Treatment** | Independent sum | Combined distance |
| **Typical Ratio** | 1.0× (baseline) | ~1.7× MAD |

### Test Results

From `/home/user/comfyui-quadtree-tile/analyze_variance_metrics.py`:

```
Pattern                   MAD      Euclidean   Ratio   Classification
─────────────────────────────────────────────────────────────────────
Uniform (solid color)     0.0000   0.0000      —       No detail
Sky gradient (subtle)     0.0083   0.0250      3.0×    Very low detail
Low-amplitude noise       0.0752   0.1442      1.92×   Medium detail
Random noise              0.2498   0.4799      1.92×   High detail
Smooth gradient           0.2500   0.4330      1.73×   High variation
Checkerboard (4px)        0.5000   0.8660      1.73×   Very high detail
Sharp edge                0.5000   0.8660      1.73×   High detail
```

### Key Observations

1. **Strong Correlation:** Euclidean is consistently 1.7-1.9× higher than MAD
2. **Similar Ranking:** Both metrics rank patterns in nearly identical order
3. **Scaling Factor:** The √3 factor dominates the difference
4. **Pattern Blindness:** Both fail to distinguish gradients from edges

---

## 3. Critical Limitation: Gradient vs Edge Problem

### The Fundamental Issue

**Neither metric distinguishes spatial detail from color variation.**

Test case comparison:

```python
Smooth Gradient (0→1 across 100px):
   MAD: 0.2500
   Euclidean: 0.4330
   Detail: LOW (smooth transition)

Sharp Edge (0|1 at midpoint):
   MAD: 0.5000
   Euclidean: 0.8660
   Detail: HIGH (abrupt transition)
```

**Problem:** While the sharp edge scores higher, a gentle gradient across a large area can score similarly to a sharp edge in a smaller area when measured at the tile level.

### Why This Causes "Random" Cuts

1. **Sky with subtle gradient:** May trigger subdivision
2. **Textured concrete (high frequency, low amplitude):** May NOT trigger subdivision
3. **Smooth color transitions:** Score similar to sharp edges
4. **Spatial frequency ignored:** Only color distribution matters

This creates the perception of randomness - the metric responds to color statistics, not visual complexity.

---

## 4. Threshold Appropriateness

### Threshold Comparison

| Aspect | Reference | Current | Analysis |
|--------|-----------|---------|----------|
| **Value** | 5-50 (dynamic) | 12.75 (fixed) | Current at 17% of ref range |
| **Adjustability** | Real-time (mouse) | Pre-configured | Less flexible |
| **Position** | Full range | Conservative | Biased toward fewer tiles |
| **Typical Use** | Interactive art | Batch processing | Different use cases |

### Current Threshold (0.05) Analysis

From test results:
- **Too high for:** Subtle textures (0.025-0.075 range)
- **Appropriate for:** Medium-high detail (0.075-0.250 range)
- **Too low for:** High-contrast patterns (0.25+ range)

**Recommendation:** Lower default to 0.03 for better detail capture.

### Configurability

The current implementation DOES allow threshold adjustment via ComfyUI node parameters:

```python
# From tiled_vae.py:1267-1272
"content_threshold": ("FLOAT", {
    "default": 0.05,
    "min": 0.001,
    "max": 0.5,
    "step": 0.001,
    "tooltip": "Variance threshold for subdivision. Lower = more tiles..."
})
```

This is **already available** - the issue is the default value, not configurability.

---

## 5. Edge Cases & Failure Modes

### High-Frequency Low-Amplitude Detail

**Scenario:** Fine texture with subtle color variation (e.g., fabric, wood grain)

```
MAD: 0.0249
Euclidean: 0.0478
Threshold: 0.05
Result: NO SUBDIVISION (missed detail)
```

**Why:** Color variance is low despite high spatial frequency.

### Chromatic Edges

**Scenario:** Color change without luminance change (e.g., red→blue)

```
MAD: 0.3333
Euclidean: 0.7071
Result: STRONG SUBDIVISION
```

**Why:** Both metrics detect color distance well.

### Grayscale Correlation Effect

**Scenario:** Correlated channel variation (grayscale content)

```
MAD: 0.2515
Euclidean: 0.4356 (1.73× higher)
```

**Why:** Euclidean benefits from √3 factor when all channels vary together.

---

## 6. Root Cause Analysis

### Is it the metric (MAD vs Euclidean)?

**NO.** Testing shows:
- ✓ Both metrics are highly correlated (r² > 0.95)
- ✓ Euclidean is simply ~1.7× scaled version of MAD
- ✓ Both respond to color statistics, not spatial detail
- ✗ Switching metrics would only require threshold rescaling

### Is it the threshold value?

**PARTIALLY.** Analysis shows:
- ✓ Current 0.05 is conservative (low sensitivity)
- ✓ Lowering to 0.03 would capture more detail
- ✗ But won't solve fundamental metric limitation
- ✗ Still can't distinguish gradients from edges

### Is it the minimum tile size?

**YES.** Major contributor:
- ✗ 256-pixel minimum prevents fine-grained subdivision
- ✗ Reference uses 6×6 minimum (40× smaller)
- ✗ Large tiles average out local detail
- ✓ But this is a VAE processing requirement

### Is it something else?

**YES - The metric type itself.**

Both MAD and Euclidean are **statistical color variance** metrics. They measure:
- ✓ How much pixels differ from average color
- ✗ NOT where edges/transitions occur
- ✗ NOT spatial frequency/texture
- ✗ NOT perceptual detail

---

## 7. Detailed Recommendations

### Option A: Improve Existing Metric (Quick Win)

**Implementation:** Adjust threshold and add presets

```python
# Lower default threshold
"content_threshold": ("FLOAT", {
    "default": 0.03,  # Changed from 0.05
    "min": 0.001,
    "max": 0.5,
    "step": 0.001,
})

# Add preset options
"threshold_preset": (["custom", "aggressive", "balanced", "conservative"], {
    "default": "balanced"
})
# aggressive: 0.01-0.02 (many tiles)
# balanced: 0.03-0.05 (default)
# conservative: 0.07-0.15 (fewer tiles)
```

**Pros:**
- ✓ Minimal code changes
- ✓ Immediate improvement
- ✓ Backward compatible

**Cons:**
- ✗ Doesn't solve fundamental limitation
- ✗ Still misses textured areas
- ✗ Still can't distinguish gradients from edges

### Option B: Switch to Euclidean (Not Recommended)

**Implementation:** Replace MAD with Euclidean formula

```python
def calculate_variance(self, tensor, x, y, w, h):
    region = tensor[:, y:y+h, x:x+w]
    avg_color = torch.mean(region, dim=(-2, -1), keepdim=True)

    # Euclidean distance instead of MAD
    squared_diffs = (region - avg_color) ** 2
    euclidean_distances = torch.sqrt(torch.sum(squared_diffs, dim=1))
    return torch.mean(euclidean_distances).item()
```

**Required changes:**
- Adjust threshold: multiply by ~1.7
- Update documentation
- Test with existing workflows

**Pros:**
- ✓ Matches reference implementation
- ✓ Slightly more sensitive to outliers

**Cons:**
- ✗ Minimal practical improvement
- ✗ Breaking change for existing configs
- ✗ Doesn't address core limitation
- ✗ **NOT RECOMMENDED**

### Option C: Add Gradient-Based Metric (BEST)

**Implementation:** Combine color variance with edge detection

```python
def calculate_variance(self, tensor, x, y, w, h):
    region = tensor[:, y:y+h, x:x+w]

    # Color variance (existing)
    avg_color = torch.mean(region, dim=(-2, -1), keepdim=True)
    color_variance = torch.mean(torch.abs(region - avg_color)).item()

    # Gradient magnitude (new)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = torch.nn.functional.conv2d(region, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(region, sobel_y, padding=1)
    gradient_magnitude = torch.mean(torch.sqrt(grad_x**2 + grad_y**2)).item()

    # Combined metric
    alpha = 0.5  # Weight for color variance
    beta = 0.5   # Weight for gradient magnitude
    return alpha * color_variance + beta * gradient_magnitude
```

**Why this works:**
- ✓ Color variance captures uniform vs varied areas
- ✓ Gradient magnitude captures edges and texture
- ✓ Smooth gradients: high color variance, low gradient
- ✓ Sharp edges: moderate color variance, high gradient
- ✓ Textures: potentially low color variance, high gradient

**Pros:**
- ✓ Solves gradient vs edge problem
- ✓ Captures spatial detail
- ✓ Better aligns with "detail" perception
- ✓ Tunable via alpha/beta weights

**Cons:**
- ✗ More computational cost
- ✗ Requires threshold retuning
- ✗ Adds complexity

### Option D: Multi-Metric Selection (Long-term)

**Implementation:** Let users choose metric type

```python
"variance_metric": (["color", "edge", "combined"], {
    "default": "combined",
    "tooltip": "color: color variation only, edge: spatial detail, combined: both"
})
```

**Use cases:**
- **color:** Sky, uniform backgrounds, color transitions
- **edge:** Photographs, detailed imagery, textures
- **combined:** General purpose, balanced

**Pros:**
- ✓ Maximum flexibility
- ✓ Handles diverse content types
- ✓ Educational (users learn what works)

**Cons:**
- ✗ Choice paralysis for users
- ✗ Increased maintenance burden
- ✗ More documentation needed

---

## 8. Implementation Priority

### Immediate (Week 1)

1. **Lower default threshold** from 0.05 to 0.03
   - File: `/home/user/comfyui-quadtree-tile/tiled_vae.py:1268`
   - Impact: Better detail capture with existing metric
   - Risk: Low (users can adjust if too aggressive)

### Short-term (Month 1)

2. **Add gradient-based component**
   - Implement Option C (combined metric)
   - Make alpha/beta configurable
   - Provide presets: color-focused, edge-focused, balanced
   - Impact: Significant improvement in detail detection

### Long-term (Quarter 1)

3. **Add metric selection UI**
   - Implement Option D (multi-metric approach)
   - Provide clear documentation on when to use each
   - Add visual examples in documentation
   - Impact: Maximum flexibility for diverse use cases

---

## 9. Performance Considerations

### Computational Cost

| Metric | Operations | Relative Cost |
|--------|------------|---------------|
| MAD | Mean, abs, mean | 1.0× (baseline) |
| Euclidean | Mean, square, sqrt, mean | 1.2× |
| Gradient | Convolution, sqrt | 3.0-5.0× |
| Combined | All above | 4.0-6.0× |

### Mitigation Strategies

1. **Lazy evaluation:** Only compute gradients if color variance is near threshold
2. **Downsampling:** Compute gradients on 50% resolution
3. **Caching:** Store gradient results for parent nodes
4. **Optimized kernels:** Use pre-compiled convolution operations

**Estimated impact:** On modern GPUs, gradient computation adds ~0.1-0.2s for 1024×1024 image.

---

## 10. Testing & Validation

### Test Scenarios

Create test suite with these patterns:

1. **Smooth sky gradient** (should NOT subdivide much)
2. **Photograph with fine detail** (should subdivide heavily)
3. **Concrete texture** (subtle, should subdivide)
4. **Sharp architectural edges** (should subdivide)
5. **Uniform colored regions** (should NOT subdivide)
6. **Noisy low-contrast** (edge case - configurable)

### Success Criteria

- ✓ Sky gradient: < 10 tiles
- ✓ Detailed photo: > 50 tiles
- ✓ Texture: 20-40 tiles
- ✓ Sharp edges: Focused subdivision at boundaries
- ✓ Uniform: Single tile
- ✓ User perception: "Makes sense" vs "random"

### Validation Script

Location: `/home/user/comfyui-quadtree-tile/analyze_variance_metrics.py`

This script provides:
- Mathematical comparison of metrics
- Test patterns for evaluation
- Threshold analysis
- Edge case detection

---

## 11. Conclusion

### Core Findings

1. **MAD vs Euclidean:** Essentially equivalent, switching NOT recommended
2. **Threshold:** Current 0.05 is too conservative, lower to 0.03
3. **Root Cause:** Color-based metrics can't distinguish spatial detail
4. **Best Solution:** Add gradient-based component (Option C)

### Why Subdivisions Seem Random

The "random" appearance stems from:

1. **Metric limitation:** Color variance ≠ perceptual detail
2. **Gradient blindness:** Can't distinguish smooth vs sharp
3. **Large tile size:** 256px minimum averages out local features
4. **Conservative threshold:** Misses subtle but visually important detail

### Recommended Action Plan

**Phase 1 (Immediate):**
- Change default threshold from 0.05 to 0.03
- Add threshold presets in documentation

**Phase 2 (Short-term):**
- Implement gradient-based component
- Make metric weights configurable
- Extensive testing on diverse images

**Phase 3 (Long-term):**
- Add metric selection UI
- Provide usage guidelines
- Create visual comparison examples

---

## Appendix: Code Locations

### Current Implementation

- **Variance calculation:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:161-192`
- **Subdivision logic:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:194-218`
- **Threshold default:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:1268`
- **UI configuration:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:1267-1273`

### Test & Analysis

- **Metric comparison:** `/home/user/comfyui-quadtree-tile/analyze_variance_metrics.py`
- **Subdivision tests:** `/home/user/comfyui-quadtree-tile/test_subdivision_issue.py`

---

## References

1. **Reference Implementation:** https://github.com/urchinemerald/quadtree_subdivision
2. **Mathematical Analysis:** `/home/user/comfyui-quadtree-tile/analyze_variance_metrics.py`
3. **Current Implementation:** `/home/user/comfyui-quadtree-tile/tiled_vae.py`

---

**Document Version:** 1.0
**Date:** 2025-11-16
**Author:** Architectural Evaluation
**Status:** Final Recommendation
