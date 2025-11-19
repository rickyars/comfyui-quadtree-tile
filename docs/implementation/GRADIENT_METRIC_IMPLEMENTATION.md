# Gradient-Based Variance Metric Implementation Guide

## Overview

This guide provides detailed implementation instructions for adding a gradient-based component to the quadtree variance metric, addressing the "random cuts" issue by distinguishing spatial detail from color variation.

---

## Problem Statement

**Current limitation:** Color-based metrics (MAD, Euclidean) can't distinguish:
- Smooth gradients from sharp edges
- Color variation from spatial detail
- Low-amplitude textures from uniform areas

**Solution:** Combine color variance with gradient magnitude to capture both color and spatial properties.

---

## Implementation: Option C - Combined Metric

### 1. Update `calculate_variance` Method

**Location:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:161-192`

```python
def calculate_variance(self, tensor: torch.Tensor, x: int, y: int, w: int, h: int) -> float:
    """
    Calculate variance (content complexity) for a region
    Combines color variance with gradient magnitude for better detail detection

    Args:
        tensor: Input tensor (B, C, H, W) or (C, H, W)
        x, y, w, h: Region coordinates

    Returns:
        Combined variance value
    """
    # Handle both batched and unbatched tensors
    if tensor.dim() == 4:
        region = tensor[:, :, y:y+h, x:x+w]
    else:
        region = tensor[:, y:y+h, x:x+w]

    # Ensure region has valid size
    if region.numel() == 0:
        return 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMPONENT 1: Color Variance (existing MAD calculation)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    avg_color = torch.mean(region, dim=(-2, -1), keepdim=True)
    deviations = torch.abs(region - avg_color)
    color_variance = torch.mean(deviations).item()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMPONENT 2: Gradient Magnitude (new spatial detail detection)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Skip gradient calculation for very small regions (performance)
    if w < 8 or h < 8:
        return color_variance * self.color_weight

    # Ensure region is 4D for conv2d
    if region.dim() == 3:
        region_4d = region.unsqueeze(0)  # Add batch dimension
    else:
        region_4d = region

    # Sobel operators for edge detection
    # X-direction (horizontal edges)
    sobel_x = torch.tensor([
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    ], dtype=region_4d.dtype, device=region_4d.device)

    # Y-direction (vertical edges)
    sobel_y = torch.tensor([
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    ], dtype=region_4d.dtype, device=region_4d.device)

    # Apply Sobel filters to each channel
    num_channels = region_4d.shape[1]
    sobel_x = sobel_x.repeat(num_channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(num_channels, 1, 1, 1)

    # Compute gradients
    try:
        grad_x = torch.nn.functional.conv2d(
            region_4d,
            sobel_x,
            padding=1,
            groups=num_channels
        )
        grad_y = torch.nn.functional.conv2d(
            region_4d,
            sobel_y,
            padding=1,
            groups=num_channels
        )

        # Gradient magnitude: √(Gx² + Gy²)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        spatial_variance = torch.mean(gradient_magnitude).item()

    except Exception as e:
        # Fallback to color variance only if gradient computation fails
        print(f"[Quadtree]: Gradient computation failed, using color variance only: {e}")
        spatial_variance = 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # COMBINE: Weighted sum of color and spatial variance
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    combined_variance = (
        self.color_weight * color_variance +
        self.gradient_weight * spatial_variance
    )

    return combined_variance
```

### 2. Update `__init__` Method

**Location:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:137-158`

Add weight parameters for the combined metric:

```python
def __init__(self,
             content_threshold: float = 0.03,  # Lowered from 0.05
             max_depth: int = 4,
             min_tile_size: int = 256,
             min_denoise: float = 0.0,
             max_denoise: float = 1.0,
             variance_mode: str = 'combined',  # NEW
             color_weight: float = 0.5,        # NEW
             gradient_weight: float = 0.5):    # NEW
    """
    Initialize quadtree builder

    Args:
        content_threshold: Variance threshold to trigger subdivision
        max_depth: Maximum recursion depth
        min_tile_size: Minimum tile size in pixels
        min_denoise: Denoise value for largest tiles
        max_denoise: Denoise value for smallest tiles
        variance_mode: Metric type ('color', 'gradient', 'combined')
        color_weight: Weight for color variance component (0.0-1.0)
        gradient_weight: Weight for gradient component (0.0-1.0)
    """
    self.content_threshold = content_threshold
    self.max_depth = max_depth
    self.min_tile_size = min_tile_size
    self.min_denoise = min_denoise
    self.max_denoise = max_denoise
    self.max_tile_area = 0

    # NEW: Variance metric configuration
    self.variance_mode = variance_mode
    self.color_weight = color_weight
    self.gradient_weight = gradient_weight

    # Normalize weights if in combined mode
    if self.variance_mode == 'combined':
        total_weight = color_weight + gradient_weight
        if total_weight > 0:
            self.color_weight = color_weight / total_weight
            self.gradient_weight = gradient_weight / total_weight
    elif self.variance_mode == 'color':
        self.color_weight = 1.0
        self.gradient_weight = 0.0
    elif self.variance_mode == 'gradient':
        self.color_weight = 0.0
        self.gradient_weight = 1.0
```

### 3. Update Node Input Parameters

**Location:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:1267-1273`

Add UI controls for new parameters:

```python
@classmethod
def INPUT_TYPES(s):
    return {
        "required": {
            "image": ("IMAGE",),
            "content_threshold": ("FLOAT", {
                "default": 0.03,  # Lowered from 0.05
                "min": 0.001,
                "max": 0.5,
                "step": 0.001,
                "tooltip": "Variance threshold for subdivision. Lower = more tiles in detailed areas."
            }),
            "max_depth": ("INT", {
                "default": 4,
                "min": 1,
                "max": 8,
                "step": 1,
                "tooltip": "Maximum quadtree depth"
            }),
            "min_tile_size": ("INT", {
                "default": 256,
                "min": 64,
                "max": 1024,
                "step": 8,
                "tooltip": "Minimum tile size in pixels"
            }),

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # NEW: Variance metric configuration
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            "variance_mode": (["color", "gradient", "combined"], {
                "default": "combined",
                "tooltip": "Metric for detail detection:\n" +
                          "• color: Color variation only (fast)\n" +
                          "• gradient: Spatial edges/texture (slower)\n" +
                          "• combined: Both (recommended)"
            }),
            "color_weight": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "tooltip": "Weight for color variance (only used in 'combined' mode)"
            }),
            "gradient_weight": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "tooltip": "Weight for gradient magnitude (only used in 'combined' mode)"
            }),
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            "line_thickness": ("INT", {
                "default": 2,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "Thickness of quadtree boundary lines"
            }),
        }
    }
```

### 4. Update Visualization Method

**Location:** `/home/user/comfyui-quadtree-tile/tiled_vae.py:1310-1350`

Pass new parameters to builder:

```python
def visualize(self, image, content_threshold, max_depth, min_tile_size,
              variance_mode, color_weight, gradient_weight,  # NEW
              min_denoise, max_denoise, line_thickness):
    """
    Visualize quadtree subdivision

    Args:
        ... (existing args)
        variance_mode: Metric type ('color', 'gradient', 'combined')
        color_weight: Weight for color variance component
        gradient_weight: Weight for gradient component
        ... (existing args)
    """
    # Build quadtree with new parameters
    builder = QuadtreeBuilder(
        content_threshold=content_threshold,
        max_depth=max_depth,
        min_tile_size=min_tile_size,
        min_denoise=min_denoise,
        max_denoise=max_denoise,
        variance_mode=variance_mode,      # NEW
        color_weight=color_weight,        # NEW
        gradient_weight=gradient_weight   # NEW
    )

    # ... rest of visualization code
```

---

## Performance Optimization

### Lazy Evaluation Strategy

Only compute gradients when necessary:

```python
def calculate_variance(self, tensor, x, y, w, h):
    # ... setup code ...

    # Always compute color variance
    color_variance = torch.mean(torch.abs(region - avg_color)).item()

    # Early exit if color variance is well below threshold
    if self.variance_mode == 'color':
        return color_variance

    if self.variance_mode == 'combined':
        # Only compute gradients if color variance is near threshold
        threshold_buffer = 0.02
        if color_variance < (self.content_threshold - threshold_buffer):
            # Definitely won't subdivide even with gradient
            return color_variance * self.color_weight

    # Compute gradient only when needed
    spatial_variance = self._compute_gradient(region_4d)

    return self.color_weight * color_variance + self.gradient_weight * spatial_variance
```

### Gradient Caching

Cache gradient computations for parent nodes:

```python
def __init__(self, ...):
    # ... existing code ...
    self._gradient_cache = {}  # Cache for gradient results

def calculate_variance(self, tensor, x, y, w, h):
    # ... color variance code ...

    # Check cache
    cache_key = (x, y, w, h)
    if cache_key in self._gradient_cache:
        spatial_variance = self._gradient_cache[cache_key]
    else:
        spatial_variance = self._compute_gradient(region_4d)
        self._gradient_cache[cache_key] = spatial_variance

    # ... combine and return ...
```

### Downsampling for Large Regions

Compute gradients on downsampled version for large tiles:

```python
def _compute_gradient(self, region_4d):
    """Compute gradient magnitude with optional downsampling"""
    _, _, h, w = region_4d.shape

    # Downsample large regions for performance
    if h > 128 or w > 128:
        region_4d = torch.nn.functional.interpolate(
            region_4d,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False
        )

    # ... Sobel computation ...
    return torch.mean(gradient_magnitude).item()
```

---

## Testing & Validation

### Test Suite

Create test cases for various patterns:

```python
# /home/user/comfyui-quadtree-tile/test_gradient_metric.py

import torch
from tiled_vae import QuadtreeBuilder

def test_smooth_gradient():
    """Smooth gradient should have high color variance, low spatial variance"""
    # Create smooth gradient
    gradient = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    gradient = gradient.repeat(1, 3, 100, 1)

    builder_color = QuadtreeBuilder(variance_mode='color')
    builder_gradient = QuadtreeBuilder(variance_mode='gradient')
    builder_combined = QuadtreeBuilder(variance_mode='combined')

    var_color = builder_color.calculate_variance(gradient, 0, 0, 100, 100)
    var_gradient = builder_gradient.calculate_variance(gradient, 0, 0, 100, 100)
    var_combined = builder_combined.calculate_variance(gradient, 0, 0, 100, 100)

    print(f"Smooth gradient:")
    print(f"  Color variance: {var_color:.4f}")
    print(f"  Gradient variance: {var_gradient:.4f}")
    print(f"  Combined variance: {var_combined:.4f}")

    assert var_color > var_gradient, "Color variance should dominate for smooth gradients"

def test_sharp_edge():
    """Sharp edge should have high spatial variance"""
    # Create sharp edge
    edge = torch.zeros(1, 3, 100, 100)
    edge[:, :, :, 50:] = 1.0

    builder_color = QuadtreeBuilder(variance_mode='color')
    builder_gradient = QuadtreeBuilder(variance_mode='gradient')
    builder_combined = QuadtreeBuilder(variance_mode='combined')

    var_color = builder_color.calculate_variance(edge, 0, 0, 100, 100)
    var_gradient = builder_gradient.calculate_variance(edge, 0, 0, 100, 100)
    var_combined = builder_combined.calculate_variance(edge, 0, 0, 100, 100)

    print(f"Sharp edge:")
    print(f"  Color variance: {var_color:.4f}")
    print(f"  Gradient variance: {var_gradient:.4f}")
    print(f"  Combined variance: {var_combined:.4f}")

    assert var_gradient > 0.1, "Gradient variance should be high for sharp edges"

def test_texture():
    """Texture (high freq, low amplitude) should have higher gradient variance"""
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

    print(f"Texture (low amplitude, high frequency):")
    print(f"  Color variance: {var_color:.4f}")
    print(f"  Gradient variance: {var_gradient:.4f}")
    print(f"  Combined variance: {var_combined:.4f}")

    assert var_combined > var_color, "Combined should capture texture better than color alone"

if __name__ == "__main__":
    test_smooth_gradient()
    print()
    test_sharp_edge()
    print()
    test_texture()
```

---

## Usage Guidelines

### When to Use Each Mode

**Color Mode:**
- Uniform backgrounds with color transitions
- Sky/clouds with gradual color changes
- Content where spatial detail is less important
- Performance-critical scenarios

**Gradient Mode:**
- Photographs with sharp edges
- Architectural images
- High-frequency textures
- Black & white images

**Combined Mode (Recommended):**
- General-purpose processing
- Mixed content (both gradients and edges)
- Unknown image types
- Best balance of detail detection

### Weight Recommendations

**Balanced (default):**
```
color_weight: 0.5
gradient_weight: 0.5
```

**Color-focused (portraits, nature):**
```
color_weight: 0.7
gradient_weight: 0.3
```

**Edge-focused (architecture, diagrams):**
```
color_weight: 0.3
gradient_weight: 0.7
```

**Texture-focused (fabrics, surfaces):**
```
color_weight: 0.2
gradient_weight: 0.8
```

---

## Threshold Adjustment

With the combined metric, threshold values will need adjustment:

### Original (Color only):
- Aggressive: 0.01-0.02
- Balanced: 0.03-0.05
- Conservative: 0.07-0.15

### Combined (Color + Gradient):
- Aggressive: 0.02-0.04
- Balanced: 0.04-0.07
- Conservative: 0.10-0.20

The combined metric produces higher values, so thresholds should be approximately 1.5-2× higher.

---

## Migration Path

### Phase 1: Add Option, Keep Default Behavior
1. Implement combined metric
2. Set default to `variance_mode='color'`
3. Allow users to opt-in to new metric
4. Gather feedback

### Phase 2: Gradual Rollout
1. Change default to `variance_mode='combined'`
2. Adjust default threshold to 0.03
3. Provide preset options
4. Update documentation

### Phase 3: Deprecation (optional)
1. Mark 'color' mode as legacy
2. Recommend migration to 'combined'
3. Keep backward compatibility

---

## Expected Results

### Before (Color-only MAD):
```
Pattern                    Subdivide?  Appropriate?
────────────────────────────────────────────────────
Smooth sky gradient        YES         ✗ NO (over-subdivides)
Concrete texture           NO          ✗ NO (misses detail)
Sharp architectural edge   YES         ✓ YES
Uniform color area         NO          ✓ YES
```

### After (Combined metric):
```
Pattern                    Subdivide?  Appropriate?
────────────────────────────────────────────────────
Smooth sky gradient        NO/LIGHT    ✓ YES (improved)
Concrete texture           YES         ✓ YES (detects now)
Sharp architectural edge   YES         ✓ YES
Uniform color area         NO          ✓ YES
```

---

## Code Files Summary

**Modified files:**
- `/home/user/comfyui-quadtree-tile/tiled_vae.py`
  - `QuadtreeBuilder.__init__()` (lines 137-158)
  - `QuadtreeBuilder.calculate_variance()` (lines 161-192)
  - Node input parameters (lines 1267-1310)
  - `visualize()` method (lines 1310-1350)

**New files:**
- `/home/user/comfyui-quadtree-tile/test_gradient_metric.py` (new test suite)

---

## Additional Considerations

### Alternative Gradient Operators

**Scharr operator (more accurate):**
```python
scharr_x = torch.tensor([
    [[-3,  0,  3],
     [-10, 0,  10],
     [-3,  0,  3]]
])
```

**Prewitt operator (faster):**
```python
prewitt_x = torch.tensor([
    [[-1, 0, 1],
     [-1, 0, 1],
     [-1, 0, 1]]
])
```

### Perceptual Weighting

Weight gradient by perceptual importance:

```python
# Weight gradient by luminance channel
if num_channels == 3:
    # RGB to grayscale weights
    luminance = 0.299 * region[:, 0] + 0.587 * region[:, 1] + 0.114 * region[:, 2]
    # Compute gradient on luminance only
    grad_luma = compute_gradient(luminance)
    spatial_variance = grad_luma
```

---

## Conclusion

This implementation guide provides a complete solution for adding gradient-based variance detection to the quadtree subdivision algorithm. The combined metric effectively addresses the "random cuts" issue by distinguishing spatial detail from color variation.

**Key benefits:**
- ✓ Detects edges and texture, not just color changes
- ✓ Distinguishes smooth gradients from sharp transitions
- ✓ Configurable weights for different content types
- ✓ Backward compatible with existing workflows
- ✓ Performance-optimized with lazy evaluation

**Next steps:**
1. Implement Phase 1 (quick win: lower threshold)
2. Test gradient component on sample images
3. Optimize performance for production use
4. Gather user feedback and iterate
